#!/usr/bin/env python3
"""
mempalace migrate — Recover a palace created with a different ChromaDB version.

Reads documents and metadata directly from the palace's SQLite database
(bypassing ChromaDB's API, which fails on version-mismatched palaces),
then re-imports everything into a fresh palace using the currently installed
ChromaDB version.

Since mempalace 3.2.0 (chromadb>=1.5.4), chromadb automatically migrates
0.4.1+ databases on first open — no manual migration needed for upgrades.
Use this command only when downgrading chromadb (e.g. rolling back to an
older mempalace release) or if automatic migration fails.

Usage:
    mempalace migrate                          # migrate default palace
    mempalace migrate --palace /path/to/palace  # migrate specific palace
    mempalace migrate --dry-run                # show what would be migrated
"""

import errno
import glob
import os
import shutil
import tempfile
import uuid
from collections import defaultdict
from datetime import datetime

from .backups import copy_palace_dir, prune_backups
from .config import MempalaceConfig


def _restore_stale_palace(palace_path: str, stale_path: str) -> None:
    """Roll back a failed swap.

    shutil.move() can partially create palace_path before raising, which
    would make a bare os.replace(stale_path, palace_path) fail (dest exists).
    Clear any partial destination first, then restore. Best-effort: if the
    restore itself fails, log both paths so the operator can recover by hand.
    """
    try:
        if os.path.lexists(palace_path):
            shutil.rmtree(palace_path, ignore_errors=True)
        os.replace(stale_path, palace_path)
    except Exception as err:
        print(
            f"  CRITICAL: rollback failed — original palace at {stale_path}, "
            f"partial migration data at {palace_path}. Restore manually. "
            f"({err})"
        )


def extract_drawers_from_sqlite(db_path: str) -> list:
    """Read drawer rows from ChromaDB's SQLite, bypassing the API.

    Works regardless of which ChromaDB version created the database.
    Returns list of dicts with 'id', 'document', and 'metadata' keys.

    Only the drawers collection is read. An unscoped join also returns
    closets and assets, and filing those rows into ``mempalace_drawers``
    is how a migrate used to destroy the other collections. The read goes
    through :func:`mempalace.repair.extract_via_sqlite`, which opens the
    database with :func:`mempalace.backends._inproc_sqlite.open_reader` and
    closes it before returning.
    """
    from .repair import _drawers_collection_name, extract_via_sqlite

    palace_path = os.path.dirname(os.path.abspath(db_path))
    drawers = []
    for embedding_id, document, metadata in extract_via_sqlite(
        palace_path, _drawers_collection_name()
    ):
        if not document:
            continue
        drawers.append(
            {
                "id": embedding_id,
                "document": document,
                "metadata": metadata,
            }
        )
    return drawers


def _sibling_collection_rows(palace_path: str) -> dict:
    """Rows in recoverable collections other than drawers.

    A file that is not a SQLite database yields nothing. Callers that are
    exercising the swap with a stand-in ``chroma.sqlite3`` never see a
    schema error from this helper. A real palace that fails to read raises.
    """
    from .backends._magic import has_sqlite_magic
    from .repair import _drawers_collection_name, _recoverable_collections, extract_via_sqlite

    db_path = os.path.join(palace_path, "chroma.sqlite3")
    if not has_sqlite_magic(db_path):
        return {}

    drawer_name = _drawers_collection_name()
    found = {}
    for name in _recoverable_collections(palace_path):
        if name == drawer_name:
            continue
        rows = []
        for embedding_id, document, metadata in extract_via_sqlite(palace_path, name):
            if not document:
                continue
            rows.append(
                {
                    "id": embedding_id,
                    "document": document,
                    "metadata": metadata or {},
                }
            )
        if rows:
            found[name] = rows
    return found


def _import_collection_rows(backend, palace_path: str, collection_name: str, rows: list) -> int:
    """Add ``rows`` to ``collection_name`` and return the collection's count."""
    col = backend.get_or_create_collection(palace_path, collection_name)
    batch_size = 500
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        col.add(
            ids=[row["id"] for row in batch],
            documents=[row["document"] for row in batch],
            metadatas=[row["metadata"] for row in batch],
        )
    return col.count()


def _preserve_sidecars(source_palace: str, dest_palace: str) -> None:
    """Copy the knowledge graph and embedder sidecar into ``dest_palace``.

    The rebuilt directory replaces the live palace. Anything not copied
    here is gone after the swap, including the graph and the recorded
    embedder identity.
    """
    from .backends._sidecar import EMBEDDER_SIDECAR_FILENAME
    from .repair import _copy_file_no_follow, _preserve_knowledge_graph_sqlite

    _preserve_knowledge_graph_sqlite(source_palace, dest_palace)
    sidecar = os.path.join(source_palace, EMBEDDER_SIDECAR_FILENAME)
    if not os.path.isfile(sidecar):
        return
    os.makedirs(dest_palace, exist_ok=True)
    _copy_file_no_follow(
        sidecar,
        os.path.join(dest_palace, EMBEDDER_SIDECAR_FILENAME),
        replace=True,
    )


def _counts_match(
    drawer_count: int, drawers: list, sibling_rows: dict, sibling_counts: dict
) -> bool:
    if drawer_count != len(drawers):
        return False
    for name, rows in sibling_rows.items():
        if sibling_counts.get(name) != len(rows):
            return False
    return True


def _unique_aside_path(palace_path: str) -> str:
    """A rollback directory this call creates, never an existing ``.old``.

    ``_restore_stale_palace`` tells the operator the original palace is at
    the aside path. Deleting a pre-existing ``.old`` removes the only copy
    that message can point at.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = f"{palace_path}.old.{stamp}.{os.getpid()}"
    if not os.path.lexists(candidate):
        return candidate
    return f"{candidate}.{uuid.uuid4().hex[:8]}"


def _refuse_swap(
    drawer_count: int, drawers: list, sibling_rows: dict, sibling_counts: dict
) -> None:
    print(
        "  ERROR: verification failed before swap. Refusing to swap; the palace was left unchanged."
    )
    if drawer_count != len(drawers):
        print(f"  ERROR: drawers count {drawer_count} != {len(drawers)} extracted.")
    for name, rows in sibling_rows.items():
        got = sibling_counts.get(name)
        if got != len(rows):
            print(f"  ERROR: {name} count {got} != {len(rows)} extracted.")


def detect_chromadb_version(db_path: str) -> str:
    """Detect which ChromaDB version created the database by checking schema."""
    from .backends._inproc_sqlite import open_reader as open_palace_reader

    conn = open_palace_reader(db_path)
    try:
        # 1.x has schema_str column in collections table
        cols = [r[1] for r in conn.execute("PRAGMA table_info(collections)").fetchall()]
        if "schema_str" in cols:
            return "1.x"
        # 0.6.x has embeddings_queue but no schema_str
        tables = [
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        ]
        if "embeddings_queue" in tables:
            return "0.6.x"
        return "unknown"
    finally:
        conn.close()


def contains_palace_database(path: str) -> bool:
    """Return True when path looks like a MemPalace ChromaDB directory."""
    return os.path.isfile(os.path.join(path, "chroma.sqlite3"))


def confirm_destructive_action(
    operation_name: str, palace_path: str, assume_yes: bool = False
) -> bool:
    """Require confirmation before destructive palace operations."""
    if assume_yes:
        return True

    print(f"\n  {operation_name} will replace data in: {palace_path}")
    print("  A backup will be created first, then the palace will be rebuilt.")
    try:
        answer = input("  Continue? [y/N]: ").strip().lower()
    except EOFError:
        print("  Aborted. Re-run with --yes to confirm destructive changes.")
        return False

    if answer not in {"y", "yes"}:
        print("  Aborted.")
        return False
    return True


def _result_ids(result) -> list:
    """Return ids from either the backend typed result or raw Chroma dict."""

    if isinstance(result, dict):
        return list(result.get("ids") or [])

    return list(getattr(result, "ids", []) or [])


def collection_write_roundtrip_works(col) -> bool:
    """Return True only if the collection can upsert, read, and delete.

    Some ChromaDB 0.6.x -> 1.5.x migrated collections remain readable while
    writes and deletes silently no-op. A plain ``count()`` probe misses that
    failure mode, so migrate must verify an actual write round-trip before
    deciding that no rebuild is needed.
    """

    probe_id = f"_mempalace_migrate_probe_{uuid.uuid4().hex}"
    probe_doc = "mempalace migrate write round-trip probe"
    probe_meta = {
        "wing": "_mempalace_probe",
        "room": "_mempalace_probe",
        "source_file": "mempalace_migrate_probe",
        "chunk_index": 0,
    }

    try:
        col.upsert(
            ids=[probe_id],
            documents=[probe_doc],
            metadatas=[probe_meta],
        )

        after_upsert = col.get(ids=[probe_id], include=[])
        if probe_id not in _result_ids(after_upsert):
            return False

        col.delete(ids=[probe_id])

        after_delete = col.get(ids=[probe_id], include=[])
        if probe_id in _result_ids(after_delete):
            return False

        return True
    except Exception:
        return False


def migrate(palace_path: str, dry_run: bool = False, confirm: bool = False):
    """Migrate a palace to the currently installed ChromaDB version."""
    from .palace import MineAlreadyRunning, mine_palace_lock

    palace_path = os.path.abspath(os.path.expanduser(palace_path))
    db_path = os.path.join(palace_path, "chroma.sqlite3")

    if not os.path.isdir(palace_path) or not contains_palace_database(palace_path):
        print(f"\n  No palace database found at {db_path}")
        return False

    try:
        with mine_palace_lock(palace_path):
            return _migrate_locked(palace_path, dry_run=dry_run, confirm=confirm)
    except MineAlreadyRunning as exc:
        print(f"  Another writer already holds this palace: {exc}")
        print("  Migration stopped before reading or copying anything; nothing changed.")
        raise SystemExit(2) from exc


def _migrate_locked(palace_path: str, dry_run: bool = False, confirm: bool = False):
    """Body of :func:`migrate`, called while holding ``mine_palace_lock``."""
    from .backends.chroma import ChromaBackend

    db_path = os.path.join(palace_path, "chroma.sqlite3")

    print(f"\n{'=' * 60}")
    print("  MemPalace Migrate")
    print(f"{'=' * 60}\n")
    print(f"  Palace:    {palace_path}")
    print(f"  Database:  {db_path}")
    print(f"  DB size:   {os.path.getsize(db_path) / 1024 / 1024:.1f} MB")

    # Detect version
    source_version = detect_chromadb_version(db_path)
    target_version = ChromaBackend.backend_version()
    print(f"  Source:    ChromaDB {source_version}")
    print(f"  Target:    ChromaDB {target_version}")

    # Try reading and writing with current chromadb first.
    #
    # A plain count() is not enough: some 0.6.x -> 1.5.x migrated collections
    # are readable but silently drop upsert/delete operations. In that state,
    # migrate must rebuild from SQLite instead of returning "No migration needed."
    #
    # Preflight HNSW divergence before touching the collection at all: the
    # #1222 SIGSEGV/panic class crashes on count() against a diverged
    # segment, and a native crash can't be caught by the except Exception
    # below -- a diverged palace must never reach col.count() in the first
    # place. The already-diverged case degrades to exactly the same
    # SQLite-extraction path the except branch below already falls back to.
    from .backends.chroma import hnsw_capacity_status

    try:
        capacity_info = hnsw_capacity_status(palace_path, "mempalace_drawers")
    except Exception:
        capacity_info = {}

    if capacity_info.get("diverged"):
        print(
            f"\n Palace is NOT readable by chromadb {target_version}: HNSW index diverged from SQLite."
        )
        print(f" ({capacity_info.get('message', 'divergence detected')})")
        print(" Extracting from SQLite directly...")
    else:
        probe = ChromaBackend()
        try:
            col = probe.get_collection(palace_path, "mempalace_drawers")
            count = col.count()

            if collection_write_roundtrip_works(col):
                print(f"\n Palace is already readable and writable by chromadb {target_version}.")
                print(f" {count} drawers found. No migration needed.")
                return True

            print(
                f"\n Palace is readable by chromadb {target_version}, but write/delete verification failed."
            )
            print(" Rebuilding from SQLite to restore native write/delete behavior...")
        except Exception:
            print(f"\n Palace is NOT readable by chromadb {target_version}.")
            print(" Extracting from SQLite directly...")
        finally:
            # Drop the probe client before the SQLite read. A raw connect
            # beside a live Chroma handle releases Chroma's POSIX lock.
            probe.close()

    # Extract all drawers via raw SQL
    drawers = extract_drawers_from_sqlite(db_path)
    print(f"  Extracted {len(drawers)} drawers from SQLite")

    if not drawers:
        print("  Nothing to migrate.")
        return True

    # Show summary
    wings = defaultdict(lambda: defaultdict(int))
    for d in drawers:
        w = d["metadata"].get("wing", "?")
        r = d["metadata"].get("room", "?")
        wings[w][r] += 1

    print("\n  Summary:")
    for wing, rooms in sorted(wings.items()):
        total = sum(rooms.values())
        print(f"    WING: {wing} ({total} drawers)")
        for room, count in sorted(rooms.items(), key=lambda x: -x[1]):
            print(f"      ROOM: {room:30} {count:5}")

    if dry_run:
        print("\n  DRY RUN -- no changes made.")
        print(f"  Would migrate {len(drawers)} drawers.")
        return True

    if not confirm_destructive_action("Migration", palace_path, assume_yes=confirm):
        return False

    # Backup the old palace
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{palace_path}.pre-migrate.{timestamp}"
    print(f"\n  Backing up to {backup_path}...")
    copy_palace_dir(palace_path, backup_path, symlinks=True, log=print)

    # Enforce backup retention so repeated migrations cannot fill the disk
    # with full-palace copies. The backup we just created is the newest, so
    # it survives; only older ``.pre-migrate.*`` siblings beyond the limit
    # are removed. Best-effort — never let cleanup fail the migration.
    prune_backups(
        glob.escape(palace_path) + ".pre-migrate.*",
        MempalaceConfig().max_backups,
        log=print,
    )

    # Build fresh palace in a temp directory (avoids chromadb reading old state).
    # Wrap the whole import-and-swap dance in try/finally so the temp dir is
    # cleaned up if any of the chromadb writes, the verify count, or the
    # rename fails — without try/finally a crashed migration leaves a partial
    # palace dir under the system temp root that the user has to find by hand.
    temp_palace = tempfile.mkdtemp(prefix="mempalace_migrate_")
    try:
        print(f"  Creating fresh palace in {temp_palace}...")
        fresh_backend = ChromaBackend()
        try:
            col = fresh_backend.get_or_create_collection(temp_palace, "mempalace_drawers")

            # Re-import drawers in batches. Closets and assets are separate
            # collections; they are not added here.
            batch_size = 500
            imported = 0
            for i in range(0, len(drawers), batch_size):
                batch = drawers[i : i + batch_size]
                col.add(
                    ids=[d["id"] for d in batch],
                    documents=[d["document"] for d in batch],
                    metadatas=[d["metadata"] for d in batch],
                )
                imported += len(batch)
                print(f"  Imported {imported}/{len(drawers)} drawers...")

            final_count = col.count()
            sibling_rows = _sibling_collection_rows(palace_path)
            sibling_counts = {
                name: _import_collection_rows(fresh_backend, temp_palace, name, rows)
                for name, rows in sibling_rows.items()
            }
            _preserve_sidecars(palace_path, temp_palace)
            # Compare before the rename. A mismatch after the swap can only
            # warn, and the aside copy is already what a rollback would need.
            if not _counts_match(final_count, drawers, sibling_rows, sibling_counts):
                _refuse_swap(final_count, drawers, sibling_rows, sibling_counts)
                return False
        finally:
            fresh_backend.close()

        # Swap: rename old palace aside, then move new one into place.
        # This avoids a window where both old and new are missing.
        # The aside name is unique so a pre-existing ``.old`` — the path a
        # failed rollback's CRITICAL line points at — is not deleted.
        print("  Swapping old palace for migrated version...")
        stale_path = _unique_aside_path(palace_path)
        os.replace(palace_path, stale_path)
        try:
            os.replace(temp_palace, palace_path)
        except OSError as e:
            # EXDEV = temp lives on a different filesystem; fall back to copy+delete.
            # Anything else is a real error — don't mask it with shutil.move.
            if getattr(e, "errno", None) != errno.EXDEV:
                _restore_stale_palace(palace_path, stale_path)
                raise
            try:
                shutil.move(temp_palace, palace_path)
            except Exception:
                _restore_stale_palace(palace_path, stale_path)
                raise
        # This aside is the one this call just created. The CRITICAL message
        # is only printed when rollback fails, and that path raises before
        # this removal, so the directory the message names is still on disk.
        shutil.rmtree(stale_path, ignore_errors=True)
    finally:
        # On the happy path os.replace/shutil.move consumed temp_palace, so
        # the directory no longer exists at the temp location — the existence
        # guard makes this a no-op then. On any failure path it actually
        # removes the orphan.
        if os.path.exists(temp_palace):
            shutil.rmtree(temp_palace, ignore_errors=True)

    print("\n  Migration complete.")
    print(f"  Drawers migrated: {final_count}")
    print(f"  Backup at: {backup_path}")

    print(f"\n{'=' * 60}\n")
    return True


# ---------------------------------------------------------------------------
# Wing-name normalization migration (#1675 follow-up)
# ---------------------------------------------------------------------------
#
# normalize_wing_name now strips leading/trailing separators, so a path-encoded
# dirname like ``-home-user-proj`` derives ``home_user_proj`` instead of
# ``_home_user_proj``. Palaces built before that rule filed drawers under the
# old, leading-underscore wing, which the new derivation no longer matches —
# searches and diary reads under the new name miss the old memories.
#
# This migration re-keys the ``wing`` metadata field on drawers and closets to
# the normalized form, merging collisions. Drawer/closet IDs embed the wing as
# an opaque prefix that is never decoded back into a wing (verified: nothing
# splits a wing out of an ID; mining idempotency keys on ``source_file``), so
# the IDs are left untouched — closet ``→drawer_id`` pointers stay valid and
# future mining still skips already-mined files. Tunnels resolve via existing
# read-time normalization and need no rewrite. The pass is idempotent.


def _normalized_wing_target(wing):
    """Return the normalized wing if it differs from ``wing``, else ``None``.

    ``None`` means "no migration needed" — either the value is not a non-empty
    string, normalization is a no-op, or it would normalize to empty.
    """
    from .config import normalize_wing_name

    if not isinstance(wing, str) or not wing:
        return None
    # Apply the full normalization and explicitly strip leading/trailing
    # separators. The strip is this migration's whole purpose (#1675); doing it
    # here rather than relying on normalize_wing_name keeps the migration correct
    # even when run against a build whose normalize_wing_name predates #1675, and
    # matches the post-#1675 derivation exactly.
    target = normalize_wing_name(wing).strip("_")
    if not target or target == wing:
        return None
    return target


def plan_wing_renames(items):
    """Pure planner over ``(id, metadata)`` pairs.

    Returns ``(summary, updates)`` where ``summary`` is ``{(old, new): count}``
    and ``updates`` is ``[(id, new_metadata), ...]`` for only the records whose
    wing changes. Metadata is copied; only the ``wing`` key is rewritten.
    """
    summary = defaultdict(int)
    updates = []
    for rec_id, meta in items:
        meta = dict(meta or {})
        target = _normalized_wing_target(meta.get("wing"))
        if target is None:
            continue
        summary[(meta["wing"], target)] += 1
        meta["wing"] = target
        updates.append((rec_id, meta))
    return summary, updates


def _iter_collection_items(col, batch_size=1000):
    """Yield ``(id, metadata)`` for every record in a backend collection."""
    total = col.count()
    offset = 0
    while offset < total:
        batch = col.get(limit=batch_size, offset=offset, include=["metadatas"])
        ids = batch.ids if hasattr(batch, "ids") else batch["ids"]
        metas = batch.metadatas if hasattr(batch, "metadatas") else batch["metadatas"]
        if not ids:
            break
        for rec_id, meta in zip(ids, metas):
            yield rec_id, meta
        offset += len(ids)


def _apply_wing_updates(col, updates, batch_size=500):
    """Re-label the ``wing`` metadata field in place for the planned updates."""
    for i in range(0, len(updates), batch_size):
        chunk = updates[i : i + batch_size]
        col.update(ids=[u[0] for u in chunk], metadatas=[u[1] for u in chunk])


def _plan_topics_by_wing_renames():
    """Return ``{old_wing: new_wing}`` for ``topics_by_wing`` keys to normalize."""
    try:
        from .miner import _load_known_entities_raw

        reg = _load_known_entities_raw()
    except Exception:
        return {}
    tbw = reg.get("topics_by_wing")
    if not isinstance(tbw, dict):
        return {}
    renames = {}
    for wing in list(tbw.keys()):
        target = _normalized_wing_target(wing)
        if target is not None:
            renames[wing] = target
    return renames


def _apply_topics_by_wing_renames(renames):
    """Re-key ``topics_by_wing`` in known_entities.json, merging on collision."""
    if not renames:
        return
    import json

    from .miner import _ENTITY_REGISTRY_PATH, _load_known_entities_raw

    try:
        reg = _load_known_entities_raw()
    except Exception:
        return
    tbw = reg.get("topics_by_wing")
    if not isinstance(tbw, dict):
        return
    for old, new in renames.items():
        if old not in tbw:
            continue
        old_topics = tbw.pop(old) or []
        if new in tbw:
            merged = list(tbw[new])
            for topic in old_topics:
                if topic not in merged:
                    merged.append(topic)
            tbw[new] = merged
        else:
            tbw[new] = old_topics
    reg["topics_by_wing"] = tbw
    os.makedirs(os.path.dirname(_ENTITY_REGISTRY_PATH), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(_ENTITY_REGISTRY_PATH), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(reg, f, ensure_ascii=False, indent=2)
        os.replace(tmp, _ENTITY_REGISTRY_PATH)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def migrate_wing_names(palace_path: str, dry_run: bool = False, confirm: bool = False) -> bool:
    """Normalize legacy wing names in ``palace_path`` (strip leading/trailing
    separators), so palaces built before #1675 keep their memories discoverable.

    Re-keys the ``wing`` metadata on drawers and closets in place (IDs untouched)
    and the ``topics_by_wing`` registry, merging collisions. Idempotent.

    Returns True if anything was (or, in dry-run, would be) migrated.
    """
    from .palace import MineAlreadyRunning, mine_palace_lock

    palace_path = os.path.abspath(os.path.expanduser(palace_path))
    try:
        with mine_palace_lock(palace_path):
            return _migrate_wing_names_locked(palace_path, dry_run=dry_run, confirm=confirm)
    except MineAlreadyRunning as exc:
        print(f"  Another writer already holds this palace: {exc}")
        print("  Wing-name migration stopped before changing anything; nothing changed.")
        raise SystemExit(2) from exc


def _migrate_wing_names_locked(
    palace_path: str, dry_run: bool = False, confirm: bool = False
) -> bool:
    from .palace import get_closets_collection, get_collection

    try:
        drawers = get_collection(palace_path, create=False)
    except Exception as exc:
        print(f"  No drawer collection found at {palace_path} ({exc}).")
        return False

    d_items = list(_iter_collection_items(drawers))
    all_wings = {(m or {}).get("wing") for _, m in d_items if (m or {}).get("wing")}
    d_summary, d_updates = plan_wing_renames(d_items)

    closets = None
    c_summary, c_updates = defaultdict(int), []
    try:
        closets = get_closets_collection(palace_path, create=False)
        c_summary, c_updates = plan_wing_renames(_iter_collection_items(closets))
    except Exception:
        closets = None

    topic_renames = _plan_topics_by_wing_renames()

    if not d_updates and not c_updates and not topic_renames:
        print("  All wing names are already normalized -- nothing to migrate.")
        return False

    print("\n  Wing-name migration plan:")
    merged = defaultdict(lambda: [0, 0])
    for key, count in d_summary.items():
        merged[key][0] = count
    for key, count in c_summary.items():
        merged[key][1] = count
    for (old, new), (d_count, c_count) in sorted(merged.items()):
        note = "  (MERGE into existing wing)" if new in all_wings else ""
        print(f"    {old!r} -> {new!r}: {d_count} drawer(s), {c_count} closet(s){note}")
    if topic_renames:
        print(f"    topics_by_wing: {len(topic_renames)} key(s) re-keyed")

    if dry_run:
        print("\n  DRY RUN -- no changes made.\n")
        return True

    if not confirm:
        try:
            resp = input("  Apply this wing-name migration? [y/N] ").strip().lower()
        except EOFError:
            resp = ""
        if resp not in ("y", "yes"):
            print("  Aborted.")
            return False

    _apply_wing_updates(drawers, d_updates)
    if closets is not None and c_updates:
        _apply_wing_updates(closets, c_updates)
    _apply_topics_by_wing_renames(topic_renames)

    parts = [f"{len(d_updates)} drawer(s)"]
    if c_updates:
        parts.append(f"{len(c_updates)} closet(s)")
    if topic_renames:
        parts.append(f"{len(topic_renames)} topic key(s)")
    print(f"\n  Migrated {', '.join(parts)}.\n")
    return True
