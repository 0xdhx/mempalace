"""Append-only prior text for drawer updates.

The write-ahead log redacts content. This log lives in the palace directory,
the same privacy domain as the drawers, and keeps the verbatim document
``update_drawer`` is about to replace. Revisions are retained: nothing caps
the file, and deleting a drawer does not remove its earlier revisions.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

_REVISION_FILENAME = "drawer_revisions.jsonl"


def revisions_path(palace_path: str) -> Path:
    return Path(palace_path) / _REVISION_FILENAME


def append_drawer_revision(
    palace_path: str,
    drawer_id: str,
    content: str,
    metadata: dict | None = None,
) -> str:
    """Append one prior drawer and return its revision id.

    Raises ``OSError`` when the palace directory cannot take the append.
    Callers that are about to overwrite a drawer must not continue after that.
    """
    if not palace_path:
        raise OSError("no palace path for drawer revision")
    path = revisions_path(palace_path)
    # A remote backend keeps no drawers on disk, so the directory may not exist yet.
    path.parent.mkdir(parents=True, exist_ok=True)
    revision_id = uuid4().hex
    record = {
        "revision_id": revision_id,
        "drawer_id": drawer_id,
        "content": content,
        "metadata": metadata or {},
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    if _ends_mid_line(path):
        line = "\n" + line  # a crash tore the last append; start a fresh line
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0)
    fd = os.open(str(path), flags, 0o600)
    try:
        os.write(fd, line.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    return revision_id


def _ends_mid_line(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            handle.seek(-1, os.SEEK_END)
            return handle.read(1) != b"\n"
    except OSError:
        return False


def load_drawer_revisions(palace_path: str, drawer_id: str | None = None) -> list[dict]:
    """Return stored revisions, oldest first. Optionally one drawer's only.

    A line torn by a crash mid-append is skipped, not fatal.
    """
    path = revisions_path(palace_path)
    if not path.is_file():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and (drawer_id is None or row.get("drawer_id") == drawer_id):
                rows.append(row)
    return rows
