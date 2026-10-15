"""Shared helpers for the remote storage backends.

Metadata values are coerced one key at a time, the way Milvus already did:
a single non-JSON value must not wipe ``wing``, ``room``, or ``source_file``.
Backend marker files are published with :func:`mempalace.fsio.atomic_write_json`
and rewritten only when the remote target changed, so a crash mid-write
cannot leave the truncated marker ``_read_marker`` reports as a mismatch.
"""

import json
import logging
import os
from typing import Any, Optional

from ..fsio import atomic_write_json

logger = logging.getLogger(__name__)

_COERCED_KEYS: set = set()


def jsonable_metadata(meta: Optional[dict], *, reserved: frozenset = frozenset()) -> dict:
    """Return a JSON-serializable copy of ``meta``.

    Values ``json.dumps`` accepts are kept. Anything else is stored as
    ``str(value)`` and the key is named once in a warning. An empty input
    is ``{}``; a non-empty input is never replaced with ``{}`` because one
    value failed.
    """
    cleaned = {}
    for key, value in (meta or {}).items():
        name = str(key)
        if name in reserved:
            raise ValueError(f"metadata key {name!r} clashes with a reserved field")
        cleaned[name] = _jsonable_value(name, value)
    return cleaned


def _jsonable_value(key: str, value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        if key not in _COERCED_KEYS:
            _COERCED_KEYS.add(key)
            logger.warning(
                "metadata key %r is not JSON-serializable (%s); storing str(%s)",
                key,
                type(value).__name__,
                key,
            )
        return str(value)
    return value


def _load_marker(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _same_marker(existing: dict, marker: dict, target_key: str) -> bool:
    if existing.get("backend") != marker.get("backend"):
        return False
    if existing.get("palace_id") != marker.get("palace_id"):
        return False
    if existing.get("schema_version") != marker.get("schema_version"):
        return False
    return existing.get(target_key) == marker.get(target_key)


def publish_marker(path: str, marker: dict, *, target_key: str) -> None:
    """Replace ``path`` with ``marker`` only when the remote target changed.

    An existing ``created_at`` is kept. A crash while the new bytes are
    still in the temporary file leaves the previous marker readable.
    """
    existing = _load_marker(path) if os.path.isfile(path) else None
    if isinstance(existing, dict) and _same_marker(existing, marker, target_key):
        return
    published = dict(marker)
    if isinstance(existing, dict) and existing.get("created_at"):
        published["created_at"] = existing["created_at"]
    atomic_write_json(path, published)
