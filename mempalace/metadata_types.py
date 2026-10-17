"""Declared types for the drawer metadata fields MemPalace compares.

Every backend write runs through :func:`coerce_metadata_types`, so a value
that arrives in the wrong shape (``"3"`` for a chunk index, ``"1.7e9"`` for
an mtime, ``"true"`` for a flag, a ``datetime`` for a timestamp) is stored as
its declared type and range filters and the mined-set checks compare numbers
with numbers. A value that does not parse cleanly is stored as given: these
fields are bookkeeping, and nothing is dropped to make them fit. Readers that
compare these fields call the same function on legacy rows.
"""

import math
import re
from datetime import datetime

INT_FIELDS = frozenset(
    {
        "chunk_index",
        "chunk_total",
        "line_start",
        "line_end",
        "entry_index",
        "normalize_version",
        "convo_chunker_version",
        "general_extractor_version",
    }
)
FLOAT_FIELDS = frozenset({"source_mtime", "compression_ratio"})
BOOL_FIELDS = frozenset({"is_sentinel"})
TIMESTAMP_FIELDS = frozenset({"filed_at", "created_at"})

_INT_TEXT = re.compile(r"[+-]?[0-9]+")
_FLOAT_TEXT = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_BOOL_TEXT = {"true": True, "1": True, "false": False, "0": False}


def _as_int(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    if isinstance(value, str) and _INT_TEXT.fullmatch(value.strip()):
        return int(value.strip())
    return value


def _as_float(value):
    if isinstance(value, str) and _FLOAT_TEXT.fullmatch(value.strip()):
        number = float(value.strip())
        return number if math.isfinite(number) else value
    return value


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        return _BOOL_TEXT.get(value.strip().lower(), value)
    return value


def _as_timestamp(value):
    return value.isoformat() if isinstance(value, datetime) else value


def _coerce(key, value):
    if value is None:
        return value
    if key in INT_FIELDS:
        return _as_int(value)
    if key in FLOAT_FIELDS:
        return _as_float(value)
    if key in BOOL_FIELDS:
        return _as_bool(value)
    if key in TIMESTAMP_FIELDS:
        return _as_timestamp(value)
    return value


def coerce_metadata_types(metadata):
    """Return ``metadata`` with declared fields in their declared types.

    The same dict comes back when nothing changed, so a write batch is not
    copied for the common case.
    """
    if not metadata:
        return metadata
    coerced = None
    for key, value in metadata.items():
        new = _coerce(key, value)
        if new is not value:
            if coerced is None:
                coerced = dict(metadata)
            coerced[key] = new
    return coerced if coerced is not None else metadata
