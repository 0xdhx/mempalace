"""Timestamps for persisted values, and the one parser that reads them back.

Every timestamp MemPalace stores is UTC ISO-8601 with an explicit offset
(``utc_now_iso()``, e.g. ``2026-10-17T13:00:00.123456+00:00``). Palaces
written before that also hold naive values, which were local wall-clock
time when written; ``parse_timestamp`` reads those as local time so both
forms compare as the same instants. Stored values are never rewritten.

Text order of ``filed_at`` stays chronological within each form. Across
the switch from naive local to UTC it can be off by at most the local UTC
offset, and only for drawers filed within that many hours of the upgrade.
"""

from datetime import date, datetime, timezone
from typing import Optional

# Tests pin the zone naive values are read in; None means the system zone.
_LOCAL_TZ = None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """The persisted form: UTC, microseconds, explicit ``+00:00`` offset."""
    return utc_now().isoformat(timespec="microseconds")


def utc_now_z() -> str:
    """Second-precision ``...Z`` form used by the knowledge graph and logstream."""
    return utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


def filename_stamp(sep: str = "-", utc: bool = False) -> str:
    """``YYYYMMDD<sep>HHMMSS`` for backup and set-aside file names."""
    now = utc_now() if utc else datetime.now()
    return now.strftime(f"%Y%m%d{sep}%H%M%S")


def localize(dt: datetime) -> datetime:
    """Attach the local zone to a naive datetime; aware values pass through."""
    if dt.tzinfo is not None:
        return dt
    if _LOCAL_TZ is not None:
        return dt.replace(tzinfo=_LOCAL_TZ)
    try:
        return dt.astimezone()
    except (OverflowError, OSError, ValueError):
        # Windows cannot resolve the zone for dates before 1970.
        return dt.replace(tzinfo=datetime.now().astimezone().tzinfo)


def parse_iso(value: str) -> datetime:
    """Parse an ISO date or timestamp into an aware datetime, or raise ValueError.

    Accepts a date, a timestamp with optional fraction, and an optional
    ``Z``/``±HH:MM`` offset (a trailing ``Z`` is stripped because
    ``fromisoformat`` rejects it before Python 3.11). Naive input is local time.
    """
    iso = value.strip()
    zulu = iso.endswith(("Z", "z"))
    if zulu:
        iso = iso[:-1]
    parsed = datetime.fromisoformat(iso)
    if zulu and parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return localize(parsed)


def parse_timestamp(value) -> Optional[datetime]:
    """Aware datetime for a stored timestamp, or None if absent or unparseable."""
    if isinstance(value, datetime):
        return localize(value)
    if isinstance(value, date):
        return localize(datetime(value.year, value.month, value.day))
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return parse_iso(value)
    except ValueError:
        return None
