#!/usr/bin/env python3
"""
split_mega_files.py — Split concatenated transcript files into per-session files
=================================================================================

Scans a directory for .txt files that contain multiple Claude Code sessions
(identified by "Claude Code v" headers). Splits each into individual files
named with: date, time, people detected, and subject from first prompt.

Distinguishes true session starts from mid-session context restores
(which show "Ctrl+E to show X previous messages").

Output files are written to --output-dir (default: same dir as source).
Original files are renamed with .mega_backup extension (not deleted).

Usage:
    python3 split_mega_files.py                          # scan ~/Desktop/transcripts
    python3 split_mega_files.py --source ~/Desktop/transcripts  # explicit source
    python3 split_mega_files.py --dry-run                # show what would happen
    python3 split_mega_files.py --min-sessions 2         # only files with 2+ sessions

By: Ben, 2026-03-30
"""

import argparse
import json
import os
import re
import stat
from pathlib import Path

HOME = Path.home()
LUMI_DIR = Path(os.environ.get("MEMPALACE_SOURCE_DIR", str(HOME / "Desktop/transcripts")))

# People we know about (for name detection in content)
# Loaded from ~/.mempalace/known_names.json if it exists; no built-in fallback names.
_KNOWN_NAMES_PATH = HOME / ".mempalace" / "known_names.json"
_KNOWN_NAMES_CACHE = None


def _load_known_names_config(force_reload: bool = False):
    """Load and cache the optional known-names config file."""
    global _KNOWN_NAMES_CACHE

    if force_reload:
        _KNOWN_NAMES_CACHE = None

    if _KNOWN_NAMES_CACHE is not None:
        return _KNOWN_NAMES_CACHE

    if _KNOWN_NAMES_PATH.exists():
        try:
            _KNOWN_NAMES_CACHE = json.loads(_KNOWN_NAMES_PATH.read_text(encoding="utf-8"))
            return _KNOWN_NAMES_CACHE
        except (json.JSONDecodeError, OSError):
            pass

    _KNOWN_NAMES_CACHE = None
    return None


def _load_known_people() -> list:
    """Load known names from config file, or return empty list if not configured."""
    data = _load_known_names_config()
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("names", [])
    return []


# Loaded on first use so importing this module does not read the home directory.
KNOWN_PEOPLE = None


def _active_known_people() -> list:
    global KNOWN_PEOPLE
    if KNOWN_PEOPLE is None:
        KNOWN_PEOPLE = _load_known_people()
    return KNOWN_PEOPLE


def _load_username_map() -> dict:
    """Load username-to-name mapping from config file."""
    data = _load_known_names_config()
    if isinstance(data, dict):
        return data.get("username_map", {})
    return {}


def is_true_session_start(lines, idx):
    """
    True session start: 'Claude Code v' header NOT followed by 'Ctrl+E'/'previous messages'
    within the next 6 lines (those are context restores, not new sessions).
    """
    nearby = "".join(lines[idx : idx + 6])
    return "Ctrl+E" not in nearby and "previous messages" not in nearby


def find_session_boundaries(lines):
    """Return list of line indices where true new sessions begin."""
    boundaries = []
    for i, line in enumerate(lines):
        if "Claude Code v" in line and is_true_session_start(lines, i):
            boundaries.append(i)
    return boundaries


def extract_timestamp(lines):
    """
    Find the first timestamp line: ⏺ H:MM AM/PM Weekday, Month DD, YYYY
    Returns (datetime_str, iso_str) or (None, None).
    """
    ts_pattern = re.compile(r"⏺\s+(\d{1,2}:\d{2}\s+[AP]M)\s+\w+,\s+(\w+)\s+(\d{1,2}),\s+(\d{4})")
    months = {
        "January": "01",
        "February": "02",
        "March": "03",
        "April": "04",
        "May": "05",
        "June": "06",
        "July": "07",
        "August": "08",
        "September": "09",
        "October": "10",
        "November": "11",
        "December": "12",
    }
    for line in lines[:50]:
        m = ts_pattern.search(line)
        if m:
            time_str, month, day, year = m.groups()
            mon = months.get(month, "00")
            day_z = day.zfill(2)
            time_safe = time_str.replace(":", "").replace(" ", "")
            iso = f"{year}-{mon}-{day_z}"
            human = f"{year}-{mon}-{day_z}_{time_safe}"
            return human, iso
    return None, None


def extract_people(lines):
    """
    Detect people mentioned as speakers or by name in first 100 lines.
    Returns sorted list of detected names.
    """
    found = set()
    text = "".join(lines[:100])

    # Speaker tags: "Alice:", "Ben:", etc.
    for person in _active_known_people():
        # Lookarounds, not \\b: a name like ``C++`` ends on a non-word
        # character, so a trailing word-boundary never matches it.
        if re.search(rf"(?<!\w){re.escape(person)}(?!\w)", text, re.IGNORECASE):
            found.add(person)

    # Working directory username hint — map to known people if configured
    dir_match = re.search(r"/Users/(\w+)/", text)
    if dir_match:
        username = dir_match.group(1)
        # User can map usernames to names in ~/.mempalace/known_names.json
        # under a "username_map" key, e.g. {"username_map": {"jdoe": "John"}}
        username_map = _load_username_map()
        if username in username_map:
            found.add(username_map[username])

    return sorted(found)


def extract_subject(lines):
    """
    Find the first meaningful user prompt (> line that isn't a shell command).
    Returns cleaned, filename-safe subject string.
    """
    skip_patterns = re.compile(
        r"^(\.\/|cd |ls |python|bash|git |cat |source |export |claude|./activate)"
    )
    for line in lines:
        if line.startswith("> "):
            prompt = line[2:].strip()
            if prompt and not skip_patterns.match(prompt) and len(prompt) > 5:
                # Clean for filename
                subject = re.sub(r"[^\w\s-]", "", prompt)
                subject = re.sub(r"\s+", "-", subject.strip())
                return subject[:60]
    return "session"


def _decode_source_bytes(raw: bytes) -> str:
    """Decode transcript bytes without the locale codec or ``errors='replace'``.

    UTF-8 first, so valid UTF-8 stays verbatim. Otherwise cp1252, then
    latin-1, which maps every byte and does not invent U+FFFD.
    """
    if not raw:
        return ""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    try:
        return raw.decode("cp1252")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _read_lines(path: Path) -> list[str]:
    return _decode_source_bytes(path.read_bytes()).splitlines(keepends=True)


def _session_segments(lines: list[str], boundaries: list[int]) -> list[tuple[list[str], bool, int]]:
    """Preamble plus every session, including sessions under 10 lines."""
    segments = []
    if boundaries[0] > 0:
        preamble = lines[: boundaries[0]]
        if any(line.strip() for line in preamble):
            segments.append((preamble, True, 0))
    ends = boundaries + [len(lines)]
    for index, (start, end) in enumerate(zip(ends, ends[1:])):
        chunk = lines[start:end]
        if chunk:
            segments.append((chunk, False, index))
    return segments


def _suffixed_names(base_name: str):
    suffix = Path(base_name).suffix
    stem = base_name[: -len(suffix)] if suffix else base_name
    yield base_name
    number = 2
    while True:
        yield f"{stem}-{number}{suffix}"
        number += 1


def _output_name(path: Path, chunk: list[str], is_preamble: bool, index: int) -> str:
    ts_human, _ts_iso = extract_timestamp(chunk)
    people = extract_people(chunk)
    subject = extract_subject(chunk)
    if is_preamble:
        ts_part = "part00"
    else:
        ts_part = ts_human or f"part{index + 1:02d}"
    people_part = "-".join(people[:3]) if people else "unknown"
    src_stem = re.sub(r"[^\w-]", "_", path.stem)[:40]
    name = f"{src_stem}__{ts_part}_{people_part}_{subject}.txt"
    name = re.sub(r"[^\w\.\-]", "_", name)
    return re.sub(r"_+", "_", name)


def _claim_output_path(out_dir: Path, base_name: str, reserved: set, dry_run: bool):
    """Return a path that does not collide, opening it with ``O_EXCL`` when writing."""
    for name in _suffixed_names(base_name):
        candidate = out_dir / name
        if candidate in reserved or os.path.lexists(candidate):
            continue
        if dry_run:
            reserved.add(candidate)
            return candidate, None
        try:
            fd = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            continue
        reserved.add(candidate)
        return candidate, fd
    raise RuntimeError(f"could not allocate an output name for {base_name}")


def _write_exclusive(out_path: Path, fd: int, text: str) -> None:
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
    except Exception:
        out_path.unlink(missing_ok=True)
        raise


def split_file(filepath, output_dir, dry_run=False, lines=None):
    """
    Split a single mega-file into per-session files.
    Returns list of output paths written (or would be written if dry_run).

    Text before the first session and sessions under 10 lines are written
    too. Two sessions that generate the same filename get ``-2``, ``-3``, ...
    instead of overwriting each other.
    """
    path = Path(filepath)
    max_size = 500 * 1024 * 1024  # 500 MB safety limit
    if lines is None:
        if path.stat().st_size > max_size:
            print(f"  SKIP: {path.name} exceeds {max_size // (1024 * 1024)} MB limit")
            return []
        lines = _read_lines(path)

    boundaries = find_session_boundaries(lines)
    if len(boundaries) < 2:
        return []  # Not a mega-file

    out_dir = Path(output_dir) if output_dir else path.parent
    segments = _session_segments(lines, boundaries)
    written = []
    reserved: set = set()

    for position, (chunk, is_preamble, index) in enumerate(segments, 1):
        name = _output_name(path, chunk, is_preamble, index)
        out_path, fd = _claim_output_path(out_dir, name, reserved, dry_run)
        if dry_run:
            print(f"  [{position}/{len(segments)}] {out_path.name}  ({len(chunk)} lines)")
        else:
            _write_exclusive(out_path, fd, "".join(chunk))
            print(f"  + {out_path.name}  ({len(chunk)} lines)")
        written.append(out_path)

    return written


def _retire_original(path: Path) -> bool:
    """Rename ``path`` to ``.mega_backup`` without replacing an existing backup.

    The rename is how the original bytes stay recoverable. If that name is
    already taken, the source file is left where it is.
    """
    backup = path.with_suffix(".mega_backup")
    if os.path.lexists(backup):
        print(f"  ! leaving {path.name} in place; {backup.name} already exists")
        return False
    path.rename(backup)
    print(f"  -> Original renamed to {backup.name}\n")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Split concatenated transcript mega-files into per-session files"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Source directory (default: MEMPALACE_SOURCE_DIR or ~/Desktop/transcripts)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory (default: same as source)"
    )
    parser.add_argument(
        "--min-sessions",
        type=int,
        default=2,
        help="Only split files with at least N sessions (default: 2)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would happen without writing files"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Split a single specific file instead of scanning dir",
    )
    args = parser.parse_args()

    src_dir = Path(args.source).expanduser().resolve() if args.source else LUMI_DIR
    output_dir = args.output_dir or None  # None = same dir as file

    if args.file:
        files = [Path(args.file)]
    else:
        files = sorted(src_dir.glob("*.txt"))

    mega_files = []
    max_scan_size = 500 * 1024 * 1024  # 500 MB
    for f in files:
        file_stat = f.stat()
        # ``glob`` lists a FIFO named ``x.txt`` like any other match, and
        # read_text() on one blocks in the kernel until a writer appears.
        # stat() never blocks, so the type decides before the open does.
        if not stat.S_ISREG(file_stat.st_mode):
            print(f"  SKIP: {f.name} (not a regular file)")
            continue
        if file_stat.st_size > max_scan_size:
            print(f"  SKIP: {f.name} exceeds {max_scan_size // (1024 * 1024)} MB limit")
            continue
        lines = _read_lines(f)
        boundaries = find_session_boundaries(lines)
        if len(boundaries) >= args.min_sessions:
            mega_files.append((f, len(boundaries), lines))

    if not mega_files:
        print(f"No mega-files found in {src_dir} (min {args.min_sessions} sessions).")
        return

    print(f"\n{'=' * 60}")
    print(f"  Mega-file splitter -- {'DRY RUN' if args.dry_run else 'SPLITTING'}")
    print(f"{'=' * 60}")
    print(f"  Source:      {src_dir}")
    print(f"  Output:      {output_dir or 'same dir as source'}")
    print(f"  Mega-files:  {len(mega_files)}")
    print(f"{'-' * 60}\n")

    total_written = 0
    for f, n_sessions, lines in mega_files:
        print(f"  {f.name}  ({n_sessions} sessions, {f.stat().st_size // 1024}KB)")
        written = split_file(f, output_dir, dry_run=args.dry_run, lines=lines)
        total_written += len(written)

        if not args.dry_run and written:
            _retire_original(f)
        else:
            print()

    print(f"{'-' * 60}")
    if args.dry_run:
        print(f"  DRY RUN -- would create {total_written} files from {len(mega_files)} mega-files")
    else:
        print(f"  Done -- created {total_written} files from {len(mega_files)} mega-files")
    print()


if __name__ == "__main__":
    main()
