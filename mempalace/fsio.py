"""Durable replacement of palace state files.

``tempfile.mkstemp`` is not used. On Windows it treats ``PermissionError`` as
a name collision and retries ``TMP_MAX`` times (hundreds of thousands, and on
older Pythons ``os.TMP_MAX``), which reads as a hang. Callers that need the
permission error — a directory that will take no new name — never see it.
The opener below retries only ``FileExistsError``, and only
``TEMP_NAME_ATTEMPTS`` times. (#2530, #2531)

A publish writes a temporary file in the destination directory, fsyncs it,
then ``os.replace``s it into place and fsyncs the directory. An exception
before the replace leaves the previous file untouched and removes the
temporary file.
"""

import errno
import json
import os
import stat
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Union

PathLike = Union[str, os.PathLike]

# Only a name already taken is retried. Every other OSError reaches the caller.
TEMP_NAME_ATTEMPTS = 8

_IN_PLACE_ERRNOS = (errno.EACCES, errno.EPERM, errno.EROFS)


def unlink_quietly(path: PathLike) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def fsync_directory(directory: PathLike) -> None:
    """Make a rename in ``directory`` durable.

    On ext4 the kernel can acknowledge a rename and, after a crash, come
    back with the temporary file still present and the target holding the
    old bytes. Windows cannot open a directory this way; answering nothing
    there matches a filesystem that does not implement directory fsync.
    """
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_target(path: PathLike) -> Path:
    """The file a write should replace, following a symlink to it.

    Replacing the link itself would leave the real file unchanged. A link
    this call cannot ``lstat`` is raised rather than treated as a non-link:
    that reading would put a regular file where the link was.
    """
    path = Path(path)
    try:
        is_link = stat.S_ISLNK(os.lstat(str(path)).st_mode)
    except FileNotFoundError:
        return path
    if is_link:
        return Path(os.path.realpath(str(path)))
    return path


def open_new_temp_file(directory: PathLike, prefix: str, suffix: str = ""):
    """Create one new file under ``directory`` and return ``(fd, path)``.

    Only ``FileExistsError`` is retried, and only ``TEMP_NAME_ATTEMPTS``
    times. ``EPERM``, ``EACCES`` and ``EROFS`` propagate so the caller can
    fall back. Exhausting the attempts raises ``FileExistsError``.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    names = tempfile._get_candidate_names()
    directory = str(directory)
    for _ in range(TEMP_NAME_ATTEMPTS):
        candidate = os.path.join(directory, f"{prefix}{next(names)}{suffix}")
        try:
            return os.open(candidate, flags, 0o600), candidate
        except FileExistsError:
            continue
    raise FileExistsError(
        errno.EEXIST,
        f"no usable temporary file name found in {directory} after {TEMP_NAME_ATTEMPTS} attempts",
    )


def _chmod_quiet(path: PathLike, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except (OSError, NotImplementedError):
        pass


def write_json_in_place(path: PathLike, payload: Any) -> None:
    """Write JSON without a rename, for a directory that refuses a new name.

    This truncates ``path`` before the new bytes are durable. Callers use it
    only when the atomic publish cannot create a temporary file at all.
    """
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    _chmod_quiet(path, 0o600)


def write_text_in_place(path: PathLike, text: str) -> None:
    """Write text without a rename. See ``write_json_in_place``."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    _chmod_quiet(path, 0o600)


def _fill_temp(fd: int, tmp: str, write_body: Callable[[Any], None]) -> None:
    """Serialize into ``fd`` and fsync before anything is published.

    The mode is set before the write. Under a umask that clears the owner's
    write bit, ``O_CREAT`` leaves the file at 0400, and one left behind by a
    killed run is then a name its owner cannot open.
    """
    try:
        _chmod_quiet(tmp, 0o600)
        os.ftruncate(fd, 0)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            write_body(handle)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        unlink_quietly(tmp)
        raise


def _open_publish_temp(path: Path):
    """Return ``(fd, tmp)`` for a publish into ``path``.

    The pid-named file is tried first. ``O_NOFOLLOW`` keeps a symlink at that
    name from being written through. A hard link is not a symlink, so a
    second link is rejected before the truncate: truncating it would empty a
    file nobody named here. Anything else in the way of the pid name asks
    the directory for a name of its own.
    """
    tmp = str(path.with_name(f"{path.name}.tmp-{os.getpid()}"))
    flags = os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(tmp, flags, 0o600)
        if os.fstat(fd).st_nlink > 1:
            os.close(fd)
            raise OSError(errno.EMLINK, "temporary name has another link", tmp)
    except OSError:
        fd, tmp = open_new_temp_file(
            path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
    return fd, tmp


def _warn_in_place(path: Path, *, why: str) -> None:
    print(
        f"  ! {path.parent} would not take {why}, so {path.name} was "
        "written in place and an interrupted write can truncate it",
        file=sys.stderr,
    )


def _finish_failed_replace(
    exc: BaseException, tmp: str, path: Path, in_place: Callable[[Path], None]
) -> None:
    """Publish in place when the directory itself refused the rename.

    A non-``OSError`` (a signal between the write and the rename) has not
    published anything, so the temporary file is removed. The same is true
    of an errno the fallback is not for. The temporary file holds a finished,
    fsynced copy: it is removed only after the in-place write returns.
    """
    if not isinstance(exc, OSError) or exc.errno not in _IN_PLACE_ERRNOS:
        unlink_quietly(tmp)
        raise exc
    try:
        in_place(path)
    except BaseException:
        print(
            f"  ! {path.name} was not written and may have been truncated; "
            f"{tmp} holds what this call was asked to save",
            file=sys.stderr,
        )
        raise
    unlink_quietly(tmp)
    if os.path.exists(tmp):
        print(
            f"  ! {tmp} holds a copy of what was written and could not be "
            "removed; nothing here removes it later either",
            file=sys.stderr,
        )
    _warn_in_place(path, why="the rename")


def atomic_publish(
    path: PathLike,
    write_body: Callable[[Any], None],
    in_place: Callable[[Path], None],
    *,
    resolve_target: Optional[Callable[[PathLike], PathLike]] = None,
    sync_directory: Optional[Callable[[Path], None]] = None,
) -> None:
    """Replace ``path`` with what ``write_body`` writes, or leave it untouched.

    ``in_place`` runs only when the directory refuses a new name
    (``EACCES``, ``EPERM``, ``EROFS``) while the destination itself may
    still be writable. Every other failure propagates and does not truncate
    ``path``. ``sync_directory`` defaults to ``fsync_directory``; callers
    pass their own when tests or a wrapper own that step.
    """
    if resolve_target is not None:
        path = resolve_target(path)
    else:
        path = write_target(path)
    path = Path(path)

    try:
        fd, tmp = _open_publish_temp(path)
    except OSError as exc:
        if exc.errno not in _IN_PLACE_ERRNOS:
            raise
        in_place(path)
        _warn_in_place(path, why="a temporary file")
        return

    _fill_temp(fd, tmp, write_body)
    try:
        os.replace(tmp, str(path))
    except BaseException as exc:
        _finish_failed_replace(exc, tmp, path, in_place)
        return
    if sync_directory is not None:
        sync_directory(path.parent)
    else:
        fsync_directory(path.parent)


def atomic_write_json(
    path: PathLike,
    payload: Any,
    *,
    resolve_target: Optional[Callable[[PathLike], PathLike]] = None,
    write_in_place: Optional[Callable[[PathLike, Any], None]] = None,
    sync_directory: Optional[Callable[[Path], None]] = None,
) -> None:
    """Serialize ``payload`` into ``path`` through a temporary file.

    JSON is indented, UTF-8, with non-ASCII kept as characters. The rename
    is what publishes it, so a crash while serializing leaves the previous
    file exactly as it was.
    """
    writer = write_in_place or write_json_in_place

    def _body(handle: Any) -> None:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    def _in_place(dest: Path) -> None:
        writer(dest, payload)

    atomic_publish(
        path,
        _body,
        _in_place,
        resolve_target=resolve_target,
        sync_directory=sync_directory,
    )


def atomic_write_text(
    path: PathLike,
    text: str,
    *,
    resolve_target: Optional[Callable[[PathLike], PathLike]] = None,
    write_in_place: Optional[Callable[[PathLike, str], None]] = None,
    sync_directory: Optional[Callable[[Path], None]] = None,
) -> None:
    """Replace ``path`` with ``text`` through a temporary file.

    Same crash rules as ``atomic_write_json``.
    """
    writer = write_in_place or write_text_in_place

    def _body(handle: Any) -> None:
        handle.write(text)

    def _in_place(dest: Path) -> None:
        writer(dest, text)

    atomic_publish(
        path,
        _body,
        _in_place,
        resolve_target=resolve_target,
        sync_directory=sync_directory,
    )


def move_aside(
    path: PathLike,
    *,
    label: str,
    resolve_target: Optional[Callable[[PathLike], PathLike]] = None,
) -> Optional[str]:
    """Rename ``path`` to ``<name>.<label>-<stamp>-<random>``, keeping its bytes.

    Returns the new path, or ``None`` when nothing was there. A failure
    other than the file already being gone leaves it where it was and
    raises: a file that could not be moved is not one to write over.
    """
    if resolve_target is not None:
        path = resolve_target(path)
    else:
        path = write_target(path)
    path = Path(path)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    fd, target = open_new_temp_file(path.parent, prefix=f"{path.name}.{label}-{stamp}-")
    os.close(fd)
    try:
        os.replace(str(path), target)
    except FileNotFoundError:
        unlink_quietly(target)
        return None
    except OSError:
        unlink_quietly(target)
        raise
    return target
