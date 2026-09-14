"""Symlink-safe file writes for sidecar and temp outputs.

The pipeline writes sidecars (SRT, JSON manifests, temp WAV, muxed video) next
to user-supplied videos. That directory is untrusted: a planted symlink with a
predictable sidecar name would otherwise be followed by ``open(..., "w")`` or
FFmpeg ``-y`` and overwrite whatever the link points at.

Every write is reserved as a ``ScratchReservation``: a private ``0700``
directory beside the destination, an exclusively created file inside it, and
the recorded identities (device, inode) of both. Text is written through the
creation descriptor and never reopened by pathname. Promotion and discard are
bound to the reservation: on POSIX the held directory descriptor is used
(``renameat``/``unlinkat``), so a renamed scratch directory cannot redirect the
move; elsewhere the identities are re-verified before any pathname operation.
Cleanup never recurses (``rmdir`` only), so a planted junction or directory
that happens to carry a scratch name is never walked.
"""

import contextlib
import hashlib
import os
import secrets
import stat
from dataclasses import dataclass

SCRATCH_PREFIX = ".asg-tmp-"
_TAG_STEM_LENGTH = 24
_NAME_ATTEMPTS = 64
_CREATE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
_DIR_OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
DIR_FD_SUPPORTED = {os.open, os.stat, os.unlink, os.rename} <= os.supports_dir_fd


class SymlinkRefusedError(OSError):
    """Raised when a pipeline output path is occupied by a symbolic link or was tampered with."""


@dataclass
class ScratchReservation:
    """A private scratch file plus the identities needed to promote or discard it safely."""

    path: str
    dir_path: str
    file_name: str
    dir_id: tuple
    file_id: tuple
    dir_fd: int | None = None
    fd: int | None = None

    def __fspath__(self):
        return self.path


def reject_symlink(path):
    """Raise SymlinkRefusedError if ``path`` is a symbolic link."""
    if os.path.islink(path):
        raise SymlinkRefusedError(f"Refusing to write through symlink: {path}")


def scratch_tag(name):
    """Return a bounded tag that identifies the owner of a scratch entry without collisions."""
    digest = hashlib.sha256(name.encode("utf-8", "surrogateescape")).hexdigest()[:8]
    return f"{name[:_TAG_STEM_LENGTH]}-{digest}"


def reserve_temp_path(final_path):
    """Reserve a private scratch file beside ``final_path`` for a pathname-based writer (FFmpeg).

    The extension of ``final_path`` is preserved so tools that pick a format
    from the suffix keep working. Pass ``reservation.path`` to the tool, then
    finish with ``promote_temp_path`` or ``discard_temp_path``.
    """
    reservation = _reserve(final_path)
    os.close(reservation.fd)
    reservation.fd = None
    return reservation


def promote_temp_path(reservation, final_path):
    """Atomically move the reserved file onto ``final_path``, refusing symlink or tampered targets."""
    reject_symlink(final_path)
    _verify_reservation(reservation)
    if reservation.dir_fd is not None:
        os.rename(reservation.file_name, final_path, src_dir_fd=reservation.dir_fd)
    else:
        os.replace(reservation.path, final_path)
    _release(reservation)


def discard_temp_path(reservation):
    """Best-effort removal of a reserved scratch file that will not be promoted.

    With a held directory descriptor the unlink is bound to the reserved
    directory even if its pathname was renamed or replaced; otherwise the
    identities are re-verified first so a replacement entry is never touched.
    """
    with contextlib.suppress(OSError):
        if reservation.dir_fd is not None:
            _verify_file(reservation, os.stat(reservation.file_name, dir_fd=reservation.dir_fd, follow_symlinks=False))
            os.unlink(reservation.file_name, dir_fd=reservation.dir_fd)
        else:
            _verify_reservation(reservation)
            os.remove(reservation.path)
    _release(reservation)


@contextlib.contextmanager
def atomic_text_writer(path, encoding="utf-8"):
    """Yield a text handle whose content lands on ``path`` only on success.

    Content goes through the creation descriptor of a reserved scratch file
    and is promoted with a bound rename after the identities are re-verified.
    On any exception the scratch file is discarded and the error re-raised.
    """
    reject_symlink(path)
    reservation = _reserve(path)
    try:
        with os.fdopen(reservation.fd, "w", encoding=encoding) as file_handle:
            reservation.fd = None
            yield file_handle
        promote_temp_path(reservation, path)
    except BaseException:
        discard_temp_path(reservation)
        raise


def _reserve(final_path):
    """Create the private directory and exclusive file for ``final_path``; the fd stays open."""
    directory = os.path.dirname(final_path) or "."
    extension = os.path.splitext(final_path)[1] or ".tmp"
    dir_path = _create_private_dir(directory, _scratch_prefix(final_path))
    dir_fd = os.open(dir_path, _DIR_OPEN_FLAGS) if DIR_FD_SUPPORTED else None
    try:
        fd, file_name = _create_exclusive(dir_path, dir_fd, "output-", extension)
        dir_id = _identity(_stat_dir(dir_path, dir_fd))
        file_id = _identity(os.fstat(fd))
    except BaseException:
        _close_quietly(dir_fd)
        with contextlib.suppress(OSError):
            os.rmdir(dir_path)
        raise
    return ScratchReservation(os.path.join(dir_path, file_name), dir_path, file_name, dir_id, file_id, dir_fd, fd)


def _stat_dir(dir_path, dir_fd):
    """Stat the scratch directory through its held descriptor when one exists."""
    return os.fstat(dir_fd) if dir_fd is not None else os.lstat(dir_path)


def _scratch_prefix(final_path):
    """Build the opaque directory prefix for ``final_path``."""
    stem = os.path.splitext(os.path.basename(final_path))[0]
    return f"{SCRATCH_PREFIX}{scratch_tag(stem)}-"


def _create_private_dir(directory, prefix):
    """Create a ``0700`` directory with an unpredictable name under ``directory``."""
    for _ in range(_NAME_ATTEMPTS):
        dir_path = os.path.join(directory, f"{prefix}{secrets.token_hex(8)}")
        try:
            os.mkdir(dir_path, 0o700)
        except FileExistsError:
            continue
        return dir_path
    raise FileExistsError(f"Could not reserve a unique scratch directory in {directory}")


def _create_exclusive(dir_path, dir_fd, prefix, suffix):
    """Exclusively create a scratch file (umask applied at creation) and return ``(fd, name)``."""
    for _ in range(_NAME_ATTEMPTS):
        file_name = f"{prefix}{secrets.token_hex(8)}{suffix}"
        target = file_name if dir_fd is not None else os.path.join(dir_path, file_name)
        try:
            fd = os.open(target, _CREATE_FLAGS, 0o666, dir_fd=dir_fd)
        except FileExistsError:
            continue
        return fd, file_name
    raise FileExistsError(f"Could not reserve a unique scratch file in {dir_path}")


def _verify_reservation(reservation):
    """Raise SymlinkRefusedError unless the scratch directory and file are still the ones reserved."""
    dir_stat = os.lstat(reservation.dir_path)
    if not stat.S_ISDIR(dir_stat.st_mode) or _identity(dir_stat) != reservation.dir_id:
        raise SymlinkRefusedError(f"Scratch directory was replaced: {reservation.dir_path}")
    if reservation.dir_fd is not None:
        file_stat = os.stat(reservation.file_name, dir_fd=reservation.dir_fd, follow_symlinks=False)
    else:
        file_stat = os.lstat(reservation.path)
    _verify_file(reservation, file_stat)


def _verify_file(reservation, file_stat):
    """Raise SymlinkRefusedError unless ``file_stat`` describes the reserved regular file."""
    if not stat.S_ISREG(file_stat.st_mode) or _identity(file_stat) != reservation.file_id:
        raise SymlinkRefusedError(f"Scratch file was replaced: {reservation.path}")


def _release(reservation):
    """Close held descriptors and remove the (now empty) private directory without recursing."""
    _close_quietly(reservation.fd)
    _close_quietly(reservation.dir_fd)
    reservation.fd = reservation.dir_fd = None
    with contextlib.suppress(OSError):
        if _identity(os.lstat(reservation.dir_path)) == reservation.dir_id:
            os.rmdir(reservation.dir_path)


def _identity(stat_result):
    """Return the (device, inode) pair that identifies a filesystem object."""
    return (stat_result.st_dev, stat_result.st_ino)


def _close_quietly(fd):
    """Close a descriptor if one is held, ignoring errors."""
    if fd is not None:
        with contextlib.suppress(OSError):
            os.close(fd)
