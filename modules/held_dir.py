"""A work directory pinned for the reads and writes of a long-running process.

The translation worker validates the manifest paths against the work directory
and then spends minutes loading a model before it touches them. Opening those
validated names afterwards would follow whatever the input folder's owner put
there in the meantime: a link at a job file, or the whole work directory
replaced by a link to another directory.

``HeldDirectory`` opens the directory once, at validation time, and routes
every later operation through it:

* POSIX: the directory is held as an ``O_NOFOLLOW`` descriptor and every entry
  is opened, created, renamed or unlinked relative to it (``dir_fd``), so a
  replaced work directory cannot redirect anything. Entries are opened with
  ``O_NOFOLLOW`` and must be regular files.
* Windows: there is no ``dir_fd``. The directory's identity (device, inode) is
  recorded and re-checked, together with a link/junction check, before every
  operation; reads ``lstat`` the entry, refuse a link and compare the opened
  file's identity with it.

Writes create a fresh ``.asg-tmp-*`` entry exclusively (``O_EXCL``), write
through that same descriptor, and rename it onto the destination only if the
entry is still the one created. Nothing is ever reopened by name.
"""

import contextlib
import os
import secrets
import stat

from modules.media.input_binding import is_link
from modules.safe_io import DIR_FD_SUPPORTED, SCRATCH_PREFIX, SymlinkRefusedError

_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_COMMON_FLAGS = _NOFOLLOW | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_BINARY", 0)
_DIR_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | _NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
# O_NONBLOCK keeps a planted FIFO from blocking the open before the regular-file check.
_READ_FLAGS = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | _COMMON_FLAGS
_CREATE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _COMMON_FLAGS


def _identity(stat_result):
    """Return the (device, inode) pair that identifies a filesystem object."""
    return (stat_result.st_dev, stat_result.st_ino)


def _is_expected_regular(before, opened):
    """True for a regular file that is, when ``before`` was taken, the same object as ``before``."""
    if not stat.S_ISREG(opened.st_mode):
        return False
    return before is None or _identity(before) == _identity(opened)


def _plain_directory_identity(path):
    """Return the identity of ``path``, refusing a link, a junction or anything but a directory."""
    before = os.lstat(path)
    if is_link(path) or not stat.S_ISDIR(before.st_mode):
        raise SymlinkRefusedError(f"Refusing work directory that is not a plain directory: {path}")
    return _identity(before)


class HeldDirectory:
    """A directory whose entries are reached through a held descriptor or a re-verified identity."""

    def __init__(self, path):
        self.path = path
        self.identity = _plain_directory_identity(path)
        self.fd = os.open(path, _DIR_FLAGS) if DIR_FD_SUPPORTED else None
        if self.fd is not None and _identity(os.fstat(self.fd)) != self.identity:
            self.close()
            raise SymlinkRefusedError(f"Work directory changed while it was opened: {path}")

    def close(self):
        """Release the directory descriptor."""
        if self.fd is not None:
            os.close(self.fd)
        self.fd = None

    def entry_name(self, path):
        """Return the name of ``path``, which must sit directly inside this directory."""
        if os.path.dirname(path) != self.path:
            raise SymlinkRefusedError(f"Path is not directly inside {self.path}: {path}")
        return os.path.basename(path)

    def exists(self, path):
        """Return True when an entry (of any kind, links included) exists at ``path``."""
        try:
            self._stat(self.entry_name(path))
        except FileNotFoundError:
            return False
        return True

    def read_text(self, path, encoding="utf-8"):
        """Return the text of the regular file at ``path``, refusing a link."""
        fd = self._open_regular(self.entry_name(path))
        with os.fdopen(fd, "r", encoding=encoding) as handle:
            return handle.read()

    def remove(self, path):
        """Unlink the entry at ``path`` (a link itself, never its target); a missing entry is fine."""
        name = self.entry_name(path)
        self._verify()
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self._target(name), dir_fd=self.fd)

    @contextlib.contextmanager
    def text_writer(self, path, encoding="utf-8"):
        """Yield a text handle whose content replaces ``path`` atomically on success."""
        name = self.entry_name(path)
        temp_name, fd = self._create_temp()
        temp_id = _identity(os.fstat(fd))
        promoted = False
        try:
            with os.fdopen(fd, "w", encoding=encoding) as handle:
                yield handle
            self._promote(temp_name, temp_id, name)
            promoted = True
        finally:
            if not promoted:
                self._discard(temp_name, temp_id)

    def _target(self, name):
        """Return the argument for an operation on ``name``: relative with a descriptor, joined without."""
        return name if self.fd is not None else os.path.join(self.path, name)

    def _verify(self):
        """Without a descriptor, refuse when the directory was replaced since it was held."""
        if self.fd is not None:
            return
        if is_link(self.path) or _identity(os.lstat(self.path)) != self.identity:
            raise SymlinkRefusedError(f"Work directory was replaced: {self.path}")

    def _stat(self, name):
        """``lstat`` an entry through the held directory."""
        self._verify()
        return os.stat(self._target(name), dir_fd=self.fd, follow_symlinks=False)

    def _open_regular(self, name):
        """Open a regular file for reading without following a link at its name."""
        before = None if self.fd is not None else self._stat(name)
        fd = os.open(self._target(name), _READ_FLAGS, dir_fd=self.fd)
        if not _is_expected_regular(before, os.fstat(fd)):
            os.close(fd)
            raise SymlinkRefusedError(f"Refusing to read {name}: not the regular file it was")
        return fd

    def _create_temp(self):
        """Exclusively create a fresh scratch entry and return ``(name, fd)``."""
        self._verify()
        temp_name = f"{SCRATCH_PREFIX}{secrets.token_hex(8)}.tmp"
        return temp_name, os.open(self._target(temp_name), _CREATE_FLAGS, 0o666, dir_fd=self.fd)

    def _promote(self, temp_name, temp_id, name):
        """Rename the scratch entry onto ``name`` if it is still the file that was created."""
        if _identity(self._stat(temp_name)) != temp_id:
            raise SymlinkRefusedError(f"Scratch entry was replaced: {temp_name}")
        if self.fd is None and is_link(self._target(name)):
            raise SymlinkRefusedError(f"Refusing to write through symlink: {self._target(name)}")
        os.replace(self._target(temp_name), self._target(name), src_dir_fd=self.fd, dst_dir_fd=self.fd)

    def _discard(self, temp_name, temp_id):
        """Best-effort removal of the scratch entry, only while it is still the file that was created."""
        with contextlib.suppress(OSError):
            if _identity(self._stat(temp_name)) == temp_id:
                os.unlink(self._target(temp_name), dir_fd=self.fd)
