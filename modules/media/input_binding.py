"""Descriptor-bound input videos.

``collect_video_files`` filters symlinks out of the input folder, but that check
is on a *pathname*. Between collection and the FFprobe/FFmpeg opens (model
loading can take minutes) a party with write access to the input folder could
swap the validated regular file for a symlink. ``bind_input`` closes that gap by
opening the file itself and handing FFmpeg the open descriptor rather than the
name:

* POSIX: the parent directory is opened as a descriptor, the file is opened
  relative to it with ``O_NOFOLLOW``, ``fstat`` confirms a regular file, and
  FFmpeg/FFprobe receive ``/dev/fd/N`` for that exact open file.
* Windows: there is no ``O_NOFOLLOW`` and no fd passing. The file is opened
  after an ``lstat`` symlink check and ``fstat`` is compared against it; the
  open handle then denies rename/delete of the file for as long as it is held,
  so the pathname FFmpeg opens cannot be re-pointed while processing runs.

The binding is active for the ``with`` block and looked up by pathname via
``media_source`` so the pipeline's string-based call chain stays unchanged.
"""

import contextlib
import os
import stat
import sys

_IS_POSIX = sys.platform != "win32"
_FILE_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_BINARY", 0)
_DIR_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
_DIR_FD_SUPPORTED = _IS_POSIX and os.open in os.supports_dir_fd

_active_bindings = {}


class InputRefusedError(OSError):
    """Raised when an input video is not a plain regular file bound to a descriptor."""


class BoundInput:
    """An open descriptor for an input video plus the arguments FFmpeg needs to read it."""

    def __init__(self, path, fd, owns_fd=True):
        self.path = path
        self.fd = fd
        self._owns_fd = owns_fd

    def __fspath__(self):
        return self.path

    def media_source(self):
        """Return ``(ffmpeg input argument, fds to inherit)`` for a child process."""
        if not _IS_POSIX:
            return self.path, ()
        # /dev/fd/N may share the file offset with our descriptor (macOS dup semantics);
        # children run one at a time, so rewind before each one.
        os.lseek(self.fd, 0, os.SEEK_SET)
        return f"/dev/fd/{self.fd}", (self.fd,)

    def close(self):
        """Close the descriptor if this binding owns it."""
        if self._owns_fd and self.fd is not None:
            os.close(self.fd)
        self.fd = None


def _refuse(fd, message):
    if fd is not None:
        os.close(fd)
    raise InputRefusedError(message)


def _open_regular_posix(path):
    """Open ``path`` relative to its parent directory descriptor without following the final link."""
    parent = os.path.dirname(path) or "."
    dir_fd = os.open(parent, _DIR_FLAGS)
    try:
        fd = os.open(os.path.basename(path), _FILE_FLAGS, dir_fd=dir_fd)
    except OSError as e:
        raise InputRefusedError(f"Refusing input {path}: {e.strerror or e}") from e
    finally:
        os.close(dir_fd)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        _refuse(fd, f"Refusing input {path}: not a regular file")
    return fd


def _open_regular_fallback(path):
    """Open ``path`` on platforms without O_NOFOLLOW, verifying the opened file is the lstat'ed one."""
    try:
        before = os.lstat(path)
    except OSError as e:
        raise InputRefusedError(f"Refusing input {path}: {e.strerror or e}") from e
    if not stat.S_ISREG(before.st_mode):
        _refuse(None, f"Refusing input {path}: not a regular file")
    fd = os.open(path, _FILE_FLAGS)
    if not _same_regular_file(before, os.fstat(fd)):
        _refuse(fd, f"Refusing input {path}: file changed while opening")
    return fd


def _same_regular_file(before, after):
    """Return True when ``after`` is a regular file with the same device/inode identity as ``before``."""
    return stat.S_ISREG(after.st_mode) and (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)


def open_bound_input(path):
    """Open ``path`` as a regular file and return a ``BoundInput`` (caller closes)."""
    path = os.path.abspath(os.fspath(path))
    fd = _open_regular_posix(path) if _DIR_FD_SUPPORTED else _open_regular_fallback(path)
    return BoundInput(path, fd)


@contextlib.contextmanager
def bind_input(path, strict=True):
    """Bind ``path`` to an open descriptor for the block; nested binds reuse the outer one.

    With ``strict=False`` a file that cannot be bound yields ``None`` instead of
    raising, for callers that only hold the binding on behalf of a stricter one.
    """
    key = os.path.abspath(os.fspath(path))
    outer = _active_bindings.get(key)
    if outer is not None:
        yield BoundInput(outer.path, outer.fd, owns_fd=False)
        return
    try:
        bound = open_bound_input(key)
    except OSError:
        if strict:
            raise
        yield None
        return
    _active_bindings[key] = bound
    try:
        yield bound
    finally:
        del _active_bindings[key]
        bound.close()


def media_source(path):
    """Return ``(ffmpeg input argument, fds to inherit)`` for ``path``, descriptor-bound when active."""
    bound = _active_bindings.get(os.path.abspath(os.fspath(path)))
    if bound is None:
        return os.fspath(path), ()
    return bound.media_source()
