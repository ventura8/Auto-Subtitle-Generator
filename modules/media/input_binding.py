"""Descriptor-bound input videos.

``collect_video_files`` filters symlinks out of the input folder, but that check
is on a *pathname*. Between collection and the FFprobe/FFmpeg opens (model
loading can take minutes) a party with write access to the input folder could
swap the validated regular file, or one of its parent directories, for a
symlink. ``bind_input`` closes that gap by opening the file itself and handing
FFmpeg the open descriptor rather than the name:

* POSIX: the user-selected input directory (``set_input_root``) is opened as a
  descriptor, every directory component below it and the file itself are
  opened relative to the previous descriptor with ``O_NOFOLLOW``, ``fstat``
  confirms a regular file, and FFmpeg/FFprobe receive ``/dev/fd/N`` for that
  exact open file. ``/dev/fd/N`` may share the file offset with our descriptor
  (macOS ``dup`` semantics), so ``rewind_inputs`` runs right before every
  child is spawned.
* Windows: there is no ``O_NOFOLLOW`` and no fd passing, and a held file
  handle pins only the file, not the directory chain FFmpeg would walk by
  pathname. Each component below the root is checked for links/junctions,
  the file is opened after an ``lstat`` symlink check and ``fstat`` is
  compared against it, and the bytes are then copied *from that handle* into
  a private, randomly named directory beside the input. FFmpeg/FFprobe read
  the copy, whose location an attacker cannot predict or re-point without
  making the run fail outright. The copy is removed when the binding closes.

The binding is active for the ``with`` block and looked up by pathname via
``media_source``/``stat_input`` so the pipeline's string-based call chain stays
unchanged. Without a registered root (direct API use) only the final
component is protected and the parent directory is opened by pathname.
"""

import contextlib
import os
import shutil
import stat
import sys
import tempfile

_IS_POSIX = sys.platform != "win32"
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_FILE_FLAGS = os.O_RDONLY | _NOFOLLOW | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_BINARY", 0)
_DIR_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
_DIR_FD_SUPPORTED = _IS_POSIX and os.open in os.supports_dir_fd
_PRIVATE_COPY_PREFIX = ".asg-input-"

_active_bindings: dict[str, "BoundInput"] = {}
# Single slot: (pathname form of the selected input directory, its resolved form) or None.
# Collected paths are matched against the pathname form; the resolved form, taken at
# selection time, is what gets opened, so a link planted at the root later is refused.
_trusted_root: list[tuple[str, str] | None] = [None]


class InputRefusedError(OSError):
    """Raised when an input video is not a plain regular file bound to a descriptor."""


class BoundInput:
    """An open descriptor for an input video plus the arguments FFmpeg needs to read it."""

    def __init__(self, path, fd):
        self.path = path
        self.fd = fd
        self._private_copy: str | None = None

    def __fspath__(self):
        return self.path

    def media_source(self):
        """Return ``(ffmpeg input argument, fds to inherit)`` for a child process."""
        if _IS_POSIX:
            return f"/dev/fd/{self.fd}", (self.fd,)
        return self._private_copy_path(), ()

    def _private_copy_path(self):
        """Copy the bound bytes into a private directory once and return the copy's path."""
        if self._private_copy is None:
            copy_dir = tempfile.mkdtemp(prefix=_PRIVATE_COPY_PREFIX, dir=os.path.dirname(self.path))
            self._private_copy = os.path.join(copy_dir, os.path.basename(self.path))
            os.lseek(self.fd, 0, os.SEEK_SET)
            with os.fdopen(os.dup(self.fd), "rb") as source, open(self._private_copy, "xb") as target:
                shutil.copyfileobj(source, target)
        return self._private_copy

    def close(self):
        """Close the descriptor and remove the private copy, if one was made."""
        if self.fd is not None:
            os.close(self.fd)
        self.fd = None
        if self._private_copy is not None:
            # Best effort by pathname: if the directory chain was swapped mid-run the copy
            # (our own bytes, nothing foreign) is left behind in the attacker-controlled folder.
            with contextlib.suppress(OSError):
                os.remove(self._private_copy)
                os.rmdir(os.path.dirname(self._private_copy))
            self._private_copy = None


def set_input_root(path):
    """Record the user-selected input directory; everything below it is opened without following links."""
    if not path:
        _trusted_root[0] = None
        return
    root = os.path.abspath(os.fspath(path))
    _trusted_root[0] = (root, os.path.realpath(root))


def is_link(path):
    """Return True for a symlink or a Windows directory junction."""
    return os.path.islink(path) or getattr(os.path, "isjunction", lambda _p: False)(path)


def _split_trusted(path):
    """Return ``(trusted directory, [components to open without following links])`` for ``path``."""
    trusted = _trusted_root[0]
    if trusted is not None:
        root, resolved_root = trusted
        if path != root and os.path.commonpath([path, root]) == root:
            return resolved_root, os.path.relpath(path, root).split(os.sep)
    return os.path.dirname(path) or os.curdir, [os.path.basename(path)]


def _refuse(fd, message):
    if fd is not None:
        os.close(fd)
    raise InputRefusedError(message)


def _open_regular_posix(path):
    """Open ``path`` component by component below the trusted directory, never following a link."""
    trusted_dir, components = _split_trusted(path)
    dir_fd = _open_trusted_dir(trusted_dir, path)
    try:
        for name in components[:-1]:
            next_fd = _open_component(name, dir_fd, path)
            os.close(dir_fd)
            dir_fd = next_fd
        fd = os.open(components[-1], _FILE_FLAGS, dir_fd=dir_fd)
    except OSError as e:
        raise InputRefusedError(f"Refusing input {path}: {e.strerror or e}") from e
    finally:
        os.close(dir_fd)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        _refuse(fd, f"Refusing input {path}: not a regular file")
    return fd


def _open_trusted_dir(trusted_dir, path):
    """Open the trusted directory itself without following a link planted at its name."""
    try:
        return os.open(trusted_dir, _DIR_FLAGS | _NOFOLLOW)
    except OSError as e:
        raise InputRefusedError(f"Refusing input {path}: cannot open {trusted_dir} without following links") from e


def _open_component(name, dir_fd, path):
    """Open directory ``name`` relative to ``dir_fd`` without following links; ENOTDIR/ELOOP means a link."""
    try:
        return os.open(name, _DIR_FLAGS | _NOFOLLOW, dir_fd=dir_fd)
    except NotADirectoryError as e:
        raise InputRefusedError(f"Refusing input {path}: directory {name!r} was replaced by a link") from e


def _open_regular_fallback(path):
    """Open ``path`` on platforms without O_NOFOLLOW, verifying the opened file is the lstat'ed one."""
    _reject_linked_components(path)
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


def _reject_linked_components(path):
    """Refuse when any directory between the trusted directory and ``path`` is a link or junction."""
    current, components = _split_trusted(path)
    if is_link(current):
        _refuse(None, f"Refusing input {path}: {current} is a link")
    for name in components[:-1]:
        current = os.path.join(current, name)
        if is_link(current):
            _refuse(None, f"Refusing input {path}: {current} is a link")


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
        # Share the outer binding (and any private copy it made); only the outer block closes it.
        yield outer
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


def stat_input(path):
    """``os.stat`` for ``path``, taken from the bound descriptor when one is active."""
    bound = _active_bindings.get(os.path.abspath(os.fspath(path)))
    if bound is None:
        return os.stat(path)
    return os.fstat(bound.fd)


def rewind_inputs(pass_fds):
    """Rewind inherited input descriptors; call immediately before spawning each child."""
    for fd in pass_fds:
        os.lseek(fd, 0, os.SEEK_SET)
