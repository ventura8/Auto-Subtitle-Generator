"""Per-video work directory: the single home for every temporary artifact.

Every intermediate the pipeline produces for one input video (extracted audio,
isolated vocals, translation manifests and worker outputs, the recorded source
language, and the private ``.asg-tmp-*`` scratch entries used for atomic
writes) lives in ``<folder>/<base_name>.asg-temp/``. Final deliverables (the
per-language SRT files and the ``_multilang`` container) stay beside the
video.

Keeping everything in one directory gives two guarantees:

* **Resumability** — after a crash or power loss the directory is left intact,
  so the next run reuses whatever stage outputs are already valid.
* **Hygiene** — once a video reaches a terminal state (muxed output written, or
  no speech found) the whole directory is removed and nothing is left behind.

Removal is deliberately shallow. The work directory is created by this
pipeline, but it sits in an untrusted input folder, so cleanup never
recurses: it unlinks regular files (and dangling links) directly inside the
work directory, empties ``.asg-tmp-*`` scratch directories one level down,
and finishes with ``rmdir``. Anything it does not recognise is left in place
and reported.

Removal is also *bound* to the directory it validated, the same way
``modules.safe_io`` binds its scratch reservations. The directory identity
(device, inode) is recorded, and on POSIX a descriptor is opened with
``O_NOFOLLOW`` so every listing, stat and unlink runs relative to that
descriptor: a work directory swapped for a symlink or junction midway
through cleanup is never followed. A nested ``.asg-tmp-*`` directory is
opened relative to that same descriptor rather than by pathname, so the
binding is never dropped partway down. Where ``dir_fd`` is unavailable
(Windows) the identity is re-verified immediately before any pathname
operation instead. Either way cleanup aborts when the identity no longer matches. The
final ``rmdir`` is safe by pathname because ``rmdir`` never follows a link
and only ever removes an empty directory.
"""

import contextlib
import json
import os
import stat
import time

from .runtime.logging_utils import log
from .safe_io import SCRATCH_PREFIX, SymlinkRefusedError, atomic_text_writer

WORK_DIR_SUFFIX = ".asg-temp"
SOURCE_STAMP_NAME = "source.json"
# Outcomes of bind_work_dir_to_source.
BIND_FRESH = "fresh"  # no work directory existed; created and stamped
BIND_SAME = "same"  # the directory belongs to this input; resume state kept
BIND_CHANGED = "changed"  # the input changed since the last run; old state discarded
_REMOVE_ATTEMPTS = 3
_REMOVE_RETRY_SECONDS = 0.5
_DIR_OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


def _probe_dir_fd_support():
    """Return True when every descriptor-relative call the cleanup relies on is available.

    ``os.supports_dir_fd`` registers ``os.stat`` for ``fstatat`` but never
    ``os.lstat``, even though ``os.lstat(..., dir_fd=...)`` is the same syscall
    with ``AT_SYMLINK_NOFOLLOW``. Probing for ``lstat`` here would read False on
    every platform and silently disable the binding, so ``stat`` is the
    capability that stands in for it.
    """
    return {os.open, os.stat, os.unlink, os.rmdir} <= os.supports_dir_fd and os.listdir in os.supports_fd


_DIR_FD_SUPPORTED = _probe_dir_fd_support()


def work_dir_path(folder, base_name):
    """Return the work directory path for ``base_name`` inside ``folder`` (no I/O)."""
    return os.path.join(folder or ".", f"{base_name}{WORK_DIR_SUFFIX}")


def work_dir_exists(folder, base_name):
    """Return True when a genuine (non-link) work directory already exists."""
    return _is_real_directory(work_dir_path(folder, base_name))


def ensure_work_dir(folder, base_name):
    """Create the work directory if needed and return its path.

    A symlink or junction planted at the work-directory name is refused with
    ``SymlinkRefusedError`` so temp files can never be redirected elsewhere.
    """
    path = work_dir_path(folder, base_name)
    if _is_real_directory(path):
        return path
    _reject_occupied_name(path)
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        # Lost a creation race: accept a real directory, refuse anything else.
        _reject_occupied_name(path)
    return path


def _reject_occupied_name(path):
    """Raise SymlinkRefusedError when ``path`` exists but is not a real directory."""
    if os.path.lexists(path) and not _is_real_directory(path):
        raise SymlinkRefusedError(f"Work directory name is occupied by a link or non-directory: {path}")


def source_stamp(video_path):
    """Return the identity (size, mtime) of the input a work directory belongs to, or None."""
    try:
        info = os.stat(video_path)
    except OSError:
        return None
    return {"size": info.st_size, "mtime_ns": info.st_mtime_ns}


def bind_work_dir_to_source(folder, base_name, video_path):
    """Discard a work directory made for a different input, then stamp it for this one.

    Resume state is keyed only by the input's folder and base name, so a video
    replaced by another with the same name (even the same duration) between a
    failed run and the retry would otherwise be transcribed from the previous
    input's extracted audio. The stamp records the input's size and mtime; a
    missing or different stamp discards the whole directory before any stage
    can reuse it. Returns ``BIND_FRESH``, ``BIND_SAME`` or ``BIND_CHANGED``,
    or None when the input cannot be stat'ed (nothing is touched then).
    """
    stamp = source_stamp(video_path)
    if stamp is None:
        return None
    outcome = _binding_outcome(folder, base_name, stamp)
    if outcome == BIND_CHANGED:
        _discard_stale_work_dir(folder, base_name, video_path)
    work_dir = ensure_work_dir(folder, base_name)
    if outcome != BIND_SAME:
        with atomic_text_writer(os.path.join(work_dir, SOURCE_STAMP_NAME), scratch_dir=work_dir) as handle:
            json.dump(stamp, handle)
    return outcome


def _binding_outcome(folder, base_name, stamp):
    """Classify an existing work directory against the current input's stamp."""
    if not work_dir_exists(folder, base_name):
        return BIND_FRESH
    stored = _read_stamp(os.path.join(work_dir_path(folder, base_name), SOURCE_STAMP_NAME))
    return BIND_SAME if stored == stamp else BIND_CHANGED


def _discard_stale_work_dir(folder, base_name, video_path):
    """Purge a work directory that belongs to a different input; fail rather than reuse it."""
    log(f"  [Resume] {os.path.basename(video_path)} changed since the last run; discarding its work directory.", "WARNING")
    if not purge_work_dir(folder, base_name):
        raise OSError(f"Could not discard the stale work directory {work_dir_path(folder, base_name)}")


def _read_stamp(stamp_path):
    """Return the stored input stamp, or None when absent, a link, or unreadable."""
    if _is_link(stamp_path):
        return None
    try:
        with open(stamp_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def purge_work_dir(folder, base_name):
    """Remove the work directory and everything the pipeline put in it.

    Returns True when the directory is gone afterwards (or never existed).
    Nothing is walked recursively; unknown sub-directories are left in place
    and reported as leftovers.
    """
    path = work_dir_path(folder, base_name)
    if not os.path.lexists(path):
        return True
    if not _is_real_directory(path):
        log(f"  [Temp] Not removing {path}: it is not a real directory.", "WARNING")
        return False
    return _purge_real_work_dir(path)


def _purge_real_work_dir(path):
    """Empty the validated work directory through a bound handle and remove it."""
    identity = _directory_identity(path)
    dir_fd = _open_dir_handle(path)
    try:
        if not _binding_is_valid(path, dir_fd, identity):
            log(f"  [Temp] Not removing {path}: it changed during cleanup.", "WARNING")
            return False
        leftovers = _purge_directory_entries(path, dir_fd)
    finally:
        _close_handle(dir_fd)
    return _report_purge_result(path, leftovers)


def _report_purge_result(path, leftovers):
    """Log leftovers, or remove the now-empty directory; True when it is gone."""
    if leftovers:
        log(f"  [Temp] Left {len(leftovers)} unexpected entries in {path}: {', '.join(sorted(leftovers))}", "WARNING")
        return False
    if not _rmdir_with_retry(path):
        log(f"  [Temp] Could not remove work directory {path}.", "WARNING")
        return False
    return True


def _purge_directory_entries(path, dir_fd):
    """Delete recognised entries inside the bound directory and return the names left behind."""
    leftovers = []
    for entry in _safe_listdir(_listdir_target(path, dir_fd)):
        if not _remove_work_entry(entry, os.path.join(path, entry), dir_fd):
            leftovers.append(entry)
    return leftovers


def _remove_work_entry(entry, entry_path, dir_fd=None):
    """Remove one entry of the work directory; return True on success.

    With a bound descriptor the stat and unlink are relative to it, so a
    replaced parent directory cannot redirect them.
    """
    target = entry if dir_fd is not None else entry_path
    if _is_unlinkable(target, dir_fd):
        return _unlink_with_retry(target, dir_fd)
    if entry.startswith(SCRATCH_PREFIX):
        return _remove_nested_scratch_dir(entry, entry_path, dir_fd)
    return False


def _remove_nested_scratch_dir(entry, entry_path, dir_fd):
    """Remove a nested ``.asg-tmp-*`` directory without leaving the parent binding.

    With a held parent descriptor the nested directory is opened relative to it
    (``O_NOFOLLOW``), emptied through its own descriptor, and removed relative
    to the parent again, so the pathname is never re-resolved from the
    filesystem root and a parent replaced after the listing cannot redirect the
    cleanup. ``O_NOFOLLOW | O_DIRECTORY`` also means a link or non-directory
    planted under a scratch name is refused rather than followed.

    Without a descriptor (Windows) ``remove_scratch_dir`` re-verifies the
    pathname identity instead, the same fallback ``modules.safe_io`` uses.
    """
    if dir_fd is None:
        return remove_scratch_dir(entry_path)
    try:
        child_fd = os.open(entry, _DIR_OPEN_FLAGS, dir_fd=dir_fd)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    try:
        if not _empty_bound_dir(entry_path, child_fd):
            return False
    finally:
        _close_handle(child_fd)
    return _rmdir_with_retry(entry, dir_fd)


def remove_scratch_dir(scratch_dir):
    """Empty one ``.asg-tmp-*`` scratch directory (files only, no recursion) and remove it.

    Cleanup is bound to the directory validated here, so a swap for a symlink
    or junction partway through is never followed. Returns True when the
    directory is gone afterwards.
    """
    identity = _directory_identity(scratch_dir)
    if identity is None:
        return not os.path.lexists(scratch_dir)
    dir_fd = _open_dir_handle(scratch_dir)
    try:
        if not _binding_is_valid(scratch_dir, dir_fd, identity):
            return False
        if not _empty_bound_dir(scratch_dir, dir_fd):
            return False
    finally:
        _close_handle(dir_fd)
    return _rmdir_with_retry(scratch_dir)


def _empty_bound_dir(path, dir_fd):
    """Unlink every file and link directly inside the bound directory; True when it is empty."""
    for name in _safe_listdir(_listdir_target(path, dir_fd)):
        target = name if dir_fd is not None else os.path.join(path, name)
        if not (_is_unlinkable(target, dir_fd) and _unlink_with_retry(target, dir_fd)):
            return False
    return True


def _directory_identity(path):
    """Return the ``(device, inode)`` of a real directory at ``path``, else None."""
    if _is_link(path):
        return None
    try:
        entry_stat = os.lstat(path)
    except OSError:
        return None
    return (entry_stat.st_dev, entry_stat.st_ino) if stat.S_ISDIR(entry_stat.st_mode) else None


def _open_dir_handle(path):
    """Open ``path`` as a directory without following links; None when unsupported or unopenable."""
    if not _DIR_FD_SUPPORTED:
        return None
    try:
        return os.open(path, _DIR_OPEN_FLAGS)
    except OSError:
        return None


def _binding_is_valid(path, dir_fd, identity):
    """Return True when cleanup may proceed against the directory that was validated.

    A held descriptor must still describe that directory. Without one, cleanup
    is only allowed where the platform has no ``dir_fd`` support at all
    (Windows), and only after re-verifying that the pathname still resolves to
    the same directory.
    """
    if identity is None:
        return False
    if dir_fd is None:
        return not _DIR_FD_SUPPORTED and _directory_identity(path) == identity
    try:
        handle_stat = os.fstat(dir_fd)
    except OSError:
        return False
    return (handle_stat.st_dev, handle_stat.st_ino) == identity


def _close_handle(dir_fd):
    """Close a held directory descriptor, ignoring errors."""
    if dir_fd is not None:
        with contextlib.suppress(OSError):
            os.close(dir_fd)


def _listdir_target(path, dir_fd):
    """Return the bound descriptor when one is held, else the pathname."""
    return path if dir_fd is None else dir_fd


def _is_unlinkable(target, dir_fd=None):
    """Return True for a regular file or a symlink (removed as a link, never followed)."""
    try:
        mode = os.lstat(target, dir_fd=dir_fd).st_mode
    except OSError:
        return False
    return stat.S_ISREG(mode) or stat.S_ISLNK(mode)


def _unlink_with_retry(target, dir_fd=None):
    """Unlink a file, retrying briefly to ride out transient Windows locks."""
    for attempt in range(_REMOVE_ATTEMPTS):
        try:
            os.unlink(target, dir_fd=dir_fd)
            return True
        except FileNotFoundError:
            return True
        except OSError:
            if attempt + 1 < _REMOVE_ATTEMPTS:
                time.sleep(_REMOVE_RETRY_SECONDS)
    return False


def _rmdir_with_retry(target, dir_fd=None):
    """Remove an empty directory, retrying briefly to ride out transient Windows locks.

    Safe by pathname: ``rmdir`` never follows a link and only removes an empty
    directory.
    """
    for attempt in range(_REMOVE_ATTEMPTS):
        try:
            os.rmdir(target, dir_fd=dir_fd)
            return True
        except FileNotFoundError:
            return True
        except OSError:
            if attempt + 1 < _REMOVE_ATTEMPTS:
                time.sleep(_REMOVE_RETRY_SECONDS)
    return False


def _safe_listdir(target):
    """List a directory by pathname or bound descriptor, returning [] when unreadable."""
    with contextlib.suppress(OSError):
        return os.listdir(target)
    return []


def _is_link(path):
    """Return True for symbolic links and Windows junctions."""
    if os.path.islink(path):
        return True
    is_junction = getattr(os.path, "isjunction", None)
    return bool(is_junction and is_junction(path))


def _is_real_directory(path):
    """Return True when ``path`` is a directory reached without following a link."""
    if _is_link(path):
        return False
    try:
        return stat.S_ISDIR(os.lstat(path).st_mode)
    except OSError:
        return False
