"""Cache corruption detection and purging utilities for downloaded AI models."""

import contextlib
import os
import shutil
import stat
import tempfile

from ..workdir import DIR_FD_SUPPORTED, open_dir_handle
from .optional_imports import resolve_hf_hub_cache

KNOWN_CORRUPT_TOKENS = (
    "failed finding central directory",
    "pytorchstreamreader failed",
    "corrupt or incomplete",
    "checkpoint file is corrupted",
    "bad zip file",
    "is not a zip file",
    "file is not a valid safetensors",
    "invalid safetensors header",
    "error loading safetensors",
    "error when deserializing header",
    "piece size is not valid",
)


def is_corrupt_model_error(error: Exception) -> bool:
    """Return True if exception message indicates a corrupted model or archive file."""
    message = str(error).lower()
    return any(token in message for token in KNOWN_CORRUPT_TOKENS)


def purge_hf_model_cache(model_id: str) -> None:
    """Purge cached HuggingFace hub snapshots for a given repository ID."""
    cache_dir = str(resolve_hf_hub_cache())
    repo_folder = f"models--{model_id.replace('/', '--')}"
    target = os.path.join(cache_dir, repo_folder)
    if os.path.isdir(target):
        try:
            shutil.rmtree(target)
        except OSError:
            pass


def purge_whisper_model_cache(model_size_or_id: str) -> None:
    """Purge cached faster-whisper hub snapshots for a given Whisper model."""
    clean_id = model_size_or_id if "/" in model_size_or_id else f"Systran/faster-whisper-{model_size_or_id}"
    purge_hf_model_cache(clean_id)


def _remove_matching_file(dir_fd: int, entry: str) -> None:
    """Unlink one matching entry relative to the held directory descriptor."""
    with contextlib.suppress(OSError):
        os.unlink(entry, dir_fd=dir_fd)


def _is_entry_matching(entry: str, model_filename: str, base_prefix: str) -> bool:
    """Return True if an entry exactly matches the model file name or its sidecar filenames."""
    allowed_entries = {
        model_filename,
        f"{base_prefix}.yaml",
        f"{base_prefix}.json",
    }
    return entry in allowed_entries


def _safe_listdir(dir_fd: int) -> list[str]:
    """List the bound directory, returning an empty list when it is unreadable."""
    with contextlib.suppress(OSError):
        return os.listdir(dir_fd)
    return []


def _is_owned_by_current_user(entry_stat: os.stat_result) -> bool:
    """Return True when ``entry_stat`` belongs to this user.

    Platforms without uids (Windows) have no owner to compare, so the check
    cannot contribute there and passes.
    """
    geteuid = getattr(os, "geteuid", None)
    return geteuid is None or entry_stat.st_uid == geteuid()


def _is_real_directory_mode(mode: int) -> bool:
    """Return True for a directory reached without following a symbolic link."""
    return stat.S_ISDIR(mode) and not stat.S_ISLNK(mode)


def _is_own_directory(dir_fd: int) -> bool:
    """Return True when the bound descriptor is a directory owned by this user.

    The separator cache may live under the shared temp directory, which is world
    writable. The check runs against the held descriptor, never the pathname, so
    a directory swapped in after the check cannot be purged instead.
    """
    try:
        entry_stat = os.fstat(dir_fd)
    except OSError:
        return False
    if not stat.S_ISDIR(entry_stat.st_mode):
        return False
    return _is_owned_by_current_user(entry_stat)


def _unlink_matching_entries(dir_fd: int, model_filename: str, base_prefix: str) -> None:
    """Unlink every entry in the bound directory belonging to the model's file set."""
    for entry in _safe_listdir(dir_fd):
        if _is_entry_matching(entry, model_filename, base_prefix):
            _remove_matching_file(dir_fd, entry)


def _is_own_real_directory(directory: str) -> bool:
    """Return True when ``directory`` is a real directory this user owns, by pathname.

    A Windows junction reports ``S_IFDIR`` from ``lstat`` and only a symbolic
    link reports ``S_IFLNK``, so the junction is rejected explicitly, the way
    ``workdir._is_link`` already does.
    """
    if os.path.isjunction(directory):
        return False
    try:
        entry_stat = os.lstat(directory)
    except OSError:
        return False
    if not _is_real_directory_mode(entry_stat.st_mode):
        return False
    return _is_owned_by_current_user(entry_stat)


def _listdir_by_pathname(directory: str) -> list[str]:
    """List ``directory`` by pathname, returning an empty list when unreadable."""
    with contextlib.suppress(OSError):
        return os.listdir(directory)
    return []


def _is_regular_file(path: str) -> bool:
    """Return True when ``path`` is a real file reached without following a link."""
    try:
        entry_stat = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISREG(entry_stat.st_mode)


def _purge_by_pathname(directory: str, model_filename: str, base_prefix: str) -> None:
    """Purge by pathname on platforms with no descriptor-relative calls at all.

    Reached only where ``os.supports_dir_fd`` is empty, which in practice means
    Windows. Binding is impossible there, and refusing outright would leave
    corrupt-checkpoint recovery permanently broken on the project's primary
    target OS, so this is an accepted platform boundary rather than an
    equivalent of the bound path. Two properties narrow it: the candidate
    directories are per-user on Windows (``%TEMP%`` and the profile cache), not
    the world-writable shared ``/tmp`` that motivates the binding on POSIX; and
    each entry is re-checked to be a real file, so a link planted in place of a
    checkpoint is skipped rather than unlinked. A directory swapped between the
    check and the unlink remains possible here, unlike on the bound path.
    """
    if not _is_own_real_directory(directory):
        return
    for entry in _listdir_by_pathname(directory):
        if not _is_entry_matching(entry, model_filename, base_prefix):
            continue
        entry_path = os.path.join(directory, entry)
        if not _is_regular_file(entry_path):
            continue
        with contextlib.suppress(OSError):
            os.unlink(entry_path)


def _purge_directory_checkpoint_files(directory: str | None, model_filename: str, base_prefix: str) -> None:
    """Purge matching checkpoint files from a single directory.

    Where the platform supports descriptor-relative calls, every listing and
    unlink runs relative to a descriptor opened with ``O_NOFOLLOW``, so the
    directory validated is the directory operated on; a failure to bind there
    can mean a link or a swapped directory, so the purge is abandoned. Only a
    platform with no descriptor support at all falls back to pathnames, which
    mirrors the rule ``workdir._binding_is_valid`` already applies.
    """
    if not directory:
        return
    dir_fd = open_dir_handle(directory)
    if dir_fd is None:
        if not DIR_FD_SUPPORTED:
            _purge_by_pathname(directory, model_filename, base_prefix)
        return
    try:
        if _is_own_directory(dir_fd):
            _unlink_matching_entries(dir_fd, model_filename, base_prefix)
    finally:
        with contextlib.suppress(OSError):
            os.close(dir_fd)


def purge_separator_checkpoint(model_filename: str, model_file_dir: str | None = None) -> None:
    """Remove corrupted separator model checkpoints and configs from cache."""
    candidate_dirs = [
        model_file_dir,
        # Honour TMPDIR rather than assuming /tmp, and never trust the result
        # blindly: _purge_directory_checkpoint_files vets each directory first.
        os.path.join(tempfile.gettempdir(), "audio-separator-models"),
        os.path.expanduser("~/.cache/audio-separator-models"),
    ]
    base_prefix = os.path.splitext(model_filename)[0]
    for directory in candidate_dirs:
        _purge_directory_checkpoint_files(directory, model_filename, base_prefix)
