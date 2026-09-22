"""Cache corruption detection and purging utilities for downloaded AI models."""

import contextlib
import os
import shutil
import stat
import tempfile

from ..workdir import open_dir_handle
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
    geteuid = getattr(os, "geteuid", None)
    return geteuid is None or entry_stat.st_uid == geteuid()


def _unlink_matching_entries(dir_fd: int, model_filename: str, base_prefix: str) -> None:
    """Unlink every entry in the bound directory belonging to the model's file set."""
    for entry in _safe_listdir(dir_fd):
        if _is_entry_matching(entry, model_filename, base_prefix):
            _remove_matching_file(dir_fd, entry)


def _purge_directory_checkpoint_files(directory: str | None, model_filename: str, base_prefix: str) -> None:
    """Purge matching checkpoint files from a single directory, bound to a descriptor.

    Every listing and unlink runs relative to a descriptor opened with
    ``O_NOFOLLOW``, so the directory validated is the directory operated on. If
    the platform cannot bind a descriptor, the purge is abandoned rather than
    falling back to pathname access.
    """
    if not directory:
        return
    dir_fd = open_dir_handle(directory)
    if dir_fd is None:
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
