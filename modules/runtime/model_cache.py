"""Cache corruption detection and purging utilities for downloaded AI models."""

import os
import shutil
import stat
import tempfile

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


def _remove_matching_file(target_path: str) -> None:
    """Safely attempt to remove a single target file."""
    try:
        os.remove(target_path)
    except OSError:
        pass


def _is_entry_matching(entry: str, model_filename: str, base_prefix: str) -> bool:
    """Return True if an entry exactly matches the model file name or its sidecar filenames."""
    allowed_entries = {
        model_filename,
        f"{base_prefix}.yaml",
        f"{base_prefix}.json",
    }
    return entry in allowed_entries


def _is_safe_purge_directory(directory: str) -> bool:
    """Reject a cache directory that another user could have planted or replaced.

    The separator cache may live under the shared temp directory, which is world
    writable. Purging there blindly would follow whatever a different user left
    behind, so only a real directory owned by this user is accepted.
    """
    try:
        entry_stat = os.lstat(directory)
    except OSError:
        return False
    if stat.S_ISLNK(entry_stat.st_mode):
        return False
    geteuid = getattr(os, "geteuid", None)
    if geteuid is not None and entry_stat.st_uid != geteuid():
        return False
    return True


def _safe_listdir(directory: str | None) -> list[str]:
    """Safely list directory contents, returning empty list on failure or missing path."""
    if not directory or not os.path.isdir(directory):
        return []
    try:
        return os.listdir(directory)
    except OSError:
        return []


def _purge_directory_checkpoint_files(directory: str | None, model_filename: str, base_prefix: str) -> None:
    """Purge matching checkpoint files from a single directory."""
    if not directory or not _is_safe_purge_directory(directory):
        return
    for entry in _safe_listdir(directory):
        if _is_entry_matching(entry, model_filename, base_prefix):
            _remove_matching_file(os.path.join(directory, entry))


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
