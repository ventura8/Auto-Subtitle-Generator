"""File collection and path resolution utilities."""

import os

from ..configuration import config
from ..runtime.logging_utils import log


def collect_video_files(path):
    """Public wrapper for collecting supported video inputs."""
    return _collect_video_files(path)


def resolve_input_path(input_path):
    """Public wrapper for resolving user-provided input paths."""
    return _resolve_input_path(input_path)


def _collect_video_files(path):
    """Collect supported input videos from a file or directory path."""
    files = []
    supported_extensions = _get_supported_video_extensions()

    if os.path.isfile(path):
        file_name = os.path.basename(path)
        if _is_supported_video_file(file_name, supported_extensions):
            files.append(os.path.abspath(path))
        return files

    if os.path.isdir(path):
        return _collect_from_directory(path, supported_extensions)

    log(f"Error: Path not found: {path}", "CRITICAL")
    raise FileNotFoundError(path)


def _resolve_input_path(input_path):
    """Resolve the input path from CLI args or prompt."""
    path = input_path
    if not path:
        print(">> Please Drag & Drop a video file here and press Enter:")
        path = input(">>Path: ").strip().strip('"')
    return path or "input"


def _get_supported_video_extensions():
    """Normalize configured video extensions to lowercase dotted form."""
    return {(ext if str(ext).startswith(".") else f".{ext}").lower() for ext in config.VIDEO_EXTENSIONS}


def _is_supported_video_file(file_name, supported_extensions):
    """Return True when file extension is supported and file is not a generated output."""
    file_stem, file_ext = os.path.splitext(file_name)
    if file_ext.lower() not in supported_extensions:
        return False
    return not file_stem.lower().endswith("_multilang")


def _collect_from_directory(path, supported_extensions):
    """Collect all supported video files from a directory tree."""
    files = []
    for root, _, filenames in os.walk(path):
        for file_name in filenames:
            if not _is_supported_video_file(file_name, supported_extensions):
                continue
            files.append(os.path.abspath(os.path.join(root, file_name)))
    return files
