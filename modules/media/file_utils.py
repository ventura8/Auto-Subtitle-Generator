"""File collection and path resolution utilities.

Input folders are untrusted (USB drops, shared directories, returned zips). A
planted symlink named ``clip.mp4`` would otherwise be read by FFmpeg and the
target stream-copied into the untrusted folder, so collection accepts only
regular files whose real path stays inside the selected input.
"""

import os
import stat

from ..configuration import config
from ..runtime.logging_utils import log
from .input_binding import is_link


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

    if is_link(path):
        log(f"Skipping symlink or junction input: {path}", "WARNING")
        return files

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
        path = _normalize_input_path(input(">>Path: "))
    else:
        path = _normalize_input_path(path)
    return path or "input"


def _normalize_input_path(path):
    """Trim whitespace and one matching pair of surrounding path quotes."""
    normalized_path = str(path).strip()
    if _has_matching_surrounding_quotes(normalized_path):
        return normalized_path[1:-1].strip()
    return normalized_path


def _has_matching_surrounding_quotes(path):
    """Return whether a path begins and ends with the same supported quote."""
    if len(path) < 2:
        return False
    return path[0] in "'\"" and path[0] == path[-1]


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
    """Collect supported regular video files that stay inside the input directory tree."""
    files = []
    input_root = os.path.realpath(path)
    for root, dir_names, filenames in os.walk(path, followlinks=False):
        # os.walk never descends into symlinked directories, but junctions and any
        # later pass are another matter: drop every linked directory explicitly.
        dir_names[:] = [name for name in dir_names if not is_link(os.path.join(root, name))]
        files.extend(_collect_safe_files(root, filenames, supported_extensions, input_root))
    return files


def _collect_safe_files(root, filenames, supported_extensions, input_root):
    """Return absolute paths of supported, symlink-free files from one walked directory."""
    candidates = (os.path.abspath(os.path.join(root, name)) for name in filenames if _is_supported_video_file(name, supported_extensions))
    return [file_path for file_path in candidates if _is_safe_input_file(file_path, input_root)]


def _is_safe_input_file(file_path, input_root):
    """Return True for a regular (non-symlink) file whose real path is inside ``input_root``."""
    try:
        if not stat.S_ISREG(os.lstat(file_path).st_mode):
            log(f"Skipping non-regular or symlink input: {file_path}", "WARNING")
            return False
    except OSError as e:
        log(f"Skipping unreadable input {file_path}: {e}", "WARNING")
        return False

    real_path = os.path.realpath(file_path)
    if os.path.commonpath([real_path, input_root]) != input_root:
        log(f"Skipping input outside the selected folder: {file_path}", "WARNING")
        return False
    return True
