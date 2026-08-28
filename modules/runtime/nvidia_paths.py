"""CUDA runtime library-path discovery used during application startup."""

import ctypes
import importlib
import os
import site
import sys
from typing import Any

_DLL_DIRECTORY_HANDLES: list[Any] = []

# Only these CUDA runtime libraries are preloaded; anything else in a bundled
# NVIDIA directory is left for the dynamic loader to resolve on demand.
_PRELOAD_ALLOWLIST = ("libcublas", "libcublaslt", "libcudnn", "libcudart", "libnvrtc")


def _collect_existing_runtime_dirs(base_dir):
    """Collect existing runtime subdirectories used by CUDA providers."""
    return [candidate for dir_name in ["bin", "lib"] if os.path.exists(candidate := os.path.join(base_dir, dir_name))]


def _get_nvidia_bin_lib_paths(site_package):
    """Find runtime bin and lib directories in NVIDIA provider packages."""
    nvidia_path = os.path.join(site_package, "nvidia")
    if not os.path.exists(nvidia_path):
        return []
    return [
        path
        for item in os.listdir(nvidia_path)
        if os.path.isdir(sub_path := os.path.join(nvidia_path, item))
        for path in _collect_existing_runtime_dirs(sub_path)
    ]


def _collect_site_packages_paths():
    """Collect site-packages paths for runtime DLL discovery."""
    site_packages = list(site.getsitepackages())
    for manual_rel in [
        os.path.join("Lib", "site-packages"),
        os.path.join("lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages"),
    ]:
        manual_site = os.path.join(sys.prefix, manual_rel)
        if os.path.isdir(manual_site) and manual_site not in site_packages:
            site_packages.append(manual_site)
    return site_packages


def _add_dll_directory_if_supported(path):
    """Register DLL directory on supported platforms without hard failure."""
    if not hasattr(os, "add_dll_directory"):
        return
    try:
        handle = os.add_dll_directory(path)
        if handle is not None:
            _DLL_DIRECTORY_HANDLES.append(handle)
    except (AttributeError, OSError):
        return


def _update_dynamic_linker_env(path):
    """Ensure dynamic linker paths include an NVIDIA runtime directory.

    The loader reads these variables at process start, so the updates only take
    effect for child processes. macOS may strip DYLD_* variables for protected
    binaries. The current process resolves the same libraries through
    _preload_runtime_libraries, which loads them via ctypes directly.
    """
    for env_var in ["LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"]:
        current_paths = os.environ.get(env_var, "").split(os.pathsep)
        if path not in current_paths:
            os.environ[env_var] = path + os.pathsep + os.environ[env_var] if os.environ.get(env_var) else path


def _preload_shared_library(filepath):
    """Attempt global ctypes preloading for a shared library."""
    if _should_skip_preload(filepath):
        return
    try:
        _load_shared_library(filepath)
    except (OSError, ValueError, TypeError):
        return


def _should_skip_preload(filepath):
    """Return whether a path is unsafe or irrelevant for shared-library loading."""
    return _is_nvblas_library(filepath) or not _is_shared_library(filepath)


def _is_nvblas_library(filepath):
    """Identify nvblas, which must not be loaded into CPU-only processes."""
    return "nvblas" in os.path.basename(filepath).lower()


def _is_shared_library(filepath):
    """Identify supported Unix and macOS shared-library filenames."""
    return filepath.endswith(".so") or ".so." in filepath or filepath.endswith(".dylib")


def _load_shared_library(filepath):
    """Load a library using global symbol visibility when supported."""
    ctypes.CDLL(filepath, mode=getattr(ctypes, "RTLD_GLOBAL", 0))


def _is_allowlisted_runtime_library(filename):
    """Return True for the CUDA runtime libraries we intentionally preload."""
    lowered = os.path.basename(filename).lower()
    return any(lowered.startswith(prefix) for prefix in _PRELOAD_ALLOWLIST)


def _preload_runtime_libraries(path):
    """Preload allowlisted CUDA libraries by absolute path from a runtime directory."""
    if not os.path.isdir(path):
        return
    try:
        for filename in sorted(os.listdir(path)):
            if _is_allowlisted_runtime_library(filename):
                _preload_shared_library(os.path.join(path, filename))
    except (OSError, PermissionError):
        return


def _apply_paths_to_env(paths):
    """Update PATH, dynamic linker paths, and Windows DLL directories."""
    raw_path = os.environ.get("PATH", "")
    normalized_entries = {os.path.normcase(os.path.normpath(entry)) for entry in raw_path.split(os.pathsep) if entry}
    for path in paths:
        _add_dll_directory_if_supported(path)
        normalized_path = os.path.normcase(os.path.normpath(path))
        if normalized_path not in normalized_entries:
            os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
            normalized_entries.add(normalized_path)
        _update_dynamic_linker_env(path)
        _preload_runtime_libraries(path)


def _collect_torch_lib_paths(torch_module):
    """Collect torch library paths needed for CUDA runtime discovery."""
    if torch_module is None or not hasattr(torch_module, "__path__"):
        return []
    return [lib_path for package_path in torch_module.__path__ if os.path.exists(lib_path := os.path.join(package_path, "lib"))]


def _collect_nvidia_runtime_paths():
    """Collect CUDA runtime directories bundled in installed NVIDIA packages."""
    return [path for site_package in _collect_site_packages_paths() for path in _get_nvidia_bin_lib_paths(site_package)]


def _import_onnxruntime_if_available():
    """Import ONNX Runtime after CUDA library paths have been prepared."""
    try:
        importlib.import_module("onnxruntime")
    except ImportError:
        pass


def _load_optional_torch():
    """Load Torch through the startup-safe optional import helper."""
    optional_imports = importlib.import_module("modules.runtime.optional_imports")
    return optional_imports.load_optional_torch()


def is_cuda_explicitly_disabled():
    """Return True when the environment opts out of CUDA entirely.

    An empty (or whitespace-only) CUDA_VISIBLE_DEVICES is the standard CPU-only
    opt-out, used by the --cpu flag and by CPU-only test subprocesses.
    """
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    return visible_devices is not None and not visible_devices.strip()


def prepare_nvidia_paths():
    """Prepare bundled NVIDIA runtime paths before importing torch.

    Skipped when CUDA is explicitly disabled: the bundled cuBLAS directory also
    contains libnvblas.so, which hijacks CPU BLAS resolution and aborts when no
    GPU is present.
    """
    if is_cuda_explicitly_disabled():
        return
    _apply_paths_to_env(_collect_nvidia_runtime_paths())


def load_nvidia_paths(torch_module=None):
    """Prepare CUDA library paths before loading ONNX Runtime."""
    if is_cuda_explicitly_disabled():
        return
    if torch_module is None:
        torch_module = _load_optional_torch()
    _apply_paths_to_env(_collect_torch_lib_paths(torch_module))
    _import_onnxruntime_if_available()
