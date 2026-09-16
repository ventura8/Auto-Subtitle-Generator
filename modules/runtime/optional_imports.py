"""Helpers for optional third-party imports used across modules."""

import importlib
import os
import threading
from typing import Any

from . import nvidia_paths
from .bootstrap import bootstrap_cpu_env

_PREPARE_STATE = {"nvidia_paths_prepared": False}
_PREPARE_LOCK = threading.Lock()


def _prepare_nvidia_paths_once() -> None:
    """Prepare bundled CUDA runtime paths exactly once before Torch is imported."""
    with _PREPARE_LOCK:
        if _PREPARE_STATE["nvidia_paths_prepared"]:
            return
        bootstrap_cpu_env()
        nvidia_paths.prepare_nvidia_paths()
        _PREPARE_STATE["nvidia_paths_prepared"] = True


def load_optional_torch() -> Any | None:
    """Return torch when installed, else None.

    CUDA runtime discovery runs first so that the bundled NVIDIA runtime
    directories are on the library search path before Torch initializes.
    """
    _prepare_nvidia_paths_once()
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        if getattr(exc, "name", None) not in {"torch", "torch.__init__"}:
            raise
        return None


def is_cuda_usable(torch_module: Any | None = None) -> bool:
    """Return whether a CUDA device can actually be used.

    ``torch.cuda.is_available()`` alone is not enough: with the GPU hidden via
    ``CUDA_VISIBLE_DEVICES`` this torch build still reports availability, so
    the device count is required to be positive as well.
    """
    if torch_module is None or not hasattr(torch_module, "cuda"):
        return False
    try:
        return bool(torch_module.cuda.is_available() and torch_module.cuda.device_count() > 0)
    except (RuntimeError, AttributeError):
        return False


def is_mps_available(torch_module: Any | None = None) -> bool:
    """Return whether the provided torch runtime exposes an available MPS backend."""
    return bool(
        torch_module is not None
        and hasattr(torch_module, "backends")
        and hasattr(torch_module.backends, "mps")
        and torch_module.backends.mps.is_available()
    )


def resolve_hf_hub_cache() -> str:
    """Resolve HuggingFace hub cache directory with fallback when huggingface_hub is not installed."""
    try:
        hub_constants = importlib.import_module("huggingface_hub.constants")
        return str(getattr(hub_constants, "HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub")))
    except (ImportError, AttributeError):
        return os.path.expanduser("~/.cache/huggingface/hub")
