"""Shared helpers for translation backend implementations."""

import importlib
from typing import Any

from modules.runtime.model_cache import is_corrupt_model_error, purge_hf_model_cache
from modules.runtime.optional_imports import is_mps_available, load_optional_torch

__all__ = [
    "add_device_load_kwargs",
    "import_transformers_module",
    "is_corrupt_model_error",
    "load_with_cache_recovery",
    "purge_hf_model_cache",
    "resolve_device_map",
]

torch: Any | None = load_optional_torch()


def _log_cache_corruption(logger, label, error):
    """Log cache corruption warning when logger is available."""
    if logger is not None:
        logger.warning("%s cache appears corrupt (%s). Purging cache and retrying download...", label, error)


def _handle_corrupt_cache_recovery(loader_callable, model_id, load_kwargs):
    """Purge model cache and retry loading."""
    purge_hf_model_cache(model_id)
    return loader_callable(model_id, **load_kwargs)


def _handle_offline_fallback(loader_callable, model_id, load_kwargs):
    """Retry model/tokenizer load in offline mode using cached local files."""
    fallback_kwargs = dict(load_kwargs)
    fallback_kwargs["local_files_only"] = True
    return loader_callable(model_id, **fallback_kwargs)


def _handle_load_error(load_target, load_kwargs, error, log_context):
    """Handle corrupt cache or offline fallback error for model loading."""
    loader_callable, model_id = load_target
    logger, label = log_context
    if is_corrupt_model_error(error):
        _log_cache_corruption(logger, label, error)
        return _handle_corrupt_cache_recovery(loader_callable, model_id, load_kwargs)
    if isinstance(error, OSError):
        return _handle_offline_fallback(loader_callable, model_id, load_kwargs)
    raise error


def load_with_cache_recovery(loader_callable, model_id, model_kwargs=None, logger=None, model_label=None):
    """Load model/tokenizer with corrupt cache auto-purge/retry and local_files_only fallback."""
    load_kwargs = dict(model_kwargs) if model_kwargs else {}
    label = model_label or model_id
    try:
        return loader_callable(model_id, **load_kwargs)
    except (RuntimeError, OSError, ValueError) as error:
        return _handle_load_error((loader_callable, model_id), load_kwargs, error, (logger, label))


def _ensure_torchaudio_safe():
    """Probe torchaudio without poisoning later imports when it is unavailable."""
    try:
        importlib.import_module("torchaudio")
    except (ImportError, RuntimeError, OSError, TypeError, ValueError):
        pass


def import_transformers_module():
    """Import transformers lazily to preserve optional dependency behavior."""
    _ensure_torchaudio_safe()
    return importlib.import_module("transformers")


def resolve_device_map():
    """Resolve explicit device mapping for transformer loading."""
    if torch is not None and torch.cuda.is_available():
        return "cuda:0"
    if is_mps_available(torch):
        return "mps"
    return None


def add_device_load_kwargs(kwargs, device_map, torch_module):
    """Add explicit device mapping and CUDA float16 precision when CUDA is selected."""
    if device_map is not None:
        kwargs["device_map"] = device_map
        if device_map.startswith("cuda") and torch_module is not None and hasattr(torch_module, "float16"):
            kwargs["dtype"] = torch_module.float16
    return kwargs
