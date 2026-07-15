"""Shared helpers for translation backend implementations."""

import importlib
from typing import Any

from modules.runtime.optional_imports import load_optional_torch

torch: Any | None = load_optional_torch()


def import_transformers_module():
    """Import transformers lazily to preserve optional dependency behavior."""
    return importlib.import_module("transformers")


def resolve_device_map():
    """Resolve explicit device mapping for transformer loading."""
    if torch is not None and torch.cuda.is_available():
        return "cuda:0"
    return None
