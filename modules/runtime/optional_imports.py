"""Helpers for optional third-party imports used across modules."""

import importlib
from typing import Any


def load_optional_torch() -> Any | None:
    """Return torch when installed, else None."""
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        if getattr(exc, "name", None) not in {"torch", "torch.__init__"}:
            raise
        return None
