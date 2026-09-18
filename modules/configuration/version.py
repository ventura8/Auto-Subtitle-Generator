"""Centralized application version resolution.

Provides a single source of truth for the project version, reading from
pyproject.toml at repository root or falling back to importlib.metadata.
"""

from __future__ import annotations

import importlib.metadata
import tomllib
from pathlib import Path

_FALLBACK_VERSION = "1.2.3"
_PACKAGE_NAME = "auto-subtitle-generator"


def _parse_version_from_file(pyproject_path: Path) -> str | None:
    """Parse version string from a specific pyproject.toml path."""
    try:
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
        project = data.get("project", {})
        version = project.get("version")
        if isinstance(version, str) and version.strip():
            return version.strip()
    except (OSError, tomllib.TOMLDecodeError):
        return None
    return None


def _read_version_from_pyproject() -> str | None:
    """Read version directly from pyproject.toml if present."""
    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        pyproject_path = candidate / "pyproject.toml"
        if pyproject_path.is_file():
            version = _parse_version_from_file(pyproject_path)
            if version:
                return version
    return None


def _read_version_from_metadata() -> str | None:
    """Read version from package metadata if installed."""
    try:
        ver = importlib.metadata.version(_PACKAGE_NAME)
        if isinstance(ver, str) and ver.strip():
            return ver.strip()
    except (importlib.metadata.PackageNotFoundError, OSError):
        return None
    return None


def get_app_version() -> str:
    """Return the application version from the canonical single source of truth."""
    version = _read_version_from_pyproject()
    if version:
        return version

    version = _read_version_from_metadata()
    if version:
        return version

    return _FALLBACK_VERSION


__version__ = get_app_version()
