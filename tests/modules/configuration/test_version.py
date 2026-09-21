"""Unit tests for centralized version resolution."""

from __future__ import annotations

import importlib.metadata
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules.configuration import version


class TestVersionResolution(unittest.TestCase):
    """Test get_app_version behavior across various scenarios."""

    def test_read_version_from_pyproject_success(self):
        """Verify successful parsing of valid pyproject.toml."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pyproject_file = Path(temp_dir) / "pyproject.toml"
            pyproject_file.write_text(
                '[project]\nname = "test-pkg"\nversion = "2.3.4"\n',
                encoding="utf-8",
            )
            with patch.object(version, "Path") as mock_path:
                mock_path.return_value.resolve.return_value.parent = Path(temp_dir)
                v = version._read_version_from_pyproject()
                self.assertEqual(v, "2.3.4")

    def test_read_version_from_pyproject_corrupt(self):
        """Verify corrupt pyproject.toml returns None gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pyproject_file = Path(temp_dir) / "pyproject.toml"
            pyproject_file.write_text("invalid [[ toml syntax", encoding="utf-8")
            with patch.object(version, "Path") as mock_path:
                mock_path.return_value.resolve.return_value.parent = Path(temp_dir)
                v = version._read_version_from_pyproject()
                self.assertIsNone(v)

    def test_read_version_from_metadata_success(self):
        """Verify reading version from importlib metadata."""
        with patch("importlib.metadata.version", return_value="3.4.5"):
            v = version._read_version_from_metadata()
            self.assertEqual(v, "3.4.5")

    def test_read_version_from_pyproject_empty_or_non_string(self):
        """Verify empty or non-string version in pyproject returns None."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pyproject_file = Path(temp_dir) / "pyproject.toml"
            pyproject_file.write_text('[project]\nversion = ""\n', encoding="utf-8")
            with patch.object(version, "Path") as mock_path:
                mock_path.return_value.resolve.return_value.parent = Path(temp_dir)
                self.assertIsNone(version._read_version_from_pyproject())

            pyproject_file.write_text("[project]\nversion = 123\n", encoding="utf-8")
            with patch.object(version, "Path") as mock_path:
                mock_path.return_value.resolve.return_value.parent = Path(temp_dir)
                self.assertIsNone(version._read_version_from_pyproject())

    def test_read_version_from_metadata_empty(self):
        """Verify empty or non-string metadata version returns None."""
        with patch("importlib.metadata.version", return_value=""):
            self.assertIsNone(version._read_version_from_metadata())
        with patch("importlib.metadata.version", return_value=123):
            self.assertIsNone(version._read_version_from_metadata())

    def test_read_version_from_metadata_not_found(self):
        """Verify PackageNotFoundError returns None."""
        with patch(
            "importlib.metadata.version",
            side_effect=importlib.metadata.PackageNotFoundError,
        ):
            v = version._read_version_from_metadata()
            self.assertIsNone(v)

    def test_get_app_version_pyproject_primary(self):
        """Verify pyproject is preferred when available."""
        with patch.object(version, "_read_version_from_pyproject", return_value="1.2.4"):
            self.assertEqual(version.get_app_version(), "1.2.4")

    def test_get_app_version_metadata_fallback(self):
        """Verify metadata fallback when pyproject returns None."""
        with (
            patch.object(version, "_read_version_from_pyproject", return_value=None),
            patch.object(version, "_read_version_from_metadata", return_value="9.8.7"),
        ):
            self.assertEqual(version.get_app_version(), "9.8.7")

    def test_get_app_version_default_fallback(self):
        """Verify default fallback when neither pyproject nor metadata is available."""
        with (
            patch.object(version, "_read_version_from_pyproject", return_value=None),
            patch.object(version, "_read_version_from_metadata", return_value=None),
        ):
            self.assertEqual(version.get_app_version(), "1.2.4")

    def test_module_dunder_version_matches_pyproject(self):
        """Verify __version__ is loaded and matches the pyproject.toml setting."""
        self.assertIsInstance(version.__version__, str)
        self.assertEqual(version.__version__, "1.2.4")
