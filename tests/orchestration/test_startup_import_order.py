"""Regression tests: CUDA runtime paths are prepared before Torch is imported.

Torch is loaded lazily through the single chokepoint
``modules.runtime.optional_imports.load_optional_torch``. That helper must run
``nvidia_paths.prepare_nvidia_paths()`` before importing Torch so the bundled
NVIDIA runtime directories are on the library search path when Torch
initializes. These tests pin both the ordering and the chokepoint itself.
"""

import pathlib
import re
import unittest
from unittest.mock import MagicMock, patch

from modules.runtime import optional_imports

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DIRECT_TORCH_IMPORT = re.compile(
    r"^\s*(?:import\s+torch\b"
    r"|from\s+torch(?:\.\w+)*\s+import\b"
    r"|(?:importlib\.)?import_module\(\s*['\"]torch(?:\.\w+)*['\"]\s*\)"
    r"|__import__\(\s*['\"]torch(?:\.\w+)*['\"]\s*\))",
    re.MULTILINE,
)


class TestStartupImportOrder(unittest.TestCase):
    """Verify Torch initialization always follows NVIDIA path preparation."""

    def setUp(self):
        original_state = optional_imports._PREPARE_STATE["nvidia_paths_prepared"]
        optional_imports._PREPARE_STATE["nvidia_paths_prepared"] = False
        self.addCleanup(optional_imports._PREPARE_STATE.__setitem__, "nvidia_paths_prepared", original_state)

    def test_prepare_nvidia_paths_runs_before_torch_import(self):
        calls = []
        prepare = MagicMock(side_effect=lambda: calls.append("prepare"))
        import_module = MagicMock(side_effect=lambda name: calls.append(name))

        with (
            patch.object(optional_imports.nvidia_paths, "prepare_nvidia_paths", prepare),
            patch.object(optional_imports.importlib, "import_module", import_module),
        ):
            optional_imports.load_optional_torch()

        self.assertEqual(calls, ["prepare", "torch"])

    def test_prepare_nvidia_paths_runs_before_torch_import_failure(self):
        calls = []
        prepare = MagicMock(side_effect=lambda: calls.append("prepare"))

        def _failing_import(name):
            calls.append(name)
            raise ImportError(name=name)

        with (
            patch.object(optional_imports.nvidia_paths, "prepare_nvidia_paths", prepare),
            patch.object(optional_imports.importlib, "import_module", _failing_import),
        ):
            self.assertIsNone(optional_imports.load_optional_torch())

        self.assertEqual(calls, ["prepare", "torch"])

    def test_prepare_nvidia_paths_is_only_run_once(self):
        prepare = MagicMock()
        with (
            patch.object(optional_imports.nvidia_paths, "prepare_nvidia_paths", prepare),
            patch.object(optional_imports.importlib, "import_module", MagicMock()),
        ):
            optional_imports.load_optional_torch()
            optional_imports.load_optional_torch()

        prepare.assert_called_once()

    def test_prepare_nvidia_paths_retries_after_preparation_failure(self):
        prepare = MagicMock(side_effect=[RuntimeError("CUDA paths unavailable"), None])
        with patch.object(optional_imports.nvidia_paths, "prepare_nvidia_paths", prepare):
            with self.assertRaisesRegex(RuntimeError, "CUDA paths unavailable"):
                optional_imports._prepare_nvidia_paths_once()
            self.assertFalse(optional_imports._PREPARE_STATE["nvidia_paths_prepared"])
            optional_imports._prepare_nvidia_paths_once()

        self.assertTrue(optional_imports._PREPARE_STATE["nvidia_paths_prepared"])

    def test_no_module_imports_torch_outside_the_chokepoint(self):
        import ast

        def has_direct_torch_import(source_path: pathlib.Path) -> bool:
            source_text = source_path.read_text(encoding="utf-8")
            try:
                tree = ast.parse(source_text, filename=str(source_path))
            except SyntaxError:
                return False
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "torch" or alias.name.startswith("torch."):
                            return True
                elif isinstance(node, ast.ImportFrom):
                    if node.module == "torch" or (node.module and node.module.startswith("torch.")):
                        return True
            return bool(_DIRECT_TORCH_IMPORT.search(source_text))

        offenders = [
            path.relative_to(_REPO_ROOT).as_posix()
            for path in sorted((_REPO_ROOT / "modules").rglob("*.py"))
            if path.name != "optional_imports.py" and has_direct_torch_import(path)
        ]
        self.assertEqual(
            offenders,
            [],
            "Torch must be loaded via optional_imports.load_optional_torch() so NVIDIA "
            f"runtime paths are prepared first. Direct imports found in: {offenders}",
        )


if __name__ == "__main__":
    unittest.main()
