"""Tests for CUDA runtime path discovery."""

import os
import unittest
from unittest.mock import MagicMock, patch

from modules.runtime import nvidia_paths


class TestNvidiaPaths(unittest.TestCase):
    """Verify startup CUDA path discovery without loading real dependencies."""

    def test_collect_site_packages_adds_existing_manual_location(self):
        with (
            patch("site.getsitepackages", return_value=["default-site"]),
            patch("os.path.isdir", return_value=True),
            patch("sys.prefix", "/venv"),
        ):
            paths = nvidia_paths._collect_site_packages_paths()
        self.assertIn("default-site", paths)
        self.assertIn(os.path.join("/venv", "Lib", "site-packages"), paths)

    def test_collect_torch_lib_paths_ignores_missing_or_invalid_modules(self):
        self.assertEqual(nvidia_paths._collect_torch_lib_paths(None), [])
        torch_module = MagicMock()
        torch_module.__path__ = ["/torch"]
        with patch("os.path.exists", return_value=False):
            self.assertEqual(nvidia_paths._collect_torch_lib_paths(torch_module), [])

    def test_load_optional_torch_uses_startup_safe_loader(self):
        loader_module = MagicMock()
        loader_module.load_optional_torch.return_value = "torch-module"

        with patch.object(nvidia_paths.importlib, "import_module", return_value=loader_module) as import_module:
            result = nvidia_paths._load_optional_torch()

        self.assertEqual(result, "torch-module")
        import_module.assert_called_once_with("modules.runtime.optional_imports")
        loader_module.load_optional_torch.assert_called_once_with()

    def test_load_nvidia_paths_imports_torch_when_not_supplied(self):
        torch_module = MagicMock()
        torch_module.__path__ = []
        with (
            patch("modules.runtime.nvidia_paths._apply_paths_to_env") as apply_paths,
            patch("modules.runtime.nvidia_paths._collect_torch_lib_paths", return_value=["/torch/lib"]) as collect_torch_lib,
            patch("modules.runtime.nvidia_paths._load_optional_torch", return_value=torch_module) as load_torch,
            patch.object(nvidia_paths.importlib, "import_module", side_effect=ImportError("no ort")) as mock_import,
        ):
            nvidia_paths.load_nvidia_paths()
        load_torch.assert_called_once_with()
        mock_import.assert_any_call("onnxruntime")
        collect_torch_lib.assert_called_once_with(torch_module)
        apply_paths.assert_called_once_with(["/torch/lib"])

    def test_load_nvidia_paths_handles_missing_torch(self):
        with (
            patch("modules.runtime.nvidia_paths._apply_paths_to_env") as apply_paths,
            patch("modules.runtime.nvidia_paths._collect_torch_lib_paths", return_value=[]) as collect_torch_lib,
            patch("modules.runtime.nvidia_paths._load_optional_torch", return_value=None) as load_torch,
            patch.object(nvidia_paths.importlib, "import_module", side_effect=ImportError("no dependency")),
        ):
            nvidia_paths.load_nvidia_paths()
        load_torch.assert_called_once_with()
        collect_torch_lib.assert_called_once_with(None)
        apply_paths.assert_called_once_with([])

    def test_preload_shared_library_handles_valid_and_invalid(self):
        with patch("modules.runtime.nvidia_paths.ctypes.CDLL") as mock_cdll:
            nvidia_paths._preload_shared_library("/path/to/libcuda.so.1")
            mock_cdll.assert_called_once()

        with patch("modules.runtime.nvidia_paths.ctypes.CDLL") as mock_cdll:
            nvidia_paths._preload_shared_library("/path/to/file.txt")
            mock_cdll.assert_not_called()

        with patch("modules.runtime.nvidia_paths.ctypes.CDLL", side_effect=OSError("load failed")):
            nvidia_paths._preload_shared_library("/path/to/libfailed.so")

    def test_preload_runtime_libraries_only_preloads_allowlisted_files(self):
        with (
            patch("os.path.isdir", return_value=True),
            patch("os.listdir", return_value=["libcudnn.so", "libnvblas.so", "readme.txt"]),
            patch("modules.runtime.nvidia_paths._preload_shared_library") as mock_preload,
        ):
            nvidia_paths._preload_runtime_libraries("/test/dir")
        mock_preload.assert_called_once_with(os.path.join("/test/dir", "libcudnn.so"))

    def test_is_cuda_explicitly_disabled_detects_cpu_only_opt_out(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}, clear=False):
            self.assertTrue(nvidia_paths.is_cuda_explicitly_disabled())
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "   "}, clear=False):
            self.assertTrue(nvidia_paths.is_cuda_explicitly_disabled())
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=False):
            self.assertFalse(nvidia_paths.is_cuda_explicitly_disabled())
        env_without_key = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
        with patch.dict(os.environ, env_without_key, clear=True):
            self.assertFalse(nvidia_paths.is_cuda_explicitly_disabled())

    def test_prepare_nvidia_paths_skipped_when_cuda_disabled(self):
        # Injecting the bundled cuBLAS dir on a CPU-only run makes libnvblas.so
        # hijack CPU BLAS and abort when no GPU is present.
        with (
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}, clear=False),
            patch("modules.runtime.nvidia_paths._apply_paths_to_env") as mock_apply,
        ):
            nvidia_paths.prepare_nvidia_paths()
            mock_apply.assert_not_called()

    def test_prepare_nvidia_paths_applied_when_cuda_enabled(self):
        with (
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=False),
            patch("modules.runtime.nvidia_paths._collect_nvidia_runtime_paths", return_value=["/nv/lib"]),
            patch("modules.runtime.nvidia_paths._apply_paths_to_env") as mock_apply,
        ):
            nvidia_paths.prepare_nvidia_paths()
            mock_apply.assert_called_once_with(["/nv/lib"])

    def test_apply_paths_to_env_deduplicates_and_updates_linker_vars(self):
        env = {"PATH": os.path.join("/existing", "lib"), "LD_LIBRARY_PATH": "", "DYLD_LIBRARY_PATH": ""}
        new_path = os.path.join("/nv", "lib")
        existing_path = os.path.join("/existing", "lib")
        with (
            patch.dict(os.environ, env, clear=True),
            patch("modules.runtime.nvidia_paths._add_dll_directory_if_supported"),
            patch("modules.runtime.nvidia_paths._preload_runtime_libraries"),
        ):
            nvidia_paths._apply_paths_to_env([new_path, new_path, existing_path])
            path_entries = os.environ["PATH"].split(os.pathsep)
            ld_entries = os.environ["LD_LIBRARY_PATH"].split(os.pathsep)
            dyld_entries = os.environ["DYLD_LIBRARY_PATH"].split(os.pathsep)

        self.assertEqual(path_entries, [new_path, existing_path])
        self.assertEqual(ld_entries.count(new_path), 1)
        self.assertEqual(dyld_entries.count(new_path), 1)
        self.assertEqual(ld_entries.count(existing_path), 1)
