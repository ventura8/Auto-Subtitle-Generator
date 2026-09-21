"""Tests for CUDA usability probing and the CPU-only bootstrap."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from modules.runtime import bootstrap, optional_imports


def _torch_stub(available, device_count):
    cuda = SimpleNamespace(is_available=lambda: available, device_count=lambda: device_count)
    return SimpleNamespace(cuda=cuda)


class TestIsCudaUsable(unittest.TestCase):
    def test_requires_available_and_at_least_one_device(self):
        self.assertTrue(optional_imports.is_cuda_usable(_torch_stub(True, 1)))
        # torch 2.14/cu132 reports available with the GPU hidden; the count is what matters.
        self.assertFalse(optional_imports.is_cuda_usable(_torch_stub(True, 0)))
        self.assertFalse(optional_imports.is_cuda_usable(_torch_stub(False, 1)))

    def test_missing_torch_or_cuda_namespace_is_not_usable(self):
        self.assertFalse(optional_imports.is_cuda_usable(None))
        self.assertFalse(optional_imports.is_cuda_usable(SimpleNamespace()))

    def test_probe_errors_are_not_usable(self):
        def _boom():
            raise RuntimeError("driver missing")

        broken = SimpleNamespace(cuda=SimpleNamespace(is_available=_boom, device_count=lambda: 1))
        self.assertFalse(optional_imports.is_cuda_usable(broken))


class TestForceCpuOnlyEnv(unittest.TestCase):
    def test_hides_every_device_with_negative_id(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=False):
            bootstrap.force_cpu_only_env()
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], bootstrap.CPU_ONLY_CUDA_DEVICES)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "-1")


if __name__ == "__main__":
    unittest.main()
