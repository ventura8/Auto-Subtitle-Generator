"""Corrupt safetensors checkpoints must reach the purge-and-retry path."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from modules.runtime.model_cache import is_corrupt_model_error
from modules.translators import common


class FakeSafetensorError(Exception):
    """Stand-in for ``safetensors.SafetensorError``, whose only base is ``Exception``."""


class TestSafetensorsCorruptTokens(unittest.TestCase):
    def test_header_error_wording_is_recognised(self):
        self.assertTrue(is_corrupt_model_error(FakeSafetensorError("Error while deserializing header: invalid header length")))

    def test_incomplete_metadata_is_recognised(self):
        error = FakeSafetensorError("Error while deserializing header: incomplete metadata, file not fully covered")
        self.assertTrue(is_corrupt_model_error(error))
        self.assertTrue(is_corrupt_model_error(RuntimeError("incomplete metadata")))


class TestSafetensorErrorResolution(unittest.TestCase):
    def test_resolves_the_library_error_type(self):
        module = SimpleNamespace(SafetensorError=FakeSafetensorError)
        with patch("modules.translators.common.importlib.import_module", return_value=module):
            self.assertEqual(common._safetensor_errors(), [FakeSafetensorError])

    def test_missing_library_adds_nothing(self):
        with patch("modules.translators.common.importlib.import_module", side_effect=ImportError("safetensors")):
            self.assertEqual(common._safetensor_errors(), [])

    def test_library_without_the_error_type_adds_nothing(self):
        with patch("modules.translators.common.importlib.import_module", return_value=SimpleNamespace()):
            self.assertEqual(common._safetensor_errors(), [])

    def test_base_load_errors_are_kept(self):
        self.assertEqual(common._LOAD_ERRORS[:3], (RuntimeError, OSError, ValueError))


class TestLoadWithCacheRecoverySafetensors(unittest.TestCase):
    def setUp(self):
        errors = patch("modules.translators.common._LOAD_ERRORS", (RuntimeError, OSError, ValueError, FakeSafetensorError))
        errors.start()
        self.addCleanup(errors.stop)
        purge = patch("modules.translators.common.purge_hf_model_cache")
        self.purge = purge.start()
        self.addCleanup(purge.stop)

    def test_corrupt_header_purges_and_retries(self):
        loader = MagicMock(side_effect=[FakeSafetensorError("Error while deserializing header: invalid header length"), "model"])
        logger = MagicMock()
        result = common.load_with_cache_recovery(loader, "nvidia/parakeet-tdt-0.6b-v3", {"revision": "abc"}, logger, "Parakeet")
        self.assertEqual(result, "model")
        self.purge.assert_called_once_with("nvidia/parakeet-tdt-0.6b-v3")
        loader.assert_called_with("nvidia/parakeet-tdt-0.6b-v3", revision="abc")
        logger.warning.assert_called_once()

    def test_other_safetensors_error_is_reraised(self):
        loader = MagicMock(side_effect=FakeSafetensorError("device mismatch"))
        with self.assertRaisesRegex(FakeSafetensorError, "device mismatch"):
            common.load_with_cache_recovery(loader, "nvidia/canary-1b-v2")
        self.purge.assert_not_called()
