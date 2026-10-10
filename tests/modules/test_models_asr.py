"""Tests for the NVIDIA ASR lifecycle, VRAM sizing and Whisper language-ID delegate in modules.models."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from modules import models
from modules.configuration import asr_settings

GB = 1024**3


def _cuda_torch(free_gb=None, allocated=0):
    """A torch stand-in with one usable CUDA device and the given free / allocated memory."""
    fake = MagicMock()
    fake.cuda.is_available.return_value = True
    fake.cuda.device_count.return_value = 1
    fake.cuda.mem_get_info.return_value = (int((free_gb or 0) * GB), 8 * GB)
    if free_gb is None:
        fake.cuda.mem_get_info.side_effect = RuntimeError("no context")
    fake.cuda.memory_allocated.return_value = allocated
    return fake


def _no_cuda_torch():
    """A torch stand-in without a usable CUDA device."""
    fake = MagicMock()
    fake.cuda.is_available.return_value = False
    return fake


class TestWhisperDetectLanguage(unittest.TestCase):
    def _wrapper(self, gpu_model, cpu_model):
        """Build a GPU-loaded WhisperModel whose CPU rebuild returns ``cpu_model``."""
        fake_module = MagicMock()
        fake_module.WhisperModel.side_effect = [gpu_model, cpu_model]
        with patch("modules.models._import_module", return_value=fake_module), patch("modules.models.torch", _cuda_torch(4.0)):
            return models.WhisperModel(), fake_module

    def test_detect_language_delegates(self):
        gpu_model = MagicMock()
        gpu_model.detect_language.return_value = ("ro", 0.9, [("ro", 0.9)])
        wrapper, _module = self._wrapper(gpu_model, MagicMock())
        self.assertEqual(wrapper.detect_language(audio="pcm"), ("ro", 0.9, [("ro", 0.9)]))
        gpu_model.detect_language.assert_called_once_with(audio="pcm")

    def test_detect_language_retries_on_cpu_when_cuda_runtime_is_missing(self):
        gpu_model = MagicMock()
        gpu_model.detect_language.side_effect = RuntimeError("libcublas.so.12: cannot open shared object file")
        cpu_model = MagicMock()
        cpu_model.detect_language.return_value = ("en", 0.8, [])
        wrapper, fake_module = self._wrapper(gpu_model, cpu_model)
        with patch("modules.models.log") as mock_log:
            self.assertEqual(wrapper.detect_language(audio="pcm"), ("en", 0.8, []))
        self.assertTrue(wrapper._using_cpu)
        self.assertEqual(fake_module.WhisperModel.call_args.kwargs, {"device": "cpu", "compute_type": "int8"})
        self.assertEqual(mock_log.call_args.args[1], "WARNING")
        self.assertIn("detect_language", mock_log.call_args.args[0])

    def test_other_errors_are_not_retried(self):
        gpu_model = MagicMock()
        gpu_model.detect_language.side_effect = ValueError("max() arg is an empty sequence")
        wrapper, _module = self._wrapper(gpu_model, MagicMock())
        with self.assertRaises(ValueError):
            wrapper.detect_language(audio="pcm")
        self.assertFalse(wrapper._using_cpu)


class TestAsrSlot(unittest.TestCase):
    def setUp(self):
        self.addCleanup(asr_settings.reset)
        self.backends = {"canary": MagicMock(name="CanaryModel"), "parakeet": MagicMock(name="ParakeetModel")}
        fake_module = SimpleNamespace(CanaryModel=self.backends["canary"], ParakeetModel=self.backends["parakeet"])
        for target, value in (("modules.models._import_module", fake_module), ("modules.models.log_vram", None)):
            patcher = patch(target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        cleanup = patch("modules.models._cleanup_torch_cache")
        self.cleanup = cleanup.start()
        self.addCleanup(cleanup.stop)

    def test_get_asr_builds_the_configured_checkpoint_once(self):
        manager = models.ModelManager()
        first = manager.get_asr("canary")
        self.assertIs(manager.get_asr("canary"), first)
        self.backends["canary"].assert_called_once_with("nvidia/canary-1b-v2", asr_settings.MODEL_REVISIONS["nvidia/canary-1b-v2"])

    def test_switching_engine_releases_the_previous_model(self):
        manager = models.ModelManager()
        canary = manager.get_asr("canary")
        with patch("modules.models._report_asr_release") as report:
            parakeet = manager.get_asr("parakeet")
        canary.release.assert_called_once_with()
        report.assert_called_once_with()
        self.assertIs(parakeet, self.backends["parakeet"].return_value)

    def test_offload_asr_without_a_model_only_cleans_up(self):
        manager = models.ModelManager()
        with patch("modules.models._report_asr_release") as report:
            manager.offload_asr()
        report.assert_not_called()
        self.cleanup.assert_called_once_with()

    def test_offload_asr_releases_and_reports(self):
        manager = models.ModelManager()
        model = manager.get_asr("parakeet")
        with patch("modules.models._report_asr_release") as report:
            manager.offload_asr()
        model.release.assert_called_once_with()
        report.assert_called_once_with()
        self.assertIsNone(manager._asr)
        self.assertIsNone(manager._asr_engine)

    def test_unknown_engine_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "Unknown NVIDIA ASR engine 'whisper'"):
            models.ModelManager().get_asr("whisper")


class TestVramReporting(unittest.TestCase):
    def test_release_report_warns_about_retained_memory(self):
        with patch("modules.models.torch", _cuda_torch(5.0, allocated=200 * 1024**2)), patch("modules.models.log") as mock_log:
            models._report_asr_release()
        mock_log.assert_any_call("  [ASR] 200 MB of VRAM still allocated after releasing the ASR model.", "WARNING")
        mock_log.assert_any_call("  [VRAM] after ASR offload: 5.0 GB free", "DEBUG")

    def test_release_report_is_quiet_when_memory_was_returned(self):
        with patch("modules.models.torch", _cuda_torch(5.0, allocated=1024)), patch("modules.models.log") as mock_log:
            models._report_asr_release()
        self.assertEqual([entry.args[1] for entry in mock_log.call_args_list], ["DEBUG"])

    def test_allocated_bytes_without_cuda_or_on_error(self):
        with patch("modules.models.torch", _no_cuda_torch()):
            self.assertEqual(models._cuda_allocated_bytes(), 0)
        broken = _cuda_torch(1.0)
        broken.cuda.memory_allocated.side_effect = RuntimeError("driver")
        with patch("modules.models.torch", broken):
            self.assertEqual(models._cuda_allocated_bytes(), 0)

    def test_log_vram_is_silent_without_a_reading(self):
        for fake in (_no_cuda_torch(), _cuda_torch(None)):
            with self.subTest(fake=fake), patch("modules.models.torch", fake), patch("modules.models.log") as mock_log:
                models.log_vram("before language ID")
                mock_log.assert_not_called()


class TestDynamicAsrBatch(unittest.TestCase):
    def setUp(self):
        saved = dict(models.OPTIMIZER.config)
        saved_profile = models.OPTIMIZER.profile
        self.addCleanup(models.OPTIMIZER.config.update, saved)
        self.addCleanup(setattr, models.OPTIMIZER, "profile", saved_profile)
        self.addCleanup(models.OPTIMIZER.config.pop, "asr_batch", None)
        models.OPTIMIZER.profile = "MID"
        models.OPTIMIZER.config["max_vram_usage_gb"] = 0
        self.model = SimpleNamespace(device="cuda:0", model_id="nvidia/canary-1b-v2")

    def test_pinned_batch_wins(self):
        models.OPTIMIZER.config["asr_batch"] = 3
        self.assertEqual(models.apply_dynamic_asr_batch(self.model, 15.0), 3)

    def test_cpu_model_gets_the_cpu_cap(self):
        cpu_model = SimpleNamespace(device="cpu", model_id="nvidia/canary-1b-v2")
        self.assertEqual(models.apply_dynamic_asr_batch(cpu_model, 15.0), 4)

    def test_unreadable_vram_keeps_the_profile_cap(self):
        with patch("modules.models.torch", _cuda_torch(None)):
            self.assertEqual(models.apply_dynamic_asr_batch(self.model, 15.0), 8)

    def test_batch_is_sized_from_free_vram_and_logged(self):
        with patch("modules.models.torch", _cuda_torch(1.0)), patch("modules.models.log") as mock_log:
            batch = models.apply_dynamic_asr_batch(self.model, 15.0)
        self.assertEqual(batch, 5)
        self.assertIn("-> batch 5 (profile cap 8)", mock_log.call_args.args[0])
        self.assertEqual(mock_log.call_args.args[1], "DEBUG")

    def test_vram_cap_limits_the_free_memory_planned_against(self):
        models.OPTIMIZER.config["max_vram_usage_gb"] = 2.5
        with patch.object(models.OPTIMIZER, "vram_gb", 8), patch("modules.models.torch", _cuda_torch(6.0)), patch("modules.models.log"):
            self.assertEqual(models.apply_dynamic_asr_batch(self.model, 15.0), 3)


class TestAsrGpuShortfall(unittest.TestCase):
    def setUp(self):
        self.addCleanup(asr_settings.reset)

    def test_no_gpu_or_no_reading_is_not_a_shortfall(self):
        for fake in (_no_cuda_torch(), _cuda_torch(None)):
            with self.subTest(fake=fake), patch("modules.models.torch", fake):
                self.assertIsNone(models.asr_gpu_shortfall("canary"))

    def test_enough_free_vram(self):
        with patch("modules.models.torch", _cuda_torch(3.0)), patch.object(models.OPTIMIZER, "vram_gb", 8):
            self.assertIsNone(models.asr_gpu_shortfall("canary"))

    def test_too_little_free_vram_is_explained(self):
        with patch("modules.models.torch", _cuda_torch(2.5)), patch.object(models.OPTIMIZER, "vram_gb", 8):
            self.assertEqual(models.asr_gpu_shortfall("canary"), "only 2.5 GB of VRAM free for nvidia/canary-1b-v2")


if __name__ == "__main__":
    unittest.main()
