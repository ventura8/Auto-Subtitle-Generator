"""Tests for the shared NVIDIA ASR helpers in ``modules.asr.common``."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from modules.asr import common
from modules.asr.common import DecodedClip, DeviceChoice

OOM = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")


def _fake_torch(cuda=True, bf16=True):
    """Torch stand-in whose CUDA availability and native bf16 support are fixed."""
    fake = MagicMock()
    fake.cuda.is_available.return_value = cuda
    fake.cuda.device_count.return_value = 1 if cuda else 0
    fake.cuda.is_bf16_supported.return_value = bf16
    return fake


class TestNamedTuples(unittest.TestCase):
    def test_decoded_clip_fields(self):
        self.assertEqual(DecodedClip._fields, ("text", "stamps", "logprob", "degenerate"))

    def test_device_choice_fields(self):
        self.assertEqual(DeviceChoice("cpu", None), ("cpu", None))


class TestSelectDevice(unittest.TestCase):
    def test_native_bf16_card_uses_bfloat16(self):
        fake = _fake_torch()
        self.assertEqual(common.select_device(fake), DeviceChoice("cuda:0", fake.bfloat16))
        fake.cuda.is_bf16_supported.assert_called_once_with(including_emulation=False)

    def test_card_without_native_bf16_uses_float32(self):
        fake = _fake_torch(bf16=False)
        self.assertEqual(common.select_device(fake), DeviceChoice("cuda:0", fake.float32))

    def test_hidden_gpu_uses_cpu_float32(self):
        fake = _fake_torch(cuda=False)
        self.assertEqual(common.select_device(fake), DeviceChoice("cpu", fake.float32))
        fake.cuda.is_bf16_supported.assert_not_called()

    def test_missing_torch_uses_cpu_without_dtype(self):
        self.assertEqual(common.select_device(None), DeviceChoice("cpu", None))


class TestLoadPretrained(unittest.TestCase):
    def setUp(self):
        self.loader = MagicMock(name="from_pretrained")
        recovery = patch("modules.asr.common.load_with_cache_recovery")
        self.recovery = recovery.start()
        self.addCleanup(recovery.stop)
        log_patch = patch("modules.asr.common.log")
        self.log = log_patch.start()
        self.addCleanup(log_patch.stop)

    def _load(self, fake_torch):
        with patch("modules.asr.common.torch", fake_torch):
            return common.load_pretrained(self.loader, "nvidia/canary-1b-v2", "abc123", "Canary")

    def test_loads_on_cuda_with_explicit_device_dtype_and_revision(self):
        fake = _fake_torch()
        model, choice = self._load(fake)
        self.assertIs(model, self.recovery.return_value)
        self.assertEqual(choice, DeviceChoice("cuda:0", fake.bfloat16))
        expected = {"revision": "abc123", "dtype": fake.bfloat16, "device_map": "cuda:0"}
        self.recovery.assert_called_once_with(self.loader, "nvidia/canary-1b-v2", expected, common._CACHE_LOGGER, "Canary")

    def test_cuda_oom_logs_warning_and_reloads_on_cpu(self):
        fake = _fake_torch()
        self.recovery.side_effect = [OOM, "cpu-model"]
        model, choice = self._load(fake)
        self.assertEqual((model, choice), ("cpu-model", DeviceChoice("cpu", fake.float32)))
        self.assertEqual(self.recovery.call_args.args[2]["device_map"], "cpu")
        self.assertEqual(self.log.call_args.args[1], "WARNING")
        fake.cuda.empty_cache.assert_called_once_with()

    def test_oom_while_already_on_cpu_reraises(self):
        self.recovery.side_effect = OOM
        with self.assertRaises(RuntimeError):
            self._load(_fake_torch(cuda=False))
        self.recovery.assert_called_once()

    def test_other_runtime_error_on_cuda_reraises(self):
        self.recovery.side_effect = RuntimeError("size mismatch for weight")
        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            self._load(_fake_torch())
        self.log.assert_not_called()

    def test_cache_corruption_warning_reaches_pipeline_log(self):
        common._CACHE_LOGGER.warning("%s cache appears corrupt (%s).", "Canary", "bad header")
        self.log.assert_called_once_with("Canary cache appears corrupt (bad header).", "WARNING")


class TestFlattenLstmWeights(unittest.TestCase):
    def test_flattens_only_modules_that_support_it(self):
        lstm = MagicMock()
        model = MagicMock()
        model.modules.return_value = [SimpleNamespace(), lstm, SimpleNamespace(flatten_parameters=None)]
        common.flatten_lstm_weights(model)
        lstm.flatten_parameters.assert_called_once_with()


class TestCudaCache(unittest.TestCase):
    def test_is_cuda_oom_matches_message_case_insensitively(self):
        self.assertTrue(common.is_cuda_oom(OOM))
        self.assertTrue(common.is_cuda_oom(RuntimeError("Out Of Memory")))
        self.assertFalse(common.is_cuda_oom(RuntimeError("device-side assert")))

    def test_release_without_torch_only_collects(self):
        with patch("modules.asr.common.torch", None), patch("modules.asr.common.gc.collect") as collect:
            common.release_cuda_cache()
        collect.assert_called_once_with()

    def test_release_skips_empty_cache_without_usable_cuda(self):
        fake = _fake_torch(cuda=False)
        with patch("modules.asr.common.torch", fake):
            common.release_cuda_cache()
        fake.cuda.empty_cache.assert_not_called()

    def test_release_empties_cache_on_cuda(self):
        fake = _fake_torch()
        with patch("modules.asr.common.torch", fake):
            common.release_cuda_cache()
        fake.cuda.empty_cache.assert_called_once_with()

    def test_release_logs_cleanup_failure(self):
        fake = _fake_torch()
        fake.cuda.empty_cache.side_effect = RuntimeError("driver gone")
        with patch("modules.asr.common.torch", fake), patch("modules.asr.common.log") as log:
            common.release_cuda_cache()
        self.assertIn("driver gone", log.call_args.args[0])
        self.assertEqual(log.call_args.args[1], "WARNING")


class TestOomBisection(unittest.TestCase):
    def setUp(self):
        release = patch("modules.asr.common.release_cuda_cache")
        self.release = release.start()
        self.addCleanup(release.stop)
        log_patch = patch("modules.asr.common.log")
        self.log = log_patch.start()
        self.addCleanup(log_patch.stop)

    def test_success_passes_through_as_list(self):
        generate = MagicMock(return_value=("a", "b"))
        self.assertEqual(common.generate_with_oom_bisection(generate, [1, 2]), ["a", "b"])
        self.release.assert_not_called()

    def test_oom_splits_in_half_and_keeps_order(self):
        def generate(clips):
            if len(clips) > 2:
                raise OOM
            return [f"t{clip}" for clip in clips]

        self.assertEqual(common.generate_with_oom_bisection(generate, [1, 2, 3, 4]), ["t1", "t2", "t3", "t4"])
        self.release.assert_called_once_with()
        self.assertEqual(self.log.call_args.args[1], "WARNING")

    def test_odd_batch_splits_into_smaller_then_larger_half(self):
        generate = MagicMock(side_effect=[OOM, ["a"], ["b", "c"]])
        self.assertEqual(common.generate_with_oom_bisection(generate, [1, 2, 3]), ["a", "b", "c"])
        self.assertEqual(generate.call_args_list, [call([1, 2, 3]), call([1]), call([2, 3])])

    def test_oom_on_single_clip_reraises(self):
        generate = MagicMock(side_effect=OOM)
        with self.assertRaises(RuntimeError):
            common.generate_with_oom_bisection(generate, [1])
        self.release.assert_not_called()

    def test_non_oom_error_is_not_retried(self):
        generate = MagicMock(side_effect=RuntimeError("index out of range"))
        with self.assertRaisesRegex(RuntimeError, "index out of range"):
            common.generate_with_oom_bisection(generate, [1, 2])
        generate.assert_called_once_with([1, 2])


class TestTokenBudgets(unittest.TestCase):
    def test_canary_budget_grows_with_duration(self):
        short = common.canary_token_budget(2.0, 9)
        self.assertEqual(short, 24 + common.TOKEN_MARGIN)
        self.assertLess(short, common.canary_token_budget(15.0, 9))

    def test_canary_budget_never_exceeds_positional_table(self):
        self.assertEqual(common.canary_token_budget(600.0, 9), 1024 - 9)
        self.assertEqual(common.canary_token_budget(600.0, 9, max_positions=512), 512 - 9)

    def test_canary_budget_is_at_least_one_token(self):
        self.assertEqual(common.canary_token_budget(5.0, 1024), 1)

    def test_parakeet_budget_allows_every_frame_its_symbols(self):
        self.assertEqual(common.parakeet_step_budget(250, 10), 2501)
        self.assertGreaterEqual(common.parakeet_step_budget(250, 10), 250 + 240)


class TestDegeneracy(unittest.TestCase):
    def test_zlib_ratio_of_empty_text_is_zero(self):
        self.assertEqual(common.zlib_ratio(""), 0.0)

    def test_zlib_ratio_separates_repetition_from_speech(self):
        self.assertGreater(common.zlib_ratio("ha " * 50), common.ZLIB_RATIO_LIMIT)
        self.assertLess(common.zlib_ratio("Bună ziua, ce mai faci?"), common.ZLIB_RATIO_LIMIT)

    def test_collapse_repeats_keeps_one_copy(self):
        self.assertEqual(common.collapse_repeats("la la la la la"), "la")
        self.assertEqual(common.collapse_repeats("I said no no no no way"), "I said no way")

    def test_collapse_repeats_leaves_short_runs_and_normal_text(self):
        self.assertEqual(common.collapse_repeats("no no no"), "no no no")
        self.assertEqual(common.collapse_repeats("Bună ziua, ce mai faci?"), "Bună ziua, ce mai faci?")

    def test_empty_or_blank_text_is_degenerate(self):
        self.assertTrue(common.is_degenerate("", 3.0))
        self.assertTrue(common.is_degenerate("  \n ", 3.0))

    def test_repetitive_text_is_degenerate(self):
        self.assertTrue(common.is_degenerate("ha " * 50, 60.0))

    def test_impossibly_fast_text_is_degenerate(self):
        self.assertTrue(common.is_degenerate("Bună ziua, ce mai faci?", 0.5))

    def test_normal_text_is_not_degenerate(self):
        self.assertFalse(common.is_degenerate("Bună ziua, ce mai faci?", 2.0))
        self.assertFalse(common.is_degenerate("Bună ziua, ce mai faci?", 0.0))
