"""VRAM-aware model selection and batch sizing, plus its wiring into the optimizer and worker."""

import contextlib
import logging
import unittest
from unittest.mock import MagicMock, patch

from modules import models
from modules.configuration import config
from modules.runtime import vram_tuning as vt

D600 = "facebook/nllb-200-distilled-600M"
D13 = "facebook/nllb-200-distilled-1.3B"
B33 = "facebook/nllb-200-3.3B"


class TestSelectNllbModel(unittest.TestCase):
    def test_auto_picks_largest_model_that_fits_the_weight_budget(self):
        # Thresholds follow measured fp16 weights (6.3 / 2.6 / 1.2 GB) at a 60 % budget.
        expectations = {32: B33, 24: B33, 16: B33, 12: B33, 11: B33, 10: D13, 8: D13, 7: D13, 5: D13, 4: D600, 3: D600}
        for vram, expected in expectations.items():
            with self.subTest(vram=vram):
                self.assertEqual(vt.select_nllb_model("auto", vram)[0], expected)

    def test_rtx3080_laptop_lands_on_distilled_1_3b(self):
        # The measured failure: 3.3B (6.3 GB) on a card reporting 7 GB thrashed at 0 % utilisation.
        model_id, note = vt.select_nllb_model("auto", 7)
        self.assertEqual(model_id, D13)
        self.assertIn("fit the 4.2 GB budget", note)

    def test_auto_without_gpu_memory_uses_smallest(self):
        model_id, note = vt.select_nllb_model("auto", 0)
        self.assertEqual(model_id, D600)
        self.assertIn("no GPU memory", note)

    def test_auto_below_every_budget_uses_smallest(self):
        model_id, note = vt.select_nllb_model("auto", 1)
        self.assertEqual(model_id, D600)
        self.assertIn("below every budget", note)

    def test_explicit_model_is_honoured_and_warned_when_too_big(self):
        model_id, note = vt.select_nllb_model(B33, 8)
        self.assertEqual(model_id, B33)
        self.assertIn("fallback to the CPU", note)

    def test_explicit_model_that_fits_has_no_warning(self):
        self.assertEqual(vt.select_nllb_model(B33, 24), (B33, None))
        self.assertEqual(vt.select_nllb_model(D13, 8), (D13, None))

    def test_unknown_explicit_model_is_passed_through_silently(self):
        self.assertEqual(vt.select_nllb_model("org/custom-nllb", 4), ("org/custom-nllb", None))

    def test_explicit_model_without_gpu_has_no_warning(self):
        self.assertEqual(vt.select_nllb_model(B33, 0), (B33, None))


class TestDynamicBatchSize(unittest.TestCase):
    def test_sizes_from_free_memory_and_per_item_cost(self):
        # 3.3B at 0.16 GB/item: 4 GB free * 0.8 = 3.2 GB -> 20 items, capped at 16.
        self.assertEqual(vt.dynamic_batch_size(4.0, B33, 10, cap=16), 16)
        self.assertEqual(vt.dynamic_batch_size(4.0, B33, 10, cap=32), 20)

    def test_fewer_beams_allow_larger_batches(self):
        self.assertGreater(vt.dynamic_batch_size(2.0, B33, 5, cap=64), vt.dynamic_batch_size(2.0, B33, 10, cap=64))

    def test_never_below_floor_or_above_cap(self):
        self.assertEqual(vt.dynamic_batch_size(0.0, B33, 10, cap=8), 1)
        self.assertEqual(vt.dynamic_batch_size(-3.0, B33, 10, cap=8), 1)
        self.assertEqual(vt.dynamic_batch_size(100.0, D600, 10, cap=8), 8)

    def test_unknown_model_uses_conservative_default(self):
        self.assertEqual(vt.dynamic_batch_size(1.6, "org/custom", 10, cap=64), int(1.6 * 0.8 / vt.NLLB_PER_ITEM_DEFAULT_GB))


class TestWhisperComputeType(unittest.TestCase):
    def test_fp16_with_room_int8_weights_on_small_cards(self):
        self.assertEqual(vt.whisper_compute_type(8), "float16")
        self.assertEqual(vt.whisper_compute_type(6), "float16")
        self.assertEqual(vt.whisper_compute_type(5), "int8_float16")
        self.assertEqual(vt.whisper_compute_type(0), "int8_float16")


class TestDescribeBudget(unittest.TestCase):
    def test_line_mentions_every_input(self):
        line = vt.describe_budget(8, 4.5, D13, 16, 16)
        for token in ("8 GB card", "4.5 GB free", "2.6 GB weights", "batch 16", "cap 16"):
            self.assertIn(token, line)
        self.assertIn("unknown weights", vt.describe_budget(8, 4.5, "org/custom", 1, 1))


class TestProfileAndCaps(unittest.TestCase):
    def test_profile_thresholds(self):
        cases = {32: "ULTRA", 24: "ULTRA", 16: "HIGH", 12: "HIGH", 11: "MID", 8: "MID", 6: "MID", 5: "LOW", 3: "LOW", 0: "LOW"}
        for vram, expected in cases.items():
            with self.subTest(vram=vram):
                self.assertEqual(models._resolve_hardware_profile(vram), expected)

    def test_caps_are_generous_because_throughput_scales_with_batch(self):
        for profile, cap in {"ULTRA": 16, "HIGH": 16, "MID": 16, "LOW": 8, "CPU_ONLY": 2}.items():
            with self.subTest(profile=profile):
                self.assertEqual(models._PROFILE_TUNING[profile][0], cap)

    def test_low_profile_tuning_applied(self):
        opt = models.SystemOptimizer()
        opt.profile = "LOW"
        opt._apply_profile_tuning()
        self.assertEqual(opt.config["nllb_batch"], 8)
        self.assertEqual(opt.config["translategemma_batch"], 1)

    def test_effective_vram_honours_cap(self):
        opt = models.SystemOptimizer()
        opt.vram_gb = 32
        self.assertEqual(opt.effective_vram_gb(), 32)
        opt.config["max_vram_usage_gb"] = 10
        self.assertEqual(opt.effective_vram_gb(), 10)
        opt.config["max_vram_usage_gb"] = 64
        self.assertEqual(opt.effective_vram_gb(), 32)


def _patch_nllb_setting(model_id, vram_gb):
    """Temporarily set the configured NLLB model and the detected VRAM."""
    stack = contextlib.ExitStack()
    stack.enter_context(patch.object(config, "NLLB_MODEL_ID", model_id))
    stack.enter_context(patch.object(models.OPTIMIZER, "vram_gb", vram_gb))
    return stack


class TestResolveNllbModelId(unittest.TestCase):
    def test_auto_logs_choice_at_info(self):
        with _patch_nllb_setting("auto", 7), patch.object(models.LOGGER, "log") as mock_log:
            self.assertEqual(models.resolve_nllb_model_id(), D13)
        self.assertEqual(mock_log.call_args[0][0], logging.INFO)

    def test_explicit_oversized_model_logs_warning(self):
        with _patch_nllb_setting(B33, 7), patch.object(models.LOGGER, "log") as mock_log:
            self.assertEqual(models.resolve_nllb_model_id(), B33)
        self.assertEqual(mock_log.call_args[0][0], logging.WARNING)

    def test_explicit_fitting_model_logs_nothing(self):
        with _patch_nllb_setting(B33, 32), patch.object(models.LOGGER, "log") as mock_log:
            self.assertEqual(models.resolve_nllb_model_id(), B33)
        mock_log.assert_not_called()

    def test_vram_cap_changes_auto_choice(self):
        with _patch_nllb_setting("auto", 32), patch.dict(models.OPTIMIZER.config, {"max_vram_usage_gb": 8}):
            self.assertEqual(models.resolve_nllb_model_id(), D13)


class TestApplyDynamicTranslationBatch(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, models.OPTIMIZER, "vram_gb", models.OPTIMIZER.vram_gb)
        self._config_backup = dict(models.OPTIMIZER.config)
        self.addCleanup(lambda: models.OPTIMIZER.config.update(self._config_backup))
        models.OPTIMIZER.vram_gb = 8
        models.OPTIMIZER.config["nllb_batch"] = 16
        models.OPTIMIZER.config["nllb_batch_overridden"] = False

    def test_sizes_batch_from_free_vram_and_logs(self):
        logger = MagicMock()
        with (
            patch("modules.models._should_try_cuda_whisper", return_value=True),
            patch("modules.models._cuda_free_gb", return_value=1.0),
            patch.object(config, "NLLB_NUM_BEAMS", 10),
        ):
            batch = models.apply_dynamic_translation_batch(D13, logger)
        # 1.0 GB * 0.8 / 0.10 = 8
        self.assertEqual(batch, 8)
        self.assertEqual(models.OPTIMIZER.config["nllb_batch"], 8)
        self.assertIn("batch 8 (profile cap 16)", logger.call_args[0][0])

    def test_plenty_of_free_vram_keeps_the_cap(self):
        with (
            patch("modules.models._should_try_cuda_whisper", return_value=True),
            patch("modules.models._cuda_free_gb", return_value=20.0),
        ):
            self.assertEqual(models.apply_dynamic_translation_batch(D13), 16)

    def test_user_override_is_respected(self):
        models.OPTIMIZER.config["nllb_batch"] = 3
        models.OPTIMIZER.config["nllb_batch_overridden"] = True
        with patch("modules.models._cuda_free_gb", return_value=20.0) as free:
            self.assertEqual(models.apply_dynamic_translation_batch(D13), 3)
        free.assert_not_called()

    def test_no_cuda_keeps_profile_cap(self):
        with patch("modules.models._should_try_cuda_whisper", return_value=False):
            self.assertEqual(models.apply_dynamic_translation_batch(D13), 16)

    def test_unreadable_free_memory_keeps_cap(self):
        with (
            patch("modules.models._should_try_cuda_whisper", return_value=True),
            patch("modules.models._cuda_free_gb", return_value=None),
        ):
            self.assertEqual(models.apply_dynamic_translation_batch(D13), 16)

    def test_cuda_free_gb_reads_mem_get_info_and_tolerates_failure(self):
        fake_torch = MagicMock()
        fake_torch.cuda.mem_get_info.return_value = (2 * 1024**3, 8 * 1024**3)
        with patch("modules.models.torch", fake_torch):
            self.assertAlmostEqual(models._cuda_free_gb(), 2.0)
        fake_torch.cuda.mem_get_info.side_effect = RuntimeError("no device")
        with patch("modules.models.torch", fake_torch):
            self.assertIsNone(models._cuda_free_gb())


class TestWhisperComputeWiring(unittest.TestCase):
    def test_cuda_whisper_uses_int8_weights_on_small_card(self):
        factory = MagicMock()
        with patch.object(models.OPTIMIZER, "vram_gb", 4), patch.dict(models.OPTIMIZER.config, {"max_vram_usage_gb": 0}):
            models._build_cuda_whisper_model(factory)
        self.assertEqual(factory.call_args.kwargs["compute_type"], "int8_float16")

    def test_cuda_whisper_uses_fp16_with_room(self):
        factory = MagicMock()
        with patch.object(models.OPTIMIZER, "vram_gb", 8), patch.dict(models.OPTIMIZER.config, {"max_vram_usage_gb": 0}):
            models._build_cuda_whisper_model(factory)
        self.assertEqual(factory.call_args.kwargs["compute_type"], "float16")


class TestConfigVramOverrides(unittest.TestCase):
    def test_explicit_nllb_batch_marks_override(self):
        opt = models.SystemOptimizer()
        config._load_performance_overrides({"nllb_batch": 4}, opt, MagicMock())
        self.assertEqual(opt.config["nllb_batch"], 4)
        self.assertTrue(opt.config["nllb_batch_overridden"])

    def test_max_vram_cap_loaded_and_validated(self):
        opt = models.SystemOptimizer()
        logger = MagicMock()
        config._load_performance_overrides({"max_vram_usage_gb": 10.5}, opt, logger)
        # Fractional caps are preserved: 0.5 must not truncate to "no cap".
        self.assertEqual(opt.config["max_vram_usage_gb"], 10.5)
        config._load_performance_overrides({"max_vram_usage_gb": 0.5}, opt, logger)
        self.assertEqual(opt.config["max_vram_usage_gb"], 0.5)
        opt.vram_gb = 8
        self.assertEqual(opt.effective_vram_gb(), 0.5)
        self.assertIn("max_vram_usage_gb", logger.call_args[0][0])
        with self.assertRaises(ValueError):
            config._load_performance_overrides({"max_vram_usage_gb": 0}, opt, logger)

    def test_default_model_id_is_auto(self):
        config._reset_config_defaults()
        self.assertEqual(config.NLLB_MODEL_ID, "auto")


class TestPinnedBatchSurvivesDetection(unittest.TestCase):
    def test_profile_tuning_keeps_a_pinned_nllb_batch(self):
        # The worker loads config (pinning nllb_batch) and only then detects hardware.
        opt = models.SystemOptimizer()
        opt.config["nllb_batch"] = 3
        opt.config["nllb_batch_overridden"] = True
        opt.profile = "ULTRA"
        opt._apply_profile_tuning()
        self.assertEqual(opt.config["nllb_batch"], 3)
        opt.config["nllb_batch_overridden"] = False
        opt._apply_profile_tuning()
        self.assertEqual(opt.config["nllb_batch"], models._PROFILE_TUNING["ULTRA"][0])


class TestWorkerBatchHook(unittest.TestCase):
    def test_nllb_worker_tunes_batch_after_model_load(self):
        from modules.pipeline import isolated_translator

        translator = MagicMock()
        translator.model_id = D13
        with (
            patch.object(config, "TRANSLATOR_ENGINE", "nllb"),
            patch("modules.pipeline.isolated_translator.apply_dynamic_translation_batch") as tune,
        ):
            isolated_translator._tune_batch_for_free_vram(translator)
        tune.assert_called_once()
        self.assertEqual(tune.call_args[0][0], D13)

    def test_translategemma_worker_is_left_alone(self):
        from modules.pipeline import isolated_translator

        with (
            patch.object(config, "TRANSLATOR_ENGINE", "translategemma"),
            patch("modules.pipeline.isolated_translator.apply_dynamic_translation_batch") as tune,
        ):
            isolated_translator._tune_batch_for_free_vram(MagicMock())
        tune.assert_not_called()

    def test_worker_runtime_tunes_before_resolving_batch(self):
        from modules.pipeline import isolated_translator

        order = []
        with (
            patch("modules.pipeline.isolated_translator.config.load_config", return_value=True),
            patch("modules.pipeline.isolated_translator.OPTIMIZER.detect_hardware"),
            patch("modules.pipeline.isolated_translator.ModelManager"),
            patch(
                "modules.pipeline.isolated_translator._resolve_worker_translator", side_effect=lambda m: order.append("load") or MagicMock()
            ),
            patch("modules.pipeline.isolated_translator._tune_batch_for_free_vram", side_effect=lambda t: order.append("tune")),
            patch("modules.pipeline.isolated_translator._resolve_translation_batch_size", side_effect=lambda: order.append("size") or 7),
        ):
            batch, _translator = isolated_translator._build_worker_runtime(0)
        self.assertEqual(order, ["load", "tune", "size"])
        self.assertEqual(batch, 7)


class TestNllbBackendResolution(unittest.TestCase):
    def test_explicit_id_wins(self):
        from modules.translators import nllb

        self.assertEqual(nllb._resolve_model_id(D600), D600)

    def test_auto_resolves_against_the_card(self):
        from modules.translators import nllb

        with patch.object(config, "NLLB_MODEL_ID", "auto"), patch("modules.translators.nllb._cuda_total_gb", return_value=7):
            self.assertEqual(nllb._resolve_model_id(), D13)
        with patch.object(config, "NLLB_MODEL_ID", "auto"), patch("modules.translators.nllb._cuda_total_gb", return_value=32):
            self.assertEqual(nllb._resolve_model_id(), B33)

    def test_cuda_total_gb_without_cuda_is_zero(self):
        from modules.translators import nllb

        fake = MagicMock()
        fake.cuda.is_available.return_value = False
        with patch("modules.translators.nllb.torch", fake):
            self.assertEqual(nllb._cuda_total_gb(), 0)
        with patch("modules.translators.nllb.torch", None):
            self.assertEqual(nllb._cuda_total_gb(), 0)

    def test_cuda_total_gb_reads_device_properties_and_tolerates_failure(self):
        from modules.translators import nllb

        fake = MagicMock()
        fake.cuda.is_available.return_value = True
        fake.cuda.device_count.return_value = 1
        fake.cuda.get_device_properties.return_value.total_memory = 8 * 1024**3
        with patch("modules.translators.nllb.torch", fake):
            self.assertEqual(nllb._cuda_total_gb(), 8)
        fake.cuda.get_device_properties.side_effect = RuntimeError("no device")
        with patch("modules.translators.nllb.torch", fake):
            self.assertEqual(nllb._cuda_total_gb(), 0)

    def test_manager_passes_resolved_id_to_backend(self):
        with (
            patch("modules.models.resolve_nllb_model_id", return_value=D13) as resolve,
            patch("modules.models.nllb_backend.NLLBTranslator") as backend,
        ):
            manager = models.ModelManager()
            manager.get_nllb()
        resolve.assert_called_once()
        backend.assert_called_once_with(model_id=D13)


class TestCpuFallbackIsLoud(unittest.TestCase):
    def test_oom_fallback_to_cpu_logs_a_warning(self):
        from modules.translators import nllb

        calls = []

        def loader(_factory, kwargs, _model_id):
            calls.append(kwargs)
            if kwargs.get("device_map") == "cuda:0":
                raise RuntimeError("CUDA out of memory. Tried to allocate 32 MiB")
            return "cpu-model"

        with (
            patch("modules.translators.nllb._build_nllb_model_kwargs", return_value={"device_map": "cuda:0"}),
            patch("modules.translators.nllb._load_nllb_from_pretrained_fallback", side_effect=loader),
            patch("modules.translators.nllb._clear_cuda_cache"),
            patch.object(nllb.LOGGER, "warning") as warn,
        ):
            self.assertEqual(nllb._load_nllb_model(MagicMock(), B33), "cpu-model")
        self.assertEqual([c.get("device_map") for c in calls], ["cuda:0", "cpu"])
        warn.assert_called_once()
        self.assertIn("Falling back to CPU", warn.call_args[0][0])
        self.assertEqual(warn.call_args[0][1], B33)


class TestLengthSortedBatching(unittest.TestCase):
    """Cues are batched by length to limit padding, and results land back in cue order."""

    def _run(self, texts, batch_size):
        from modules.pipeline import isolated_translator

        data = [{"text": t, "start": float(i), "end": float(i + 1)} for i, t in enumerate(texts)]
        translator = MagicMock()
        seen_batches = []

        def translate(chunk, _src, _tgt):
            seen_batches.append(list(chunk))
            return [f"<{t}>" for t in chunk]

        translator.translate.side_effect = translate
        with (
            patch("modules.pipeline.isolated_translator._resolve_translation_batch_size", return_value=batch_size),
            patch("modules.pipeline.isolated_translator.utils.print_progress_bar") as bar,
            patch("modules.pipeline.isolated_translator._cleanup_intermediate_memory"),
        ):
            out = isolated_translator._run_job_batches(data, translator, {"src_code": "s", "tgt_code": "t", "prefix_str": "p"})
        return out, seen_batches, bar

    def test_results_are_in_original_cue_order(self):
        texts = ["a much longer sentence here", "b", "ccc", "dd", "eeeee eeeee eeeee"]
        out, _, _ = self._run(texts, batch_size=2)
        self.assertEqual(out, [f"<{t}>" for t in texts])

    def test_batches_group_similar_lengths(self):
        texts = ["a much longer sentence here", "b", "ccc", "dd", "eeeee eeeee eeeee"]
        _, batches, _ = self._run(texts, batch_size=2)
        # Shortest two first, then the next two, then the longest alone.
        self.assertEqual(batches[0], ["b", "dd"])
        self.assertEqual(batches[1], ["ccc", "eeeee eeeee eeeee"])
        self.assertEqual(batches[2], ["a much longer sentence here"])

    def test_progress_reports_cue_counts_and_finishes_at_total(self):
        texts = ["x"] * 5
        _, _, bar = self._run(texts, batch_size=2)
        done = [c.args[0] for c in bar.call_args_list]
        self.assertEqual(done, [2, 4, 5])
        self.assertEqual(bar.call_args.args[1], 5)
        self.assertIn("5/5 cues", bar.call_args.kwargs["timestamp_str"])

    def test_empty_input(self):
        out, batches, bar = self._run([], batch_size=4)
        self.assertEqual(out, [])
        self.assertEqual(batches, [])
        bar.assert_not_called()

    def test_short_translator_result_is_an_error_not_a_blank(self):
        from modules.pipeline import isolated_translator

        data = [{"text": "one", "start": 0.0, "end": 1.0}, {"text": "two", "start": 1.0, "end": 2.0}]
        translator = MagicMock()
        translator.translate.return_value = ["only-one"]
        with (
            patch("modules.pipeline.isolated_translator._resolve_translation_batch_size", return_value=8),
            patch("modules.pipeline.isolated_translator.utils.print_progress_bar"),
        ):
            with self.assertRaises(RuntimeError):
                isolated_translator._run_job_batches(data, translator, {"src_code": "s", "tgt_code": "t", "prefix_str": "p"})


class TestVramCapReprofiles(unittest.TestCase):
    """A VRAM cap must move the whole tier, not only the model choice."""

    def _gpu_optimizer(self, vram_gb, name="NVIDIA GeForce RTX 5090"):
        opt = models.SystemOptimizer()
        with patch.object(opt, "_detect_gpu_props", return_value=(name, vram_gb, models._resolve_hardware_profile(vram_gb))):
            opt.detect_hardware()
        return opt

    def test_cap_lowers_the_tier_and_its_batch_cap(self):
        opt = self._gpu_optimizer(31)
        self.assertEqual((opt.profile, opt.config["nllb_batch"]), ("ULTRA", 16))
        opt.apply_vram_cap(4)
        self.assertEqual((opt.profile, opt.config["nllb_batch"]), ("LOW", 8))
        self.assertEqual(opt.effective_vram_gb(), 4)

    def test_detect_after_cap_keeps_the_capped_tier(self):
        # Both call orders (load_config before or after detect_hardware) must agree.
        opt = self._gpu_optimizer(31)
        opt.apply_vram_cap(8)
        with patch.object(opt, "_detect_gpu_props", return_value=("NVIDIA GeForce RTX 5090", 31, "ULTRA")):
            opt.detect_hardware()
        self.assertEqual(opt.profile, "MID")

    def test_cap_above_card_changes_nothing(self):
        opt = self._gpu_optimizer(8)
        opt.apply_vram_cap(64)
        self.assertEqual(opt.profile, "MID")
        self.assertEqual(opt.effective_vram_gb(), 8)

    def test_cpu_only_is_untouched_by_a_cap(self):
        opt = models.SystemOptimizer()
        with patch.object(opt, "_detect_gpu_props", return_value=("CPU", 0, "CPU_ONLY")):
            opt.detect_hardware()
        opt.apply_vram_cap(4)
        self.assertEqual(opt.profile, "CPU_ONLY")

    def test_apple_silicon_never_promoted_past_high(self):
        opt = self._gpu_optimizer(32, name="Apple Silicon (MPS)")
        opt.apply_vram_cap(30)
        self.assertEqual(opt.profile, "HIGH")

    def test_config_applies_cap_before_explicit_batch_override(self):
        opt = self._gpu_optimizer(31)
        config._load_performance_overrides({"max_vram_usage_gb": 4, "nllb_batch": 3}, opt, MagicMock())
        # The cap re-tuned to LOW (cap 8), then the explicit 3 won and was marked as an override.
        self.assertEqual(opt.profile, "LOW")
        self.assertEqual(opt.config["nllb_batch"], 3)
        self.assertTrue(opt.config["nllb_batch_overridden"])


class TestCapBoundsFreeMemorySizing(unittest.TestCase):
    def setUp(self):
        self._backup = dict(models.OPTIMIZER.config)
        self.addCleanup(lambda: models.OPTIMIZER.config.update(self._backup))
        self.addCleanup(setattr, models.OPTIMIZER, "vram_gb", models.OPTIMIZER.vram_gb)
        models.OPTIMIZER.vram_gb = 31
        models.OPTIMIZER.config.update({"nllb_batch": 16, "nllb_batch_overridden": False})

    def test_without_a_cap_real_free_memory_decides(self):
        models.OPTIMIZER.config["max_vram_usage_gb"] = 0
        self.assertEqual(models._budget_headroom_gb(B33), float("inf"))

    def test_cap_limits_activation_budget_even_when_the_card_is_empty(self):
        # 4 GB cap, 600M weights 1.2 GB -> only 2.8 GB may go to activations however much is free.
        models.OPTIMIZER.config["max_vram_usage_gb"] = 4
        self.assertAlmostEqual(models._budget_headroom_gb(D600), 2.8)
        with (
            patch("modules.models._should_try_cuda_whisper", return_value=True),
            patch("modules.models._cuda_free_gb", return_value=27.3),
            patch.object(config, "NLLB_NUM_BEAMS", 10),
        ):
            batch = models.apply_dynamic_translation_batch(D600)
        # 2.8 * 0.8 / 0.07 = 32 -> still capped by the profile (16 here); never sized off 27 GB.
        self.assertEqual(batch, 16)
        models.OPTIMIZER.config["max_vram_usage_gb"] = 2
        with (
            patch("modules.models._should_try_cuda_whisper", return_value=True),
            patch("modules.models._cuda_free_gb", return_value=27.3),
            patch.object(config, "NLLB_NUM_BEAMS", 10),
        ):
            # 0.8 GB headroom * 0.8 / 0.07 = 9 -> the cap, not the free memory, sized it.
            self.assertEqual(models.apply_dynamic_translation_batch(D600), 9)


if __name__ == "__main__":
    unittest.main()
