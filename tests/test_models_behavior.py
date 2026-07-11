import unittest
from unittest.mock import MagicMock, patch
import os
import sys
import importlib

# Ensure modules can be imported
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

models = importlib.import_module("modules.models")


class TestCoverageModels(unittest.TestCase):
    def setUp(self):
        from modules import config

        previous_engine = config.TRANSLATOR_ENGINE
        self.addCleanup(setattr, config, "TRANSLATOR_ENGINE", previous_engine)
        config.TRANSLATOR_ENGINE = "nllb"

    def test_system_optimizer_cores_exception(self):
        with patch("multiprocessing.cpu_count", side_effect=TypeError()):
            opt = models.SystemOptimizer()
            self.assertEqual(opt.cpu_cores, 1)

    def test_segment_helpers(self):
        seg = models.Segment(1.0, 3.5, "hello")
        self.assertEqual(seg.duration(), 2.5)
        self.assertEqual(seg.as_tuple(), (1.0, 3.5, "hello"))

    def test_getattr_invalid_raises(self):
        with self.assertRaises(AttributeError):
            getattr(models, "DOES_NOT_EXIST")

    def test_import_optional_module_missing(self):
        with patch("importlib.import_module", side_effect=ImportError()):
            self.assertIsNone(models._import_optional_module("missing.module"))

    def test_component_helpers_return_none_when_module_missing(self):
        with patch("modules.models._import_optional_module", return_value=None):
            self.assertEqual(models._get_nllb_components(), (None, None))
            self.assertEqual(models._get_translategemma_components(), (None, None))
            self.assertEqual(models._get_faster_whisper_components(), (None, None))
            self.assertIsNone(models._get_separator_class())

    def test_detect_gpu_when_torch_missing(self):
        opt = models.SystemOptimizer()
        with patch("modules.models._get_torch_module", return_value=None), patch("modules.models.log") as mock_log:
            opt._detect_gpu(verbose=True)
            mock_log.assert_called()

    def test_detect_hardware_verbose(self):
        opt = models.SystemOptimizer()
        with patch("modules.models.log") as mock_log, patch.object(opt, "_detect_gpu"), patch.object(opt, "_assign_profile"):
            opt.detect_hardware(verbose=True)
            mock_log.assert_called()

    def test_detect_gpu_mem_exception(self):
        opt = models.SystemOptimizer()
        mock_props = MagicMock()
        type(mock_props).total_memory = property(lambda x: "invalid")
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_properties", return_value=mock_props),
            patch("modules.models.log"),
        ):
            opt._detect_gpu()
            self.assertEqual(opt.vram_gb, 0.0)

    def test_detect_gpu_verbose(self):
        opt = models.SystemOptimizer()
        mock_props = MagicMock()
        mock_props.total_memory = 8 * 1024**3
        mock_props.name = "TestGPU"
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_properties", return_value=mock_props),
            patch("modules.models.log") as mock_log,
        ):
            opt._detect_gpu(verbose=True)
            mock_log.assert_called()

    def test_assign_profile_vram_exception(self):
        opt = models.SystemOptimizer()
        opt.config["device"] = "cuda"
        opt.vram_gb = "invalid"
        with patch.object(opt, "set_profile") as mock_set:
            opt._assign_profile()
            mock_set.assert_called_with("LOW", verbose=True)

    def test_set_profile_invalid(self):
        opt = models.SystemOptimizer()
        with patch("modules.models.log") as mock_log:
            opt.set_profile("INVALID")
            self.assertEqual(opt.profile, "STANDARD")
            mock_log.assert_called()

    def test_set_profile_respects_beam_override(self):
        opt = models.SystemOptimizer()
        opt.vram_gb = 24
        opt.config["whisper_beam"] = 9
        opt.config["whisper_beam_overridden"] = True
        opt.set_profile("ULTRA", verbose=False)
        self.assertEqual(opt.config["whisper_beam"], 9)

    def test_set_profile_verbose_logs(self):
        opt = models.SystemOptimizer()
        opt.vram_gb = 24
        with patch("modules.models.log") as mock_log:
            opt.set_profile("ULTRA", verbose=True)
            mock_log.assert_any_call("[Optimization] Applied Profile: ULTRA")

    def test_disable_default_max_length(self):
        mock_model = MagicMock()
        mock_model.generation_config = MagicMock()
        mock_model.generation_config.max_length = 200

        models._disable_default_max_length(mock_model)

        self.assertIsNone(mock_model.generation_config.max_length)

    def test_sanitize_generation_kwargs(self):
        params = {"max_length": 200, "num_beams": 5}

        cleaned = models._sanitize_generation_kwargs(params)

        self.assertNotIn("max_length", cleaned)
        self.assertEqual(cleaned["num_beams"], 5)

    def test_sanitize_generation_kwargs_drops_early_stopping_without_beams(self):
        params = {"early_stopping": True, "do_sample": False}

        cleaned = models._sanitize_generation_kwargs(params)

        self.assertNotIn("early_stopping", cleaned)

    def test_sanitize_generation_kwargs_keeps_early_stopping_with_beams(self):
        params = {"early_stopping": True, "num_beams": 4, "do_sample": False}

        cleaned = models._sanitize_generation_kwargs(params)

        self.assertTrue(cleaned["early_stopping"])

    def test_normalize_translategemma_lang_code(self):
        self.assertEqual(models._normalize_translategemma_lang_code("no"), "nb")
        self.assertEqual(models._normalize_translategemma_lang_code("fi"), "fi")

    def test_nllb_translator_load_lazy(self):
        # Test the lazy import logic
        with (
            patch("transformers.NllbTokenizer"),
            patch("transformers.AutoModelForSeq2SeqLM"),
            patch("torch.backends.cuda.matmul.allow_tf32"),
            patch("modules.models.log"),
            patch("torch.cuda.is_available", return_value=False),
        ):
            # Reset globals to force re-import check
            models.torch = None
            models.NllbTokenizer = None
            models.AutoModelForSeq2SeqLM = None

            # This will trigger _load
            _ = models.NLLBTranslator()

    def test_nllb_translator_load_error(self):
        with patch("transformers.NllbTokenizer.from_pretrained", side_effect=Exception("Load fail")), patch("modules.models.log"):
            with self.assertRaises(Exception):
                models.NLLBTranslator()

    def test_nllb_translator_load_missing_components(self):
        with patch("modules.models._get_nllb_components", return_value=(None, None)):
            with self.assertRaises(RuntimeError):
                models.NLLBTranslator()

    def test_nllb_translator_load_warmup(self):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        with (
            patch("transformers.NllbTokenizer.from_pretrained", return_value=mock_tokenizer),
            patch("transformers.AutoModelForSeq2SeqLM.from_pretrained", return_value=mock_model),
            patch("modules.models.OPTIMIZER") as mock_opt,
            patch("modules.models.log"),
        ):
            mock_opt.config = {"device": "cuda"}
            models.NLLBTranslator()
            mock_model.generate.assert_called()

    def test_nllb_translator_translate_none(self):
        # Use a dummy load to avoid full init
        with patch("modules.models.NLLBTranslator._load"):
            trans = models.NLLBTranslator()
            trans.model = None
            self.assertEqual(trans.translate(None, "en", "es"), None)
            self.assertEqual(trans.translate([], "en", "es"), [])

    def test_nllb_translator_translate_requires_torch(self):
        with patch("modules.models.NLLBTranslator._load"):
            trans = models.NLLBTranslator()
            trans.model = MagicMock()
            trans.tokenizer = MagicMock()
            with patch("modules.models._get_torch_module", return_value=None):
                with self.assertRaises(RuntimeError):
                    trans.translate(["Hello"], "en", "es")

    def test_nllb_translator_translate_full(self):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.batch_decode.return_value = ["Hola"]
        with patch("modules.models.NLLBTranslator._load"):
            trans = models.NLLBTranslator()
            trans.model = mock_model
            trans.tokenizer = mock_tokenizer
            res = trans.translate(["Hello"], "en", "es")
            self.assertEqual(res, ["Hola"])
            mock_model.generate.assert_called()

    def test_nllb_translator_offload(self):
        mock_model = MagicMock()
        with patch("modules.models.NLLBTranslator._load"):
            trans = models.NLLBTranslator()
            trans.model = mock_model
            trans.offload()
            mock_model.to.assert_called_with("cpu")

    def test_translategemma_translator_load_lazy(self):
        with (
            patch("transformers.AutoProcessor"),
            patch("transformers.AutoModelForImageTextToText"),
            patch("torch.backends.cuda.matmul.allow_tf32"),
            patch("modules.models.log"),
            patch("torch.cuda.is_available", return_value=False),
        ):
            models.torch = None
            _ = models.TranslateGemmaTranslator()

    def test_translategemma_translator_load_error(self):
        with patch("transformers.AutoProcessor.from_pretrained", side_effect=Exception("Load fail")), patch("modules.models.log"):
            with self.assertRaises(Exception):
                models.TranslateGemmaTranslator()

    def test_translategemma_translator_load_missing_components(self):
        with patch("modules.models._get_translategemma_components", return_value=(None, None)):
            with self.assertRaises(RuntimeError):
                models.TranslateGemmaTranslator()

    def test_translategemma_translator_load_warmup(self):
        mock_model = MagicMock()
        mock_processor = MagicMock()
        with (
            patch("transformers.AutoProcessor.from_pretrained", return_value=mock_processor),
            patch("transformers.AutoModelForImageTextToText.from_pretrained", return_value=mock_model),
            patch("modules.models.OPTIMIZER") as mock_opt,
            patch("modules.models.log"),
        ):
            mock_opt.config = {"device": "cuda"}
            models.TranslateGemmaTranslator()
            mock_model.generate.assert_called()

    def test_translategemma_translator_translate_none(self):
        with patch("modules.models.TranslateGemmaTranslator._load"):
            trans = models.TranslateGemmaTranslator()
            trans.model = None
            self.assertEqual(trans.translate(None, "en", "es"), None)
            self.assertEqual(trans.translate([], "en", "es"), [])

    def test_translategemma_translate_requires_torch(self):
        with patch("modules.models.TranslateGemmaTranslator._load"):
            trans = models.TranslateGemmaTranslator()
            trans.model = MagicMock()
            trans.tokenizer = MagicMock()
            with patch("modules.models._get_torch_module", return_value=None):
                with self.assertRaises(RuntimeError):
                    trans.translate(["Hello"], "en", "es")

    def test_translategemma_gated_error_logging(self):
        with patch("modules.models.TranslateGemmaTranslator._load"):
            trans = models.TranslateGemmaTranslator()
            with patch("modules.models.log") as mock_log:
                failing_cls = MagicMock()
                failing_cls.from_pretrained.side_effect = OSError("401 Client Error: gated repo")
                with self.assertRaises(OSError):
                    trans._perform_translategemma_load(failing_cls, MagicMock(bfloat16="bf16"))
                self.assertTrue(any("GATED MODEL ERROR" in str(call) for call in mock_log.call_args_list))

    def test_translategemma_translator_translate_full(self):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.batch_decode.return_value = ["Hola"]
        mock_inputs = MagicMock()
        mock_inputs.input_ids.shape = (1, 5)
        mock_tokenizer.return_value = mock_inputs

        with patch("modules.models.TranslateGemmaTranslator._load"):
            trans = models.TranslateGemmaTranslator()
            trans.model = mock_model
            trans.tokenizer = mock_tokenizer
            res = trans.translate(["Hello"], "en", "es")
            self.assertEqual(res, ["Hola"])
            mock_model.generate.assert_called()

    def test_translategemma_build_inputs_template_error_wrapped(self):
        with patch("modules.models.TranslateGemmaTranslator._load"):
            trans = models.TranslateGemmaTranslator()
            trans.model = MagicMock()
            trans.tokenizer = MagicMock()
            trans.tokenizer.apply_chat_template.side_effect = Exception("template fail")

            with self.assertRaises(ValueError):
                trans._build_translate_gemma_inputs(["hello"], "en", "nb")

    def test_translategemma_translator_offload(self):
        mock_model = MagicMock()
        with patch("modules.models.TranslateGemmaTranslator._load"):
            trans = models.TranslateGemmaTranslator()
            trans.model = mock_model
            trans.offload()
            mock_model.to.assert_called_with("cpu")

    def test_model_manager_get_translategemma(self):
        mm = models.ModelManager()
        with (
            patch("modules.models.TranslateGemmaTranslator") as mock_gemma,
            patch.object(mm, "offload_whisper"),
            patch.object(mm, "offload_separator"),
            patch("modules.models.config") as mock_config,
            patch("modules.models.log"),
        ):
            mock_config.TRANSLATOR_ENGINE = "translategemma"
            mm.get_nllb()
            mock_gemma.assert_called()

    def test_model_manager_whisper_batch(self):
        mm = models.ModelManager()
        mock_model = MagicMock()
        with (
            patch("faster_whisper.WhisperModel", return_value=mock_model),
            patch("faster_whisper.BatchedInferencePipeline") as mock_pipe,
            patch("modules.models.OPTIMIZER") as mock_opt,
            patch("modules.models.log"),
        ):
            mock_opt.config = {"device": "cpu", "whisper_compute": "int8", "whisper_workers": 1, "whisper_batch_size": 4}
            mm.get_whisper()
            mock_pipe.assert_called()

    def test_model_manager_get_nllb(self):
        mm = models.ModelManager()
        with (
            patch("modules.models.NLLBTranslator") as mock_nllb,
            patch.object(mm, "offload_whisper"),
            patch.object(mm, "offload_separator"),
            patch("modules.models.log"),
        ):
            mm.get_nllb()
            mock_nllb.assert_called()

    def test_model_manager_get_whisper_missing_backend(self):
        mm = models.ModelManager()
        with patch("modules.models._get_faster_whisper_components", return_value=(None, None)):
            with self.assertRaises(RuntimeError):
                mm.get_whisper()

    def test_model_manager_get_whisper_no_batch_pipeline(self):
        mm = models.ModelManager()
        mock_model = MagicMock()
        with (
            patch("modules.models._get_faster_whisper_components", return_value=(MagicMock(return_value=mock_model), MagicMock())),
            patch("modules.models.OPTIMIZER") as mock_opt,
            patch("modules.models.log"),
        ):
            mock_opt.config = {
                "device": "cpu",
                "whisper_compute": "int8",
                "whisper_workers": 1,
                "whisper_batch_size": 1,
            }
            result = mm.get_whisper()
            self.assertIs(result, mock_model)

    def test_model_manager_get_separator(self):
        mm = models.ModelManager()
        with patch("audio_separator.separator.Separator") as mock_sep, patch("modules.models.log"):
            # First call when it is None
            mm.get_separator("test_output_dir")
            mock_sep.assert_called_with(
                model_file_dir=os.path.join(os.getcwd(), "models"), output_dir="test_output_dir", output_single_stem="Vocals"
            )

            # Second call when it is already cached
            mm.get_separator("new_test_output_dir")
            self.assertEqual(mm._separator.output_dir, "new_test_output_dir")
            self.assertEqual(mm._separator.output_single_stem, "Vocals")

    def test_model_manager_get_separator_missing_backend(self):
        mm = models.ModelManager()
        with patch("modules.models._get_separator_class", return_value=None):
            with self.assertRaises(RuntimeError):
                mm.get_separator("test_output_dir")

    def test_model_manager_offload_whisper_base(self):
        mm = models.ModelManager()
        mm._whisper = MagicMock()
        mm._whisper_base = MagicMock()
        with patch("modules.models.log"), patch("torch.cuda.empty_cache"), patch("gc.collect"):
            mm.offload_whisper()
            self.assertIsNone(mm._whisper)

    def test_model_manager_offload_separator(self):
        mm = models.ModelManager()
        mm._separator = MagicMock()
        with patch("modules.models.log"), patch("torch.cuda.empty_cache"), patch("gc.collect"):
            mm.offload_separator()
            self.assertIsNone(mm._separator)

    def test_model_manager_offload_preload_nllb(self):
        mm = models.ModelManager()
        mm._nllb = MagicMock()
        with patch("modules.models.log"):
            mm.offload_nllb()
            mm._nllb.offload.assert_called()

            with patch.object(mm, "get_nllb") as mock_get:
                mm.preload_nllb()
                mock_get.assert_called()


if __name__ == "__main__":
    unittest.main()
