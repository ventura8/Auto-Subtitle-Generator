from modules import models
import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Ensure modules can be imported
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)


class TestCoverageModels(unittest.TestCase):

    def test_system_optimizer_cores_exception(self):
        with patch("multiprocessing.cpu_count", side_effect=TypeError()):
            opt = models.SystemOptimizer()
            self.assertEqual(opt.cpu_cores, 1)

    def test_detect_hardware_verbose(self):
        opt = models.SystemOptimizer()
        with patch("modules.models.log") as mock_log, \
                patch.object(opt, "_detect_gpu"), \
                patch.object(opt, "_assign_profile"):
            opt.detect_hardware(verbose=True)
            mock_log.assert_called()

    def test_detect_gpu_mem_exception(self):
        opt = models.SystemOptimizer()
        mock_props = MagicMock()
        type(mock_props).total_memory = property(lambda x: "invalid")
        with patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.get_device_properties", return_value=mock_props), \
                patch("modules.models.log"):
            opt._detect_gpu()
            self.assertEqual(opt.vram_gb, 0.0)

    def test_detect_gpu_verbose(self):
        opt = models.SystemOptimizer()
        mock_props = MagicMock()
        mock_props.total_memory = 8 * 1024**3
        mock_props.name = "TestGPU"
        with patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.get_device_properties", return_value=mock_props), \
                patch("modules.models.log") as mock_log:
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

    def test_set_profile_verbose_logs(self):
        opt = models.SystemOptimizer()
        opt.vram_gb = 24
        with patch("modules.models.log") as mock_log:
            opt.set_profile("ULTRA", verbose=True)
            mock_log.assert_any_call("[Optimization] Applied Profile: ULTRA")

    def test_nllb_translator_load_lazy(self):
        # Test the lazy import logic
        with patch("transformers.NllbTokenizer"), \
                patch("transformers.AutoModelForSeq2SeqLM"), \
                patch("torch.backends.cuda.matmul.allow_tf32"), \
                patch("modules.models.log"), \
                patch("torch.cuda.is_available", return_value=False):
            # Reset globals to force re-import check
            models.torch = None
            models.NllbTokenizer = None
            models.AutoModelForSeq2SeqLM = None

            # This will trigger _load
            _ = models.NLLBTranslator()

    def test_nllb_translator_load_error(self):
        with patch("transformers.NllbTokenizer.from_pretrained", side_effect=Exception("Load fail")), \
                patch("modules.models.log"):
            with self.assertRaises(Exception):
                models.NLLBTranslator()

    def test_nllb_translator_load_warmup(self):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        with patch("transformers.NllbTokenizer.from_pretrained", return_value=mock_tokenizer), \
                patch("transformers.AutoModelForSeq2SeqLM.from_pretrained", return_value=mock_model), \
                patch("modules.models.OPTIMIZER") as mock_opt, \
                patch("modules.models.log"):
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

    def test_model_manager_whisper_batch(self):
        mm = models.ModelManager()
        mock_model = MagicMock()
        with patch("faster_whisper.WhisperModel", return_value=mock_model), \
                patch("faster_whisper.BatchedInferencePipeline") as mock_pipe, \
                patch("modules.models.OPTIMIZER") as mock_opt, \
                patch("modules.models.log"):
            mock_opt.config = {"device": "cpu", "whisper_compute": "int8", "whisper_workers": 1, "whisper_batch_size": 4}
            mm.get_whisper()
            mock_pipe.assert_called()

    def test_model_manager_get_nllb(self):
        mm = models.ModelManager()
        with patch("modules.models.NLLBTranslator") as mock_nllb, \
                patch.object(mm, "offload_whisper"), \
                patch.object(mm, "offload_separator"), \
                patch("modules.models.log"):
            mm.get_nllb()
            mock_nllb.assert_called()

    def test_model_manager_get_separator(self):
        mm = models.ModelManager()
        with patch("audio_separator.separator.Separator") as mock_sep, \
                patch("modules.models.log"):
            mm.get_separator()
            mock_sep.assert_called()

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
