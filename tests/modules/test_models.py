import unittest
from unittest.mock import MagicMock, patch

from modules import models
from modules.translators import nllb as nllb_backend
from modules.translators import translategemma as translategemma_backend


class TestModels(unittest.TestCase):
    def test_segment_namedtuple(self):
        segment = models.Segment(0.0, 1.0, "hello")
        self.assertEqual(segment.start, 0.0)
        self.assertEqual(segment.end, 1.0)
        self.assertEqual(segment.text, "hello")

    @patch("modules.models.importlib.import_module")
    def test_import_module_helper(self, mock_import_module):
        sentinel = object()
        mock_import_module.return_value = sentinel
        self.assertIs(models._import_module("x.y"), sentinel)
        mock_import_module.assert_called_once_with("x.y")

    def test_system_optimizer_detect_hardware_cpu(self):
        opt = models.SystemOptimizer()
        with patch("modules.models.torch", None):
            opt.detect_hardware(verbose=False)
        self.assertEqual(opt.gpu_name, "CPU")
        self.assertEqual(opt.vram_gb, 0)
        self.assertEqual(opt.profile, "CPU_ONLY")
        self.assertEqual(opt.config["translategemma_batch"], 1)
        self.assertEqual(opt.config["translategemma_max_new_tokens"], 144)

    def test_system_optimizer_detect_hardware_cuda(self):
        opt = models.SystemOptimizer()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        props = MagicMock()
        props.name = "GPU"
        props.total_memory = 12 * 1024**3
        mock_torch.cuda.get_device_properties.return_value = props

        with patch("modules.models.torch", mock_torch):
            opt.detect_hardware(verbose=True)

        self.assertEqual(opt.gpu_name, "GPU")
        self.assertEqual(opt.vram_gb, 12)
        self.assertEqual(opt.profile, "HIGH")
        self.assertEqual(opt.config["translategemma_batch"], 8)
        self.assertEqual(opt.config["translategemma_max_new_tokens"], 192)
        snapshot = opt.snapshot()
        self.assertEqual(snapshot["profile"], "HIGH")

    def test_system_optimizer_detect_hardware_ultra(self):
        opt = models.SystemOptimizer()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        props = MagicMock()
        props.name = "RTX 5090"
        props.total_memory = 32 * 1024**3
        mock_torch.cuda.get_device_properties.return_value = props

        with patch("modules.models.torch", mock_torch):
            opt.detect_hardware(verbose=False)

        self.assertEqual(opt.gpu_name, "RTX 5090")
        self.assertEqual(opt.vram_gb, 32)
        self.assertEqual(opt.profile, "ULTRA")
        self.assertEqual(opt.config["nllb_batch"], 8)
        self.assertEqual(opt.config["translategemma_batch"], 24)
        self.assertEqual(opt.config["translategemma_max_new_tokens"], 192)

    def test_system_optimizer_detect_hardware_mps_high_memory(self):
        opt = models.SystemOptimizer()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True

        with (
            patch("modules.models.torch", mock_torch),
            patch(
                "os.sysconf",
                side_effect=lambda name: 4096 if name == "SC_PAGE_SIZE" else (64 * 1024**3 // 4096),
                create=True,
            ),
        ):
            opt.detect_hardware(verbose=False)

        self.assertEqual(opt.gpu_name, "Apple Silicon (MPS)")
        # Only half of the 64 GB unified pool is treated as usable GPU memory,
        # and the resulting profile is capped below the dedicated-VRAM ULTRA tier.
        self.assertEqual(opt.vram_gb, 32)
        self.assertEqual(opt.profile, "HIGH")
        self.assertEqual(opt.config["nllb_batch"], 8)
        self.assertEqual(opt.config["translategemma_batch"], 8)

    def test_system_optimizer_detect_hardware_mps_low_memory(self):
        opt = models.SystemOptimizer()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True

        with (
            patch("modules.models.torch", mock_torch),
            patch(
                "os.sysconf",
                side_effect=lambda name: 4096 if name == "SC_PAGE_SIZE" else (16 * 1024**3 // 4096),
                create=True,
            ),
        ):
            opt.detect_hardware(verbose=False)

        self.assertEqual(opt.gpu_name, "Apple Silicon (MPS)")
        # Half of the 16 GB unified pool lands below the HIGH threshold.
        self.assertEqual(opt.vram_gb, 8)
        self.assertEqual(opt.profile, "MID")
        self.assertEqual(opt.config["nllb_batch"], 6)

    def test_system_optimizer_detect_hardware_mps_non_positive_memory(self):
        opt = models.SystemOptimizer()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True

        with (
            patch("modules.models.torch", mock_torch),
            patch("os.sysconf", side_effect=OSError("Not supported"), create=True),
        ):
            opt.detect_hardware(verbose=False)

        self.assertEqual(opt.gpu_name, "Apple Silicon (MPS)")
        self.assertEqual(opt.vram_gb, 0)
        self.assertEqual(opt.profile, "CPU_ONLY")

    def test_is_cuda_runtime_missing_error_linux_and_windows(self):
        self.assertTrue(models._is_cuda_runtime_missing_error(RuntimeError("libcublas.so.12: cannot open shared object file")))
        self.assertTrue(models._is_cuda_runtime_missing_error(RuntimeError("libcudnn.so.9: cannot open shared object file")))
        self.assertTrue(models._is_cuda_runtime_missing_error(RuntimeError("cublas64_12.dll was not found")))
        self.assertTrue(models._is_cuda_runtime_missing_error(RuntimeError("Could not load library cublas64_130_0.dll")))
        self.assertTrue(models._is_cuda_runtime_missing_error(RuntimeError("cudart64_130_0 not found")))
        self.assertTrue(models._is_cuda_runtime_missing_error(RuntimeError("cublas64_13.dll was not found")))
        self.assertFalse(models._is_cuda_runtime_missing_error(RuntimeError("Some other random error")))

    def test_cuda_runtime_installation_guidance_is_platform_aware(self):
        with patch("sys.platform", "win32"):
            self.assertIn("install_dependencies.ps1", models._cuda_runtime_installation_guidance())
        with patch("sys.platform", "linux"):
            self.assertIn("install_dependencies.sh", models._cuda_runtime_installation_guidance())

    def test_resolve_hardware_profile(self):
        self.assertEqual(models._resolve_hardware_profile(32), "ULTRA")
        self.assertEqual(models._resolve_hardware_profile(12), "HIGH")
        self.assertEqual(models._resolve_hardware_profile(8), "MID")

    def test_whisper_model_wrapper(self):
        fake_model = MagicMock()
        fake_module = MagicMock()
        fake_module.WhisperModel.return_value = fake_model

        with patch("modules.models._import_module", return_value=fake_module):
            wrapper = models.WhisperModel()
            wrapper.transcribe("a")
            fake_model.transcribe.assert_called_once_with("a")
            wrapper.release()
            self.assertIsNone(wrapper._model)

    def test_whisper_model_falls_back_to_cpu_when_cuda_runtime_missing(self):
        cuda_model = RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
        cpu_model = MagicMock()
        fake_module = MagicMock()
        fake_module.WhisperModel.side_effect = [cuda_model, cpu_model]
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch("modules.models._import_module", return_value=fake_module), patch("modules.models.torch", mock_torch):
            wrapper = models.WhisperModel()

        self.assertIs(wrapper._model, cpu_model)
        self.assertEqual(fake_module.WhisperModel.call_count, 2)
        self.assertEqual(fake_module.WhisperModel.call_args_list[0].kwargs["device"], "cuda")
        self.assertEqual(fake_module.WhisperModel.call_args_list[1].kwargs["device"], "cpu")

    def test_whisper_model_falls_back_to_cpu_for_cudnn_component_dlls(self):
        for dll_name in ("cudnn_ops64_9.dll", "cudnn_cnn64_9.dll"):
            with self.subTest(dll_name=dll_name):
                fake_module = MagicMock()
                cpu_model = MagicMock()
                fake_module.WhisperModel.side_effect = [RuntimeError(dll_name), cpu_model]
                mock_torch = MagicMock()
                mock_torch.cuda.is_available.return_value = True

                with patch("modules.models._import_module", return_value=fake_module), patch("modules.models.torch", mock_torch):
                    wrapper = models.WhisperModel()

                self.assertIs(wrapper._model, cpu_model)
                self.assertEqual(fake_module.WhisperModel.call_args_list[1].kwargs["device"], "cpu")

    def test_whisper_model_transcribe_falls_back_to_cpu_on_lazy_cuda_failure(self):
        gpu_model = MagicMock()
        gpu_model.transcribe.side_effect = RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
        cpu_model = MagicMock()
        cpu_model.transcribe.return_value = ([], MagicMock())
        fake_module = MagicMock()
        fake_module.WhisperModel.side_effect = [gpu_model, cpu_model]
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch("modules.models._import_module", return_value=fake_module), patch("modules.models.torch", mock_torch):
            wrapper = models.WhisperModel()
            result = wrapper.transcribe("audio.wav")

        self.assertEqual(result, ([], cpu_model.transcribe.return_value[1]))
        gpu_model.transcribe.assert_called_once_with("audio.wav")
        cpu_model.transcribe.assert_called_once_with("audio.wav")
        self.assertEqual(fake_module.WhisperModel.call_count, 2)
        self.assertTrue(wrapper._using_cpu)

    def test_nllb_translate_empty(self):
        translator = nllb_backend.NLLBTranslator.__new__(nllb_backend.NLLBTranslator)
        translator._tokenizer = MagicMock()
        translator._model = MagicMock()
        self.assertEqual(translator.translate([], "eng_Latn", "spa_Latn"), [])

    def test_nllb_translate_non_empty_cpu(self):
        translator = nllb_backend.NLLBTranslator.__new__(nllb_backend.NLLBTranslator)
        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": MagicMock()}
        tokenizer.convert_tokens_to_ids.return_value = 42
        tokenizer.batch_decode.return_value = ["hola"]
        model = MagicMock()
        model.generate.return_value = [1, 2, 3]
        translator._tokenizer = tokenizer
        translator._model = model

        with patch("modules.translators.nllb.torch", None):
            result = translator.translate(["hello"], "eng_Latn", "spa_Latn")

        self.assertEqual(result, ["hola"])
        self.assertEqual(translator._tokenizer.src_lang, "eng_Latn")
        model.generate.assert_called_once()

    def test_nllb_translate_non_empty_cuda(self):
        translator = nllb_backend.NLLBTranslator.__new__(nllb_backend.NLLBTranslator)
        encoded_value = MagicMock()
        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": encoded_value}
        tokenizer.convert_tokens_to_ids.return_value = 7
        tokenizer.batch_decode.return_value = ["ciao"]
        model = MagicMock()
        model.generate.return_value = [7]
        translator._tokenizer = tokenizer
        translator._model = model

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch("modules.translators.nllb.torch", mock_torch):
            result = translator.translate(["hello"], "eng_Latn", "ita_Latn")

        self.assertEqual(result, ["ciao"])
        encoded_value.to.assert_called_once_with("cuda")

    def test_nllb_release(self):
        translator = nllb_backend.NLLBTranslator.__new__(nllb_backend.NLLBTranslator)
        translator._model = object()
        translator._tokenizer = object()
        translator.release()
        self.assertIsNone(translator._model)
        self.assertIsNone(translator._tokenizer)

    def test_translategemma_translate_batches_in_single_generate_call(self):
        translator = translategemma_backend.TranslateGemmaTranslator.__new__(translategemma_backend.TranslateGemmaTranslator)
        tokenizer = MagicMock()
        model = MagicMock()

        encoded = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
        }
        encoded["attention_mask"].sum.return_value = [4, 4]
        tokenizer.return_value = encoded
        model.generate.return_value = [
            [10, 11, 12, 13, 100],
            [20, 21, 22, 23],
        ]
        tokenizer.decode.side_effect = ["hola", ""]

        translator._tokenizer = tokenizer
        translator._model = model
        translator._device = None
        translator._runtime_settings = {"max_new_tokens": 128}

        with patch("modules.configuration.config.nllb_to_iso", side_effect=lambda value: value):
            result = translator.translate(["hello", "world"], "en", "es")

        self.assertEqual(result, ["hola", "world"])
        self.assertEqual(model.generate.call_count, 1)
        tokenizer.assert_called_once()

    def test_build_translategemma_model_kwargs_uses_dtype(self):
        mock_torch = MagicMock()
        mock_torch.float16 = object()

        with (
            patch("modules.translators.translategemma._resolve_hf_token", return_value="token"),
            patch("modules.translators.translategemma._resolve_device_map", return_value="cuda:0"),
            patch("modules.translators.translategemma.torch", mock_torch),
        ):
            kwargs = translategemma_backend._build_translategemma_model_kwargs()

        self.assertEqual(kwargs["token"], "token")
        self.assertEqual(kwargs["device_map"], "cuda:0")
        self.assertIs(kwargs["dtype"], mock_torch.float16)
        self.assertNotIn("torch_dtype", kwargs)

    def test_resolve_hf_token_empty_returns_none(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(translategemma_backend._resolve_hf_token())

    def test_load_translategemma_tokenizer_falls_back_to_local_files(self):
        auto_tokenizer = MagicMock()
        auto_tokenizer.from_pretrained.side_effect = [OSError("net"), "tokenizer"]

        with patch("modules.translators.translategemma._resolve_hf_token", return_value="secret"):
            loaded = translategemma_backend._load_translategemma_tokenizer(auto_tokenizer)

        self.assertEqual(loaded, "tokenizer")
        self.assertEqual(auto_tokenizer.from_pretrained.call_count, 2)
        self.assertEqual(auto_tokenizer.from_pretrained.call_args_list[1].kwargs["local_files_only"], True)
        self.assertEqual(auto_tokenizer.from_pretrained.call_args_list[1].kwargs["token"], "secret")

    def test_load_translategemma_model_falls_back_to_local_files(self):
        auto_model = MagicMock()
        auto_model.from_pretrained.side_effect = [OSError("net"), "model"]

        with patch(
            "modules.translators.translategemma._build_translategemma_model_kwargs",
            return_value={"device_map": "cuda:0"},
        ):
            loaded = translategemma_backend._load_translategemma_model(auto_model)

        self.assertEqual(loaded, "model")
        self.assertEqual(auto_model.from_pretrained.call_count, 2)
        self.assertEqual(auto_model.from_pretrained.call_args_list[1].kwargs["local_files_only"], True)

    def test_resolve_generation_device_from_parameters_fallback(self):
        parameter = MagicMock()
        parameter.device = "cpu"
        model = MagicMock()
        model.device = None
        model.parameters.return_value = iter([parameter])

        resolved = translategemma_backend._resolve_generation_device(model)

        self.assertEqual(resolved, "cpu")

    def test_resolve_generation_device_returns_none_on_invalid_model(self):
        model = MagicMock()
        model.device = None
        model.parameters.side_effect = TypeError("bad")

        self.assertIsNone(translategemma_backend._resolve_generation_device(model))

    def test_resolve_runtime_settings_invalid_uses_default(self):
        settings = translategemma_backend._resolve_runtime_settings({"max_new_tokens": "invalid"})
        self.assertEqual(settings["max_new_tokens"], 160)

    def test_resolve_prompt_lengths_without_attention_mask_uses_input_ids(self):
        encoded = {
            "input_ids": [[1, 2], [3, 4, 5]],
        }

        lengths = translategemma_backend._resolve_prompt_lengths(encoded)

        self.assertEqual(lengths, [2, 3])

    def test_inference_context_without_torch_inference_mode_returns_nullcontext(self):
        mock_torch = MagicMock()
        del mock_torch.inference_mode

        with patch("modules.translators.translategemma.torch", mock_torch):
            context = translategemma_backend._inference_context()

        self.assertEqual(type(context).__name__, "nullcontext")

    def test_separator_model_wrapper(self):
        separator_instance = MagicMock()
        separator_module = MagicMock()
        separator_module.Separator.return_value = separator_instance

        with patch("modules.models._import_module", return_value=separator_module):
            wrapper = models.SeparatorModel(output_dir="out")
            separator_module.Separator.assert_called_once_with(output_dir="out", output_single_stem="Vocals")
            separator_instance.load_model.assert_called_once_with(model_filename=models.config.AUDIO_SEPARATOR_MODEL_ID)
            wrapper.separate("a.wav")
            separator_instance.separate.assert_called_once_with("a.wav")
            wrapper.release()
            self.assertIsNone(wrapper._separator)

    def test_model_manager_lazy_and_offload(self):
        manager = models.ModelManager()
        whisper_wrapper = MagicMock()
        nllb_wrapper = MagicMock()
        translategemma_wrapper = MagicMock()
        separator_wrapper = MagicMock()
        with (
            patch("modules.models.WhisperModel", return_value=whisper_wrapper) as mock_whisper,
            patch("modules.models.nllb_backend.NLLBTranslator", return_value=nllb_wrapper) as mock_nllb,
            patch(
                "modules.models.translategemma_backend.TranslateGemmaTranslator",
                return_value=translategemma_wrapper,
            ) as mock_translategemma,
            patch("modules.models.SeparatorModel", return_value=separator_wrapper) as mock_sep,
            patch("modules.models._cleanup_torch_cache") as mock_cleanup,
        ):
            self.assertIs(manager.get_whisper(), manager.get_whisper())
            self.assertIs(manager.get_nllb(), manager.get_nllb())
            self.assertIs(manager.get_translategemma(), manager.get_translategemma())
            self.assertIs(manager.get_separator("out"), manager.get_separator("out"))
            mock_whisper.assert_called_once()
            mock_nllb.assert_called_once()
            mock_translategemma.assert_called_once()
            mock_sep.assert_called_once_with(output_dir="out")

            manager.offload_whisper()
            manager.offload_nllb()
            manager.offload_translategemma()
            manager.offload_separator()
            self.assertEqual(mock_cleanup.call_count, 4)
            whisper_wrapper.release.assert_called_once()
            nllb_wrapper.release.assert_called_once()
            translategemma_wrapper.release.assert_called_once()
            separator_wrapper.release.assert_called_once()

    def test_cleanup_torch_cache_no_torch(self):
        with patch("modules.models.torch", None), patch("modules.models.gc.collect") as mock_collect:
            models._cleanup_torch_cache()
        mock_collect.assert_called_once()

    def test_cleanup_torch_cache_with_cuda(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch("modules.models.torch", mock_torch), patch("modules.models.gc.collect"):
            models._cleanup_torch_cache()
        mock_torch.cuda.empty_cache.assert_called_once()

    def test_resolve_device_map(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch("modules.translators.common.torch", mock_torch):
            self.assertEqual(nllb_backend._resolve_device_map(), "cuda:0")
        with patch("modules.translators.common.torch", None):
            self.assertIsNone(nllb_backend._resolve_device_map())

    def test_nllb_uses_mps_for_model_and_encoded_inputs(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        encoded_value = MagicMock()

        with patch("modules.translators.nllb.torch", mock_torch), patch("modules.translators.common.torch", mock_torch):
            self.assertEqual(nllb_backend._build_nllb_model_kwargs()["device_map"], "mps")
            self.assertEqual(nllb_backend._resolve_nllb_execution_device(MagicMock(device="cpu")), "cpu")
            nllb_backend._move_nllb_encoded_to_device({"input_ids": encoded_value}, None)

        encoded_value.to.assert_called_once_with("mps")

    def test_load_nllb_model_corrupt_cache_recovery(self):
        auto_model = MagicMock()
        auto_model.from_pretrained.side_effect = [
            RuntimeError("invalid safetensors header"),
            "recovered_model",
        ]

        with patch("modules.translators.common.purge_hf_model_cache") as mock_purge:
            loaded = nllb_backend._load_nllb_model(auto_model)

        self.assertEqual(loaded, "recovered_model")
        mock_purge.assert_called_once()
        self.assertEqual(auto_model.from_pretrained.call_count, 2)

    def test_load_nllb_tokenizer_corrupt_cache_recovery(self):
        tokenizer_cls = MagicMock()
        tokenizer_cls.from_pretrained.side_effect = [
            ValueError("piece size is not valid"),
            "recovered_tok",
        ]

        with patch("modules.translators.common.purge_hf_model_cache") as mock_purge:
            loaded = nllb_backend._load_nllb_tokenizer(tokenizer_cls)

        self.assertEqual(loaded, "recovered_tok")
        mock_purge.assert_called_once()
        self.assertEqual(tokenizer_cls.from_pretrained.call_count, 2)

    def test_load_translategemma_model_corrupt_cache_recovery(self):
        auto_model = MagicMock()
        auto_model.from_pretrained.side_effect = [
            RuntimeError("file is not a valid safetensors archive"),
            "recovered_gemma",
        ]

        with patch("modules.translators.common.purge_hf_model_cache") as mock_purge:
            loaded = translategemma_backend._load_translategemma_model(auto_model)

        self.assertEqual(loaded, "recovered_gemma")
        mock_purge.assert_called_once()
        self.assertEqual(auto_model.from_pretrained.call_count, 2)

    def test_load_translategemma_tokenizer_corrupt_cache_recovery(self):
        auto_tokenizer = MagicMock()
        auto_tokenizer.from_pretrained.side_effect = [
            RuntimeError("corrupt or incomplete"),
            "recovered_tok",
        ]

        with patch("modules.translators.common.purge_hf_model_cache") as mock_purge:
            loaded = translategemma_backend._load_translategemma_tokenizer(auto_tokenizer)

        self.assertEqual(loaded, "recovered_tok")
        mock_purge.assert_called_once()
        self.assertEqual(auto_tokenizer.from_pretrained.call_count, 2)

    def test_is_corrupt_checkpoint_error(self):
        self.assertTrue(models._is_corrupt_checkpoint_error(RuntimeError("PytorchStreamReader failed reading zip archive")))
        self.assertTrue(models._is_corrupt_checkpoint_error(RuntimeError("failed finding central directory")))
        self.assertFalse(models._is_corrupt_checkpoint_error(RuntimeError("out of memory")))

    def test_purge_cached_separator_checkpoint(self):
        fake_files = ["test_model.ckpt", "test_model.yaml", "other.ckpt"]
        with (
            patch("os.path.isdir", side_effect=lambda d: d == "/fake/dir"),
            patch("os.listdir", return_value=fake_files),
            patch("os.remove") as mock_remove,
        ):
            models._purge_cached_separator_checkpoint("test_model.ckpt", "/fake/dir")
            self.assertEqual(mock_remove.call_count, 2)

    def test_separator_model_retries_on_corrupt_checkpoint(self):
        fake_module = MagicMock()
        mock_sep_instance = MagicMock()
        mock_sep_instance.model_file_dir = "/tmp/fake-models"
        mock_sep_instance.load_model.side_effect = [
            RuntimeError("PytorchStreamReader failed reading zip archive: failed finding central directory"),
            None,
        ]
        fake_module.Separator.return_value = mock_sep_instance

        with (
            patch("modules.models._import_module", return_value=fake_module),
            patch("modules.models._purge_cached_separator_checkpoint") as mock_purge,
        ):
            wrapper = models.SeparatorModel(output_dir="/fake/out")

        self.assertEqual(mock_sep_instance.load_model.call_count, 2)
        mock_purge.assert_called_once()
        self.assertIs(wrapper._separator, mock_sep_instance)

    def test_purge_whisper_model_cache(self):
        with patch("os.path.isdir", return_value=True), patch("shutil.rmtree") as mock_rmtree:
            models._purge_whisper_model_cache("large-v3")
            mock_rmtree.assert_called_once()

    def test_whisper_model_retries_on_corrupt_checkpoint(self):
        fake_module = MagicMock()
        mock_whisper_inst = MagicMock()
        fake_module.WhisperModel.side_effect = [
            RuntimeError("invalid safetensors header or bad zip file"),
            mock_whisper_inst,
        ]

        with (
            patch("modules.models._import_module", return_value=fake_module),
            patch("modules.models._purge_whisper_model_cache") as mock_purge,
            patch("modules.models._should_try_cuda_whisper", return_value=False),
        ):
            wrapper = models.WhisperModel()

        self.assertEqual(fake_module.WhisperModel.call_count, 2)
        mock_purge.assert_called_once()
        self.assertIs(wrapper._model, mock_whisper_inst)


if __name__ == "__main__":
    unittest.main()
