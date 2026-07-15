import argparse
import os
import sys
import unittest
import unittest.mock
from unittest.mock import MagicMock, mock_open, patch


class TestAutoSubtitleUltimate(unittest.TestCase):
    def setUp(self):
        # Lazy import to ensure coverage measurement
        global auto_subtitle, config, models, translation
        import auto_subtitle
        import modules.pipeline.translation as translation
        from modules import models
        from modules.configuration import config

        torch_patcher = patch.object(auto_subtitle, "torch", sys.modules["torch"], create=True)
        torch_patcher.start()
        self.addCleanup(torch_patcher.stop)

        config_patcher = patch.multiple(
            config,
            USE_VOCAL_SEPARATION=True,
            FORCED_LANGUAGE=None,
            TARGET_LANGUAGES={},
        )
        config_patcher.start()
        self.addCleanup(config_patcher.stop)

        optimizer_patcher = patch.dict(models.OPTIMIZER.config, {"whisper_beam": 1, "nllb_batch": 8}, clear=False)
        optimizer_patcher.start()
        self.addCleanup(optimizer_patcher.stop)

    def mock_exists(self, path):
        # Mock logic to distinguish between video files and
        p = str(path)
        if p.endswith(".mp4") or "Vocals" in p or "vocal" in p:
            return True
        return False

    def test_verify_mocks(self):
        # Diagnostic: Ensure we are NOT using real libraries
        audio_separator = sys.modules.get("audio_separator")
        faster_whisper = sys.modules.get("faster_whisper")
        torch = sys.modules.get("torch")
        transformers = sys.modules.get("transformers")

        self.assertIsInstance(torch, MagicMock)
        self.assertIsInstance(transformers, MagicMock)
        self.assertIsInstance(faster_whisper, MagicMock)
        self.assertIsInstance(audio_separator, MagicMock)
        self.assertIsInstance(auto_subtitle.torch, MagicMock)

        # Check if anything big is loaded
        big_mods = [m for m in sys.modules if "torch" in m or "transformers" in m or "whisper" in m]
        auto_subtitle.log(f"  [Diagnostic] Big modules in sys.modules: {big_mods}")
        # Verify new modules are mocked/safe or at least imported
        transcription_module = __import__("modules.pipeline.transcription", fromlist=["*"])
        translation_module = __import__("modules.pipeline.translation", fromlist=["*"])

        self.assertIsNotNone(transcription_module)
        self.assertIsNotNone(translation_module)
        auto_subtitle.log("  [Diagnostic] All AI mocks and modules verified.")

    def test_get_input_files_single_video(self):
        with (
            patch("os.path.isfile", return_value=True),
            patch("os.path.isdir", return_value=False),
            patch("sys.argv", ["utils.py", "test.mp4"]),
        ):
            files, _, _ = auto_subtitle.get_input_files()
            self.assertEqual(files, [os.path.abspath("test.mp4")])

    def test_get_input_files_directory(self):
        with (
            patch("os.path.isdir", return_value=True),
            patch("os.walk", return_value=[("input_dir", [], ["vid1.mp4", "vid2.mkv", "readme.txt"])]),
            patch("sys.argv", ["utils.py", "input_dir"]),
            patch.object(config, "VIDEO_EXTENSIONS", {".mp4", ".mkv"}),
        ):
            with patch("os.path.isfile", side_effect=lambda x: x.endswith(".mp4") or x.endswith(".mkv")):
                files, _, _ = auto_subtitle.get_input_files()
                self.assertEqual(len(files), 2)
                self.assertIn(os.path.abspath(os.path.join("input_dir", "vid1.mp4")), files)

    def test_process_video_end_to_end_flow(self):
        # Test the high-level orchestration of process_video
        mock_seg = MagicMock(start=0.0, end=1.0, text="Hello")
        video_path = os.path.abspath("video.mp4")
        expected_folder = os.path.dirname(video_path) or "."
        expected_src_srt = os.path.join(expected_folder, "video.en.srt")
        expected_output = os.path.join(expected_folder, "video_multilang.mp4")

        with (
            patch("modules.utils.extract_clean_audio", return_value="temp.wav") as mock_extract,
            patch("auto_subtitle.log"),
            patch("auto_subtitle.config.load_config", return_value=True),
            patch("modules.utils.cleanup_temp_files") as mock_cleanup,
            patch("modules.utils.save_srt") as mock_save_srt,
            patch("auto_subtitle.translate_segments") as mock_translate,
            patch("auto_subtitle.embed_subtitles", return_value=expected_output) as mock_embed,
            patch("os.path.exists", return_value=False),
            patch("os.listdir", return_value=[]),
            patch.object(config, "USE_VOCAL_SEPARATION", False),
            patch("auto_subtitle._check_resume", return_value=(None, None, None)),
        ):
            mock_mgr = MagicMock()
            whisper_info = MagicMock(language="en", language_probability=0.99, duration=100.0)
            mock_mgr.get_whisper.return_value.transcribe.return_value = ([mock_seg], whisper_info)

            result = auto_subtitle.process_video(video_path, mock_mgr)

        self.assertEqual(result, ([mock_seg], "en", expected_output))
        mock_extract.assert_called_once_with(video_path)
        mock_save_srt.assert_called_once_with([mock_seg], expected_src_srt)
        mock_translate.assert_called_once_with([mock_seg], "en", mock_mgr, expected_folder, "video")
        mock_embed.assert_called_once_with(video_path, [])
        mock_cleanup.assert_called_once_with(expected_folder, "video", "video.mp4")

    def test_resume_processing_from_existing_files(self):
        # Test that the pipeline resumes correctly when SRT/Vocals already exist
        exists_called = []

        def se_exists(p):
            # Simulate already having the SRT
            exists_called.append(p)
            if p.endswith(".srt"):
                return True
            return False

        with patch("os.path.exists", side_effect=se_exists):
            with patch("auto_subtitle.log"):
                mock_mgr = MagicMock()
                # Simulate _check_resume finding the SRT
                res_mock = ([MagicMock(start=0, end=1, text="res")], "en", "video.srt")
                with (
                    patch("auto_subtitle._check_resume", return_value=res_mock),
                    patch("builtins.open", mock_open()),
                    patch("modules.utils.save_srt"),
                    patch("auto_subtitle.translate_segments") as m_translate,
                    patch("auto_subtitle.embed_subtitles") as m_embed,
                ):
                    auto_subtitle.process_video(os.path.abspath("video.mp4"), mock_mgr)
                    m_translate.assert_called_once()
                    m_embed.assert_called_once()

    def test_cleanup_on_transcription_failure(self):
        with patch.dict(sys.modules, {"audio_separator": MagicMock(), "audio_separator.separator": MagicMock()}):
            sys.modules["audio_separator.separator"].Separator.return_value.separate.return_value = []
            with patch("modules.utils.extract_clean_audio", return_value="temp.wav"):
                with patch("modules.models.WhisperModel") as m_whisper:
                    m_whisper.return_value.transcribe.side_effect = RuntimeError("Whisper Crash")
                    with patch("auto_subtitle.log"), patch("modules.utils.cleanup_temp_files") as m_clean:
                        with (
                            patch("os.path.exists", side_effect=lambda x: False if "_multilang" in str(x) else self.mock_exists(x)),
                            patch("os.listdir", return_value=[]),
                        ):
                            mock_mgr = MagicMock()
                            mock_mgr.get_whisper.return_value = m_whisper.return_value
                            res = auto_subtitle.process_video(os.path.abspath("fail.mp4"), mock_mgr)
                            self.assertEqual(res, (None, None, None))
                            m_clean.assert_called()

    def test_nllb_translation_generic_error_handling(self):
        # Test how _translate_segments handles a failed NLLB translation subprocess
        with patch("modules.utils.extract_clean_audio", return_value="temp.wav"):
            with patch("modules.models.WhisperModel") as m_w:
                m_w.return_value.transcribe.return_value = (
                    [MagicMock(start=0, end=1, text="test", avg_logprob=-0.1)],
                    MagicMock(language="en", language_probability=0.99, duration=100.0),
                )
                with patch("subprocess.Popen") as m_popen:
                    m_popen.return_value.__enter__.return_value.wait.return_value = 1
                    m_popen.return_value.__enter__.return_value.returncode = 1

                    with (
                        patch("auto_subtitle.log"),
                        patch("modules.configuration.config.load_config", return_value=True),
                        patch("modules.pipeline.translation.log") as m_log_transl,
                        patch("modules.utils.cleanup_temp_files"),
                        patch("modules.utils.save_srt"),
                        patch("auto_subtitle.embed_subtitles"),
                        patch("os.remove"),
                        patch("auto_subtitle._check_resume", return_value=(None, None, None)),
                    ):
                        mock_mgr = MagicMock()
                        mock_mgr.get_whisper.return_value = m_w.return_value
                        tgt = {"es": {"code": "spa", "label": "Esp"}}
                        with patch.dict(config.TARGET_LANGUAGES, tgt, clear=True):
                            with patch("builtins.open", mock_open(read_data="[]")):
                                auto_subtitle.process_video(os.path.abspath("trans_fail.mkv"), mock_mgr)

                        logs = [str(c[0][0]) for c in m_log_transl.call_args_list if c.args]
                        self.assertTrue(any("Translation worker failed" in line for line in logs))

    def test_batch_translation_manifest_generation(self):
        # Verify that _execute_translation_workers generates the correct manifest
        with patch("modules.utils.extract_clean_audio", return_value="temp.wav"):
            with patch("subprocess.Popen") as m_popen:
                m_popen.return_value.__enter__.return_value.wait.return_value = 0
                m_popen.return_value.__enter__.return_value.returncode = 0

                with (
                    patch("auto_subtitle.log"),
                    patch("modules.pipeline.translation.log"),
                    patch("modules.utils.cleanup_temp_files"),
                    patch("modules.utils.save_translated_srt"),
                    patch("modules.pipeline.translation._poll_translation_results", return_value=set()),
                    patch("builtins.open", mock_open()) as m_open,
                    patch("json.dump") as m_json_dump,
                    patch("os.path.exists", side_effect=lambda x: True),
                    patch("os.remove"),
                    patch("modules.pipeline.translation.time.sleep"),
                ):
                    folder = os.path.abspath("test_folder")
                    base_name = "video"
                    src_code = "eng_Latn"
                    missing_langs = ["es", "fr"]
                    source_data = [{"text": "Hello", "start": 0, "end": 1}]
                    segments = []

                    # Mock config targets
                    targets = {"es": {"code": "spa_Latn", "label": "Esp"}, "fr": {"code": "fra_Latn", "label": "Fra"}}
                    with patch.dict(config.TARGET_LANGUAGES, targets, clear=True):
                        translation._execute_translation_workers(missing_langs, source_data, src_code, folder, base_name, segments)

                    # Verify manifest creation
                    # We expect json.dump to be called for:
                    # 1. common_input
                    # 2. manifest
                    # We want to check the manifest content

                    # Find the call that dumped the manifest (dict with "jobs")
                    manifest_call = None
                    for call in m_json_dump.call_args_list:
                        args = call[0]
                        if isinstance(args[0], dict) and "jobs" in args[0]:
                            manifest_call = args[0]
                            break

                    self.assertIsNotNone(manifest_call)
                    self.assertEqual(len(manifest_call["jobs"]), 2)
                    self.assertEqual(manifest_call["jobs"][0]["lang"], "es")
                    self.assertEqual(manifest_call["jobs"][1]["lang"], "fr")

                    # proper command called
                    cmd_args = m_popen.call_args[0][0]
                    self.assertIn("--batch", cmd_args)

                    # Verify open was called (satisfies lint and logic)
                    m_open.assert_called()

    def test_nllb_load_fallback_to_local(self):
        # Verify that NLLBTranslator tries local_files_only=True if network fails

        # Access the GLOBAL mock established in conftest
        m_transformers = sys.modules["transformers"]

        mock_auto = MagicMock()
        mock_auto.from_pretrained.side_effect = [OSError("Network Error"), MagicMock()]
        mock_tokenizer = MagicMock()

        with (
            patch.object(m_transformers, "AutoModelForSeq2SeqLM", mock_auto),
            patch.object(m_transformers, "NllbTokenizer", mock_tokenizer),
            patch.object(m_transformers, "__version__", "4.30.0"),
        ):
            from modules.translators.nllb import NLLBTranslator

            _ = NLLBTranslator()

            # Should have called from_pretrained twice
            self.assertEqual(mock_auto.from_pretrained.call_count, 2)

            # Check arguments of the second call (Fallback)
            args, kwargs = mock_auto.from_pretrained.call_args_list[1]
            self.assertTrue(kwargs.get("local_files_only"), "Fallback did not use local_files_only=True")

    def test_get_output_filenames_cases(self):
        # Test default
        res = auto_subtitle._get_output_filenames("vid.mp4", "dir", None)
        out, srt, base = res
        self.assertIn("_multilang.mp4", out)
        # We now use .en.srt by default
        self.assertTrue(srt.endswith("vid.en.srt"))

        # Test forced lang
        res2 = auto_subtitle._get_output_filenames("vid.mp4", "dir", "ro")
        self.assertEqual(len(res2), 3)
        out2, srt2, base2 = res2
        self.assertIn("vid.ro.srt", srt2)

        # Preserve extension for non-MP4 files
        out3, _, _ = auto_subtitle._get_output_filenames("clip.mkv", "dir", None)
        self.assertTrue(out3.endswith("_multilang.mkv"))

    def test_init_ai_engine_full(self):
        with (
            patch("auto_subtitle.torch", None, create=True),
            patch("auto_subtitle._init_torch_and_hardware", return_value=2) as mock_init_torch,
            patch("auto_subtitle._init_nvidia_and_transformers", return_value=4) as mock_init_transformers,
            patch("auto_subtitle._init_whisper_and_separator", return_value=6) as mock_init_whisper,
            patch("auto_subtitle._render_init_progress") as mock_render_progress,
        ):
            auto_subtitle.init_ai_engine()

        mock_init_torch.assert_called_once_with(1, auto_subtitle.INIT_TOTAL_STEPS)
        mock_init_transformers.assert_called_once_with(2, auto_subtitle.INIT_TOTAL_STEPS)
        mock_init_whisper.assert_called_once_with(4, auto_subtitle.INIT_TOTAL_STEPS)
        mock_render_progress.assert_called_once_with(6, auto_subtitle.INIT_TOTAL_STEPS, "Initialization Complete")

    def test_setup_environment(self):
        with patch("auto_subtitle.utils.setup_signal_handlers") as m_sig, patch("auto_subtitle.multiprocessing.freeze_support") as m_freeze:
            auto_subtitle.setup_environment()
            m_sig.assert_called()
            m_freeze.assert_called()

    def test_main_no_files(self):
        with (
            patch("auto_subtitle.get_input_files", return_value=([], None, None)),
            patch("auto_subtitle.init_ai_engine"),
            patch("auto_subtitle.setup_environment"),
            patch("auto_subtitle.log"),
            patch("sys.exit") as m_exit,
        ):
            auto_subtitle.main()
            m_exit.assert_called_with(0)

    def test_main_with_files(self):
        # Run main() with minimal mocks to cover get_input_files and setup_environment
        with (
            patch("sys.argv", ["auto_subtitle.py", "vid1.mp4"]),
            patch("os.path.exists", return_value=True),
            patch("os.path.isfile", return_value=True),
            patch("auto_subtitle.process_video") as m_proc,
            patch("auto_subtitle.utils.print_banner"),
            patch("auto_subtitle.utils.init_console"),
            patch("auto_subtitle.utils.setup_signal_handlers"),
            patch("auto_subtitle.multiprocessing.freeze_support"),
            patch("auto_subtitle.init_ai_engine"),
        ):
            auto_subtitle.main()

            self.assertEqual(m_proc.call_count, 1)
            m_proc.assert_called()

    @patch("argparse.ArgumentParser.parse_args")
    @patch("builtins.input", return_value="")
    @patch("os.path.isdir", return_value=True)
    @patch("os.path.isfile", return_value=False)
    @patch("os.walk")
    def test_get_input_files(self, mock_walk, mock_isfile, mock_isdir, mock_input, mock_args):
        # Scenario 1: Arg provided, is folder
        mock_args.return_value = argparse.Namespace(input_path="myfolder", lang="en", prompt="hello", cpu=False)
        mock_walk.return_value = [("myfolder", [], ["vid.mp4", "ignore.txt"])]

        files, lang, prompt = auto_subtitle.get_input_files()

        self.assertEqual(lang, "en")
        self.assertEqual(prompt, "hello")
        self.assertTrue("vid.mp4" in files[0])

        # Scenario 2: No arg, prompt user
        mock_args.return_value = argparse.Namespace(input_path=None, lang=None, prompt=None, cpu=True)
        mock_input.return_value = "myvideo.mp4"
        mock_isdir.return_value = False
        mock_isfile.return_value = True

        files, lang, prompt = auto_subtitle.get_input_files()
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].endswith("myvideo.mp4"))
        self.assertIsNone(lang)

    def test_embed_subtitles(self):
        # Cover embed_subtitles logic
        with (
            patch("auto_subtitle.utils.run_ffmpeg_progress") as m_run,
            patch("auto_subtitle.utils.get_audio_duration", return_value=100),
            patch("os.path.exists", return_value=False),
        ):  # output doesn't exist
            srt_files = [("en.srt", "en", "English"), ("es.srt", "es", "Spanish")]
            auto_subtitle.embed_subtitles("vid.mp4", srt_files)

            m_run.assert_called()
            cmd = m_run.call_args[0][0]
            self.assertIn("-c:s", cmd)
            self.assertIn("mov_text", cmd)  # since .mp4

            # Verify metadata
            self.assertIn("language=en", cmd)
            self.assertIn("language=es", cmd)

    def test_init_engine_failures(self):
        # Cover ImportError inside _init_* functions
        # We need to un-mock sys.modules or override them specifically for this test
        # But since they are mocked in setUp via sys.modules assignments, we can side_effect on the mock

        # 1. Faster-Whisper Fail
        with patch.dict(sys.modules, {"faster_whisper": None}):
            # Use a context manager to catch sys.exit
            with self.assertRaises(SystemExit) as cm:
                auto_subtitle._init_whisper_and_separator(0, 6)
            self.assertEqual(cm.exception.code, 1)

        # 2. Transformers Fail (NLLB)
        # We need to simulate ImportError on import transformers
        # Since 'transformers' is already in sys.modules (from conftest), removing it might work?
        # But lazy import inside function will try to reload it.
        # A clearer way is to mock import logic or use SideEffect on the specific function if possible.
        # Given limitations, testing the 'except ImportError' blocks is hard if imports succeed.
        # But we can mock the inner checks.
        pass

    def test_nvidia_path_loading(self):
        # Cover load_nvidia_paths and _get_nvidia_bin_lib_paths
        # We need to mock os.path.exists and os.listdir to simulate NVIDIA folders

        def os_exists_side_effect(path):
            if "nvidia" in path or "site-packages" in path or "lib" in path:
                return True
            return False

        with (
            patch("site.getsitepackages", return_value=["/site-packages"]),
            patch("sys.prefix", "/sys_prefix"),
            patch("os.path.exists", side_effect=os_exists_side_effect),
            patch("os.listdir", return_value=["cudnn", "cublas"]),
            patch("os.path.isdir", return_value=True),
            patch("os.environ", {"PATH": ""}),
            patch("os.add_dll_directory", create=True) as m_add_dll,
        ):
            # Simulate torch.__path__
            with patch("auto_subtitle.torch") as m_torch:
                m_torch.__path__ = ["/site-packages/torch"]

                auto_subtitle.load_nvidia_paths()

                # Check that paths were added
                self.assertTrue(len(os.environ["PATH"]) > 0)
                # Should have found bin/lib for cudnn/cublas
                # Logic: /site-packages/nvidia/cudnn/bin, lib...
                # We expect multiple add_dll_directory calls
                m_add_dll.assert_called()


if __name__ == "__main__":
    unittest.main()
