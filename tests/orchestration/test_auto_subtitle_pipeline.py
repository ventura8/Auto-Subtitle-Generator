import builtins
import os
import unittest
from unittest.mock import MagicMock, mock_open, patch


class TestCoverageAutoSubtitle(unittest.TestCase):
    def setUp(self):
        global auto_subtitle
        import auto_subtitle

        # Reset torch handle per test while restoring original value on cleanup.
        torch_patcher = patch("auto_subtitle.torch", None, create=True)
        torch_patcher.start()
        self.addCleanup(torch_patcher.stop)

    @patch("auto_subtitle.print_progress_bar")
    @patch("auto_subtitle.log")
    @patch("sys.exit")
    def test_init_torch_fail(self, mock_exit, mock_log, mock_bar):
        with patch("auto_subtitle.importlib.import_module", side_effect=ImportError("No torch")):
            auto_subtitle._init_torch_and_hardware(1, 6)
            mock_exit.assert_called_with(1)

    @patch("auto_subtitle.print_progress_bar")
    @patch("auto_subtitle.log")
    @patch("sys.exit")
    def test_init_transformers_fail(self, mock_exit, mock_log, mock_bar):
        with patch(
            "auto_subtitle.importlib.import_module",
            side_effect=lambda name: (_ for _ in ()).throw(ImportError("No transformers")) if name == "transformers" else MagicMock(),
        ):
            auto_subtitle._init_nvidia_and_transformers(3, 6)
            mock_exit.assert_called_with(1)

    @patch("auto_subtitle.print_progress_bar")
    @patch("auto_subtitle.log")
    @patch("sys.exit")
    def test_init_whisper_fail(self, mock_exit, mock_log, mock_bar):
        with patch(
            "auto_subtitle.importlib.import_module",
            side_effect=lambda name: (_ for _ in ()).throw(ImportError("No whisper")) if name == "faster_whisper" else MagicMock(),
        ):
            auto_subtitle._init_whisper_and_separator(4, 6)
            mock_exit.assert_called_with(1)

    @patch("auto_subtitle.print_progress_bar")
    @patch("auto_subtitle.log")
    def test_init_separator_skip(self, mock_log, mock_bar):
        def import_side_effect(name):
            if name == "audio_separator.separator":
                raise ImportError("No separator")
            return MagicMock()

        with patch(
            "auto_subtitle.importlib.import_module",
            side_effect=import_side_effect,
        ):
            auto_subtitle._init_whisper_and_separator(5, 6)
            mock_log.assert_called()

    def test_init_ai_engine_already_init(self):
        auto_subtitle.torch = MagicMock()
        with patch("builtins.print") as mock_print:
            auto_subtitle.init_ai_engine()
            mock_print.assert_not_called()

    def test_get_nvidia_bin_lib_paths(self):
        with (
            patch("os.path.exists", return_value=True),
            patch("os.path.isdir", return_value=True),
            patch("os.listdir", return_value=["item1"]),
        ):
            paths = auto_subtitle._get_nvidia_bin_lib_paths("site-packages")
            self.assertTrue(len(paths) > 0)

    @patch("os.add_dll_directory", create=True)
    def test_apply_paths_to_env(self, mock_add):
        with patch("os.environ", {"PATH": ""}):
            auto_subtitle._apply_paths_to_env(["/new/path"])
            self.assertIn("/new/path", os.environ["PATH"])
            # Assuming hasattr(os, 'add_dll_directory') is true on this env
            if hasattr(os, "add_dll_directory"):
                mock_add.assert_called_with("/new/path")

    def test_load_nvidia_paths_torch_fail(self):
        with (
            patch("site.getsitepackages", return_value=[]),
            patch("auto_subtitle.importlib.import_module", side_effect=ImportError("onnxruntime unavailable")),
        ):
            auto_subtitle.load_nvidia_paths()

    def test_check_resume_empty_srt(self):
        with (
            patch("os.path.exists", return_value=True),
            patch("auto_subtitle.utils.parse_srt", return_value=[]),
            patch("auto_subtitle.log"),
        ):
            res = auto_subtitle._check_resume("folder", "base", "en")
            self.assertEqual(res, (None, None, None))

    def test_embed_subtitles_empty(self):
        self.assertIsNone(auto_subtitle.embed_subtitles("vid.mp4", []))

    @patch("auto_subtitle.utils.get_audio_duration", side_effect=RuntimeError("Error"))
    @patch("auto_subtitle.log")
    @patch("os.path.exists", return_value=True)
    @patch("os.remove")
    def test_embed_subtitles_exception(self, mock_remove, mock_exists, mock_log, mock_dur):
        auto_subtitle.embed_subtitles("vid.mp4", [("s.srt", "en", "English")])
        mock_log.assert_called()
        mock_remove.assert_called()

    @patch("auto_subtitle._obtain_segments", return_value=([], None, None))
    @patch("auto_subtitle.log")
    def test_process_video_no_speech(self, mock_log, mock_ob):
        res = auto_subtitle.process_video("vid.mp4", MagicMock())
        self.assertEqual(res, ([], None, None))
        mock_log.assert_any_call("No speech detected.", "WARNING")

    @patch("auto_subtitle._obtain_segments", return_value=([MagicMock()], "en", "audio.wav"))
    @patch("auto_subtitle.utils.save_srt", side_effect=OSError("Save fail"))
    @patch("auto_subtitle.log")
    @patch("auto_subtitle.translate_segments")
    @patch("auto_subtitle.embed_subtitles")
    def test_process_video_save_srt_error(self, mock_embed, mock_trans, mock_log, mock_save, mock_ob):
        result = auto_subtitle.process_video("vid.mp4", MagicMock())
        self.assertEqual(result, (None, None, None))
        mock_log.assert_any_call("  [Error] Failed to save source SRT: Save fail", "ERROR")
        mock_trans.assert_not_called()
        mock_embed.assert_not_called()

    @patch("auto_subtitle._obtain_segments", return_value=([MagicMock()], "en", "existing.en.srt"))
    @patch("auto_subtitle.translate_segments", side_effect=RuntimeError("Trans fail"))
    @patch("auto_subtitle.log")
    def test_process_video_translation_fail(self, mock_log, mock_trans, mock_ob):
        self.assertEqual(auto_subtitle.process_video("vid.mp4", MagicMock()), (None, None, None))
        mock_log.assert_any_call("Translation failed: Trans fail", "ERROR")

    def test_check_resume_forced_lang_valid(self):
        with patch("os.path.exists", return_value=True), patch("auto_subtitle.utils.parse_srt", return_value=[MagicMock()]):
            segs, lang, srt_path = auto_subtitle._check_resume("folder", "base", "en")
            self.assertEqual(lang, "en")
            self.assertTrue(srt_path.endswith("base.en.srt"))
            self.assertEqual(len(segs), 1)

    def test_check_resume_auto_scan_valid(self):
        def _exists(path):
            return path.endswith("base.source_lang.txt") or path.endswith("base.fr.srt")

        with (
            patch("os.path.exists", side_effect=_exists),
            patch("builtins.open", mock_open(read_data="fr")),
            patch("auto_subtitle.utils.parse_srt", return_value=[MagicMock()]),
        ):
            segs, lang, srt_path = auto_subtitle._check_resume("folder", "base", None)
            self.assertEqual(lang, "fr")
            self.assertTrue(srt_path.endswith("base.fr.srt"))
            self.assertEqual(len(segs), 1)

    def test_check_resume_without_recorded_source_lang_does_not_guess(self):
        with patch("os.path.exists", return_value=False), patch("auto_subtitle.utils.parse_srt") as mock_parse:
            self.assertEqual(auto_subtitle._check_resume("folder", "base", None), (None, None, None))
            mock_parse.assert_not_called()

    def test_check_resume_forced_lang_missing_does_not_fallback(self):
        def _exists(path):
            return path.endswith("base.en.srt")

        with patch("os.path.exists", side_effect=_exists), patch("auto_subtitle.utils.parse_srt") as mock_parse:
            res = auto_subtitle._check_resume("folder", "base", "de")
            self.assertEqual(res, (None, None, None))
            mock_parse.assert_not_called()

    def test_load_nvidia_paths_adds_torch_lib_and_ignores_ort_error(self):
        fake_torch = MagicMock()
        fake_torch.__path__ = ["/fake/torch"]
        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "onnxruntime":
                raise Exception("ort fail")
            return original_import(name, *args, **kwargs)

        with (
            patch("site.getsitepackages", return_value=[]),
            patch("os.path.exists", side_effect=lambda p: p.endswith("/lib")),
            patch.dict("sys.modules", {"torch": fake_torch}),
            patch("auto_subtitle._apply_paths_to_env") as mock_apply,
            patch("builtins.__import__", side_effect=fake_import),
        ):
            auto_subtitle.load_nvidia_paths()
            self.assertTrue(mock_apply.called)

    def test_get_input_files_defaults_to_input_folder(self):
        args = MagicMock(input_path=None, cpu=False, lang=None, prompt=None)
        with (
            patch("argparse.ArgumentParser.parse_args", return_value=args),
            patch("builtins.input", return_value=""),
            patch("os.path.isfile", return_value=False),
            patch("os.path.isdir", return_value=True),
            patch("os.walk", return_value=[]) as mock_walk,
        ):
            files, lang, prompt = auto_subtitle.get_input_files()
            mock_walk.assert_called_once_with("input")
            self.assertEqual(files, [])
            self.assertIsNone(lang)
            self.assertIsNone(prompt)

    @patch("auto_subtitle.embed_subtitles")
    @patch("os.path.exists", return_value=True)
    @patch.dict("auto_subtitle.config.TARGET_LANGUAGES", {"en": {}, "ro": {"label": "Romanian"}}, clear=True)
    def test_finalize_video_processing_uses_fallback_src_label(self, _exists, mock_embed):
        auto_subtitle._finalize_video_processing("video.mp4", ".", "base", "en", "base.en.srt")
        embedded = mock_embed.call_args[0][1]
        self.assertTrue(any(item[2] == "EN" for item in embedded))

    def test_get_input_files_exclude_multilang(self):
        with (
            patch("argparse.ArgumentParser.parse_args", return_value=MagicMock(input_path="folder", cpu=False, lang=None, prompt=None)),
            patch("os.path.isfile", return_value=False),
            patch("os.path.isdir", return_value=True),
            patch("os.walk", return_value=[(".", [], ["vid.mp4", "vid_multilang.mp4"])]),
        ):
            files, _, _ = auto_subtitle.get_input_files()
            self.assertEqual(len(files), 1)

    def test_collect_video_files_file_input_filters_unsupported_or_multilang(self):
        from modules.media import file_utils

        with patch("os.path.isfile", return_value=True), patch("os.path.basename", return_value="clip_multilang.mp4"):
            self.assertEqual(file_utils._collect_video_files("clip_multilang.mp4"), [])

        with patch("os.path.isfile", return_value=True), patch("os.path.basename", return_value="clip.txt"):
            self.assertEqual(file_utils._collect_video_files("clip.txt"), [])

    def test_get_input_files_not_found(self):
        with (
            patch("argparse.ArgumentParser.parse_args", return_value=MagicMock(input_path="ghost", cpu=False, lang=None, prompt=None)),
            patch("os.path.isfile", return_value=False),
            patch("os.path.isdir", return_value=False),
        ):
            with self.assertRaises(FileNotFoundError):
                auto_subtitle.get_input_files()

    def test_format_total_processing_speed_known_duration(self):
        from modules.subtitles import timestamp_utils

        summary = timestamp_utils._format_total_processing_speed(120.0, 60.0)
        self.assertIn("2.00x realtime", summary)

    def test_process_video_batch_logs_total_speed(self):
        with (
            patch("auto_subtitle.process_video", return_value=([], None, None)),
            patch("modules.runtime.batch_summary.get_audio_duration", return_value=120.0),
            patch("auto_subtitle.time.time", side_effect=[10.0, 70.0]),
            patch("auto_subtitle.log") as mock_log,
        ):
            auto_subtitle.process_video_batch(["clip.mp4"], MagicMock(), None, None)

            self.assertTrue(any("Total processing speed: 2.00x realtime" in str(call.args[0]) for call in mock_log.call_args_list))
            self.assertTrue(any("Media duration: 00:02:00" in str(call.args[0]) for call in mock_log.call_args_list))
            self.assertTrue(any("Elapsed: 00:01:00" in str(call.args[0]) for call in mock_log.call_args_list))

    def test_process_video_batch_logs_batch_summary_for_multiple_files(self):
        with (
            patch(
                "auto_subtitle.process_video",
                side_effect=[([MagicMock()], "en", "out1.mp4"), (None, None, None)],
            ),
            patch("modules.runtime.batch_summary.get_audio_duration", side_effect=[100.0, 200.0]),
            patch("auto_subtitle.time.time", side_effect=[0.0, 1.0, 11.0, 20.0, 50.0, 70.0]),
            patch("auto_subtitle.log"),
            patch("modules.runtime.batch_summary.log") as mock_utils_log,
        ):
            auto_subtitle.process_video_batch(["clip1.mp4", "clip2.mp4"], MagicMock(), None, None)

            self._assert_batch_summary_log_lines(mock_utils_log)

    def _assert_batch_summary_log_lines(self, mock_utils_log):
        """Assert expected batch summary and per-file status log entries."""
        logged_lines = [str(call.args[0]) for call in mock_utils_log.call_args_list if call.args]
        self._assert_expected_fragments(logged_lines)
        self._assert_expected_file_statuses(logged_lines)

    def _assert_expected_fragments(self, logged_lines):
        """Assert baseline batch summary fragments are logged."""
        expected_fragments = [
            "[Batch Summary] Files: 2",
            "Succeeded: 1",
            "Failed: 1",
            "Media duration: 00:05:00",
            "[Batch Files]",
        ]
        for fragment in expected_fragments:
            self.assertTrue(any(fragment in line for line in logged_lines))

    def _assert_expected_file_statuses(self, logged_lines):
        """Assert per-file succeeded/failed lines exist in summary output."""
        self.assertTrue(any("clip1.mp4" in line and "Status: succeeded" in line for line in logged_lines))
        self.assertTrue(any("clip2.mp4" in line and "Status: failed" in line for line in logged_lines))


if __name__ == "__main__":
    unittest.main()
