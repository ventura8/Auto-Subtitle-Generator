import contextlib
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch


def _fake_work_dir(folder, base_name):
    """Return the work-directory path without touching the filesystem."""
    return os.path.join(folder, f"{base_name}.asg-temp")


class TestCoverageAutoSubtitle(unittest.TestCase):
    def setUp(self):
        global auto_subtitle
        import auto_subtitle

        # These tests use cwd-relative fake videos; never create real work directories.
        work_dir_patcher = patch("modules.workdir.ensure_work_dir", side_effect=_fake_work_dir)
        work_dir_patcher.start()
        self.addCleanup(work_dir_patcher.stop)
        purge_patcher = patch("modules.workdir.purge_work_dir", return_value=True)
        purge_patcher.start()
        self.addCleanup(purge_patcher.stop)

        # Reset torch handle per test while restoring original value on cleanup.
        torch_patcher = patch("auto_subtitle.torch", None, create=True)
        torch_patcher.start()
        self.addCleanup(torch_patcher.stop)

        # process_video binds the input to an open descriptor; these tests use paths that
        # do not exist on disk, so stub the binding (input_binding has its own tests).
        bind_patcher = patch("auto_subtitle.bind_input", lambda path, strict=True: contextlib.nullcontext())
        bind_patcher.start()
        self.addCleanup(bind_patcher.stop)

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

    def test_init_nvidia_paths_uses_runtime_module(self):
        with patch("auto_subtitle.nvidia_paths.load_nvidia_paths") as mock_load:
            auto_subtitle._init_nvidia_and_transformers(2, 6)
        mock_load.assert_called_once_with(None)

    def test_get_nvidia_bin_lib_paths(self):
        from modules.runtime import nvidia_paths

        with (
            patch("os.path.exists", return_value=True),
            patch("os.path.isdir", return_value=True),
            patch("os.listdir", return_value=["item1"]),
        ):
            paths = nvidia_paths._get_nvidia_bin_lib_paths("site-packages")
            expected_paths = [
                os.path.join("site-packages", "nvidia", "item1", "bin"),
                os.path.join("site-packages", "nvidia", "item1", "lib"),
            ]
            self.assertEqual(paths, expected_paths)

    def test_apply_paths_to_env_with_dll_directory(self):
        from modules.runtime import nvidia_paths

        mock_add = MagicMock()
        with (
            patch("os.environ", {"PATH": ""}),
            patch.object(os, "add_dll_directory", mock_add, create=True),
        ):
            nvidia_paths._apply_paths_to_env(["/new/path"])
            self.assertIn("/new/path", os.environ["PATH"])
            mock_add.assert_called_with("/new/path")

    def test_apply_paths_to_env_without_dll_directory(self):
        from modules.runtime import nvidia_paths

        with (
            patch("os.environ", {"PATH": ""}),
            patch.object(nvidia_paths, "_add_dll_directory_if_supported") as mock_add_helper,
        ):
            nvidia_paths._apply_paths_to_env(["/new/path"])
            self.assertIn("/new/path", os.environ["PATH"])
            mock_add_helper.assert_called_once_with("/new/path")

    def test_main_startup_order_preserves_arg_parsing_before_ai_init(self):
        with (
            patch("sys.argv", ["auto_subtitle.py"]),
            patch("auto_subtitle.models.OPTIMIZER.detect_hardware"),
            patch("auto_subtitle.setup_environment") as mock_setup,
            patch("auto_subtitle.get_input_files", return_value=(["video.mp4"], None, None)) as mock_get_files,
            patch("auto_subtitle.init_ai_engine") as mock_init_ai,
            patch("auto_subtitle.utils.print_banner") as mock_banner,
            patch("auto_subtitle.ModelManager"),
            patch("auto_subtitle.process_video_batch"),
            patch("auto_subtitle.log"),
        ):
            order = []
            mock_setup.side_effect = lambda: order.append("setup_environment")
            mock_get_files.side_effect = lambda *_: (order.append("get_input_files"), (["video.mp4"], None, None))[1]
            mock_init_ai.side_effect = lambda: order.append("init_ai_engine")
            mock_banner.side_effect = lambda *_: order.append("print_banner")

            auto_subtitle.main()

            # The banner is a startup banner: it must precede input handling so it
            # still appears when no videos are found. Heavy engine initialization
            # stays behind the input checks.
            self.assertEqual(order, ["setup_environment", "print_banner", "get_input_files", "init_ai_engine"])
            self.assertEqual(mock_init_ai.call_count, 1)
            self.assertEqual(mock_banner.call_count, 1)

    def test_load_nvidia_paths_torch_fail(self):
        from modules.runtime import nvidia_paths

        with (
            patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}, clear=False),
            patch("modules.runtime.nvidia_paths._load_optional_torch", return_value=None) as load_torch,
            patch.object(nvidia_paths.importlib, "import_module", return_value=MagicMock()),
            patch.object(nvidia_paths, "_apply_paths_to_env") as mock_apply,
        ):
            nvidia_paths.load_nvidia_paths()
            load_torch.assert_called_once_with()
            mock_apply.assert_called_once_with([])

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
    @patch("auto_subtitle.reserve_temp_path", return_value=SimpleNamespace(path=".temp_output.vid.scratch.mp4"))
    @patch("auto_subtitle.discard_temp_path")
    def test_embed_subtitles_exception(self, mock_discard, mock_reserve, mock_log, mock_dur):
        self.assertIsNone(auto_subtitle.embed_subtitles("vid.mp4", [("s.srt", "en", "English")]))
        mock_log.assert_called()
        mock_discard.assert_called_once_with(mock_reserve.return_value)

    @patch("auto_subtitle.log")
    @patch("auto_subtitle.reserve_temp_path", side_effect=OSError("Read-only file system"))
    @patch("auto_subtitle.discard_temp_path")
    @patch("auto_subtitle.utils.run_ffmpeg_progress")
    def test_embed_subtitles_reservation_failure_returns_none(self, mock_run, mock_discard, mock_reserve, mock_log):
        self.assertIsNone(auto_subtitle.embed_subtitles("vid.mp4", [("s.srt", "en", "English")]))
        mock_log.assert_any_call("Embedding failed: Read-only file system", "ERROR")
        mock_run.assert_not_called()
        mock_discard.assert_not_called()

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

    def test_check_resume_without_recorded_source_lang_does_not_guess_when_no_srts(self):
        with (
            patch("os.path.exists", return_value=False),
            patch("os.listdir", return_value=[]),
            patch("auto_subtitle.utils.parse_srt") as mock_parse,
        ):
            self.assertEqual(auto_subtitle._check_resume("folder", "base", None), (None, None, None))
            mock_parse.assert_not_called()

    def test_check_resume_scans_existing_srt_without_sidecar(self):
        def _exists(path):
            return path.endswith("base.ro.srt")

        with (
            patch("os.path.exists", side_effect=_exists),
            patch("os.listdir", return_value=["base.ro.srt"]),
            patch("auto_subtitle.utils.parse_srt", return_value=[MagicMock()]),
        ):
            segs, lang, srt_path = auto_subtitle._check_resume("folder", "base", None)
            self.assertEqual(lang, "ro")
            self.assertTrue(srt_path.endswith("base.ro.srt"))
            self.assertEqual(len(segs), 1)

    @patch.dict("auto_subtitle.config.TARGET_LANGUAGES", {"fr": {}, "ro": {}}, clear=True)
    def test_check_resume_ignores_translated_srts_when_recorded_srt_is_unusable(self):
        def _exists(path):
            return path.endswith("base.source_lang.txt") or path.endswith("base.fr.srt") or path.endswith("base.ro.srt")

        def _parse(path):
            return [] if path.endswith("base.fr.srt") else [MagicMock()]

        with (
            patch("os.path.exists", side_effect=_exists),
            patch("builtins.open", mock_open(read_data="fr")),
            patch("auto_subtitle.utils.parse_srt", side_effect=_parse),
            patch("os.listdir", return_value=["base.fr.srt", "base.ro.srt"]),
            patch("auto_subtitle.log"),
        ):
            result = auto_subtitle._check_resume("folder", "base", None)

        self.assertEqual(result, (None, None, None))

    @patch.dict("auto_subtitle.config.TARGET_LANGUAGES", {"fr": {}, "ro": {}}, clear=True)
    def test_check_resume_ignores_translated_srts_when_source_is_unusable(self):
        def _exists(path):
            return path.endswith("base.source_lang.txt")

        with (
            patch("os.path.exists", side_effect=_exists),
            patch("builtins.open", mock_open(read_data="en")),
            patch("os.listdir", return_value=["base.fr.srt", "base.ro.srt"]),
            patch("auto_subtitle.utils.parse_srt") as mock_parse,
        ):
            self.assertEqual(auto_subtitle._check_resume("folder", "base", None), (None, None, None))
            mock_parse.assert_not_called()

    def test_check_resume_ignores_undetermined_language(self):
        def _exists(path):
            return path.endswith("base.source_lang.txt") or path.endswith("base.und.srt")

        with (
            patch("os.path.exists", side_effect=_exists),
            patch("builtins.open", mock_open(read_data="undetermined")),
            patch("os.listdir", return_value=["base.und.srt"]),
            patch("auto_subtitle.utils.parse_srt") as mock_parse,
        ):
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
        from modules.runtime import nvidia_paths

        fake_torch = MagicMock()
        fake_torch.__path__ = ["/fake/torch"]

        with (
            patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}, clear=False),
            patch("site.getsitepackages", return_value=[]),
            patch("os.path.exists", side_effect=lambda p: p.endswith("/lib")),
            patch.dict("sys.modules", {"torch": fake_torch}),
            patch("modules.runtime.nvidia_paths._apply_paths_to_env") as mock_apply,
            patch(
                "modules.runtime.nvidia_paths.importlib.import_module",
                side_effect=ImportError("onnxruntime not found"),
            ),
        ):
            nvidia_paths.load_nvidia_paths(fake_torch)
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
            mock_walk.assert_called_once_with("input", followlinks=False)
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
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            for name in ("vid.mp4", "vid_multilang.mp4"):
                open(os.path.join(folder, name), "wb").close()
            with patch("argparse.ArgumentParser.parse_args", return_value=MagicMock(input_path=folder, cpu=False, lang=None, prompt=None)):
                files, _, _ = auto_subtitle.get_input_files()
            self.assertEqual([os.path.basename(f) for f in files], ["vid.mp4"])

    @staticmethod
    def _symlink_or_skip(target, link):
        try:
            os.symlink(target, link)
        except (OSError, NotImplementedError) as e:
            raise unittest.SkipTest(f"symlinks unavailable: {e}")

    def test_collect_video_files_skips_planted_symlinks(self):
        import tempfile

        from modules.media import file_utils

        with tempfile.TemporaryDirectory() as victim, tempfile.TemporaryDirectory() as folder:
            secret = os.path.join(victim, "secret.mp4")
            open(secret, "wb").close()
            open(os.path.join(folder, "real.mp4"), "wb").close()
            self._symlink_or_skip(secret, os.path.join(folder, "vacation.mp4"))
            # Symlinked directory containing supported media must not be walked.
            self._symlink_or_skip(victim, os.path.join(folder, "clips"))
            # Link pointing back inside the folder is still refused: only regular files count.
            self._symlink_or_skip(os.path.join(folder, "real.mp4"), os.path.join(folder, "alias.mp4"))

            with patch("modules.media.file_utils.log"):
                files = file_utils._collect_video_files(folder)
            self.assertEqual([os.path.basename(f) for f in files], ["real.mp4"])

            # A top-level symlink dropped onto the prompt is refused outright.
            with patch("modules.media.file_utils.log") as mock_log:
                self.assertEqual(file_utils._collect_video_files(os.path.join(folder, "vacation.mp4")), [])
            self.assertTrue(any("symlink" in str(c.args[0]).lower() for c in mock_log.call_args_list))

    def test_collect_video_files_rejects_real_path_outside_input_root(self):
        from modules.media import file_utils

        with (
            patch("os.lstat", return_value=SimpleNamespace(st_mode=0o100644)),
            patch("os.path.realpath", side_effect=lambda p: "/victim/secret.mp4" if p.endswith(".mp4") else "/drop"),
            patch("modules.media.file_utils.log"),
        ):
            self.assertFalse(file_utils._is_safe_input_file("/drop/vacation.mp4", "/drop"))

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
