import unittest
from unittest.mock import MagicMock, mock_open, patch

from modules import utils


class TestCoverageUtils(unittest.TestCase):
    def test_unregister_subprocess_removes_process(self):
        proc = MagicMock()
        with patch("modules.runtime.logging_utils.active_subprocesses", [proc]):
            utils.unregister_subprocess(proc)
            from modules.runtime import logging_utils

            self.assertEqual(logging_utils.active_subprocesses, [])

    def test_handle_shutdown_windows_taskkill_branch(self):
        proc = MagicMock()
        proc.pid = 9999
        proc.poll.return_value = None
        with (
            patch("modules.runtime.logging_utils.active_subprocesses", [proc]),
            patch("sys.platform", "win32"),
            patch("builtins.print"),
            patch("sys.exit"),
        ):
            utils.handle_shutdown(None, None)
            proc.terminate.assert_called_once()

    def test_print_banner_with_optimizer(self):
        mock_opt = MagicMock()
        mock_opt.gpu_name = "TestGPU"
        mock_opt.vram_gb = 16
        mock_opt.profile = "HIGH"
        mock_opt.config = {"nllb_batch": 4, "ffmpeg_threads": 8}

        with (
            patch("builtins.print"),
            patch("platform.system", return_value="Windows"),
            patch("platform.release", return_value="10"),
            patch("modules.utils.get_cpu_name", return_value="TestCPU"),
        ):
            utils.print_banner(mock_opt)

    def test_handle_shutdown_error(self):
        proc = MagicMock()
        proc.poll.return_value = None
        proc.terminate.side_effect = OSError("Kill fail")

        with (
            patch("modules.runtime.logging_utils.active_subprocesses", [proc]),
            patch("sys.platform", "linux"),
            patch("builtins.print"),
            patch("sys.exit") as mock_exit,
        ):
            utils.handle_shutdown(None, None)
            proc.terminate.assert_called()
            mock_exit.assert_called_once_with(1)

    def test_init_console_exception(self):
        kernel32 = MagicMock()
        kernel32.GetStdHandle.side_effect = OSError("Ctypes fail")
        windll_stub = MagicMock()
        windll_stub.kernel32 = kernel32

        with patch("os.name", "nt"), patch("modules.runtime.logging_utils.ctypes.windll", windll_stub, create=True):
            utils.init_console()  # Should just pass

    def test_setup_signal_handlers(self):
        with patch("signal.signal") as mock_sig, patch("sys.platform", "linux"):
            utils.setup_signal_handlers()
            self.assertEqual(mock_sig.call_count, 2)

    def test_setup_signal_handlers_windows(self):
        windll_stub = MagicMock()
        windll_stub.kernel32 = MagicMock()
        with (
            patch("signal.signal"),
            patch("sys.platform", "win32"),
            patch.dict("sys.modules", {"winreg": MagicMock()}),
            patch("ctypes.WINFUNCTYPE", return_value=lambda x: x, create=True),
            patch("modules.runtime.logging_utils.ctypes.windll", windll_stub, create=True),
        ):
            windll_stub.kernel32.SetConsoleCtrlHandler.return_value = True
            utils.setup_signal_handlers()

    def test_setup_signal_handlers_windows_failure_prints_warning(self):
        windll_stub = MagicMock()
        windll_stub.kernel32 = MagicMock()
        with (
            patch("signal.signal"),
            patch("sys.platform", "win32"),
            patch.dict("sys.modules", {"winreg": MagicMock()}),
            patch("ctypes.WINFUNCTYPE", return_value=lambda x: x, create=True),
            patch("modules.runtime.logging_utils.ctypes.windll", windll_stub, create=True),
            patch("builtins.print") as mock_print,
        ):
            windll_stub.kernel32.SetConsoleCtrlHandler.return_value = False
            utils.setup_signal_handlers()
            mock_print.assert_any_call("[Warning] Failed to set Windows Console Handler")

    def test_print_progress_bar_edge_cases(self):
        with patch("sys.stdout.write"), patch("sys.stdout.flush"), patch("shutil.get_terminal_size") as mock_size:
            mock_size.return_value = MagicMock(columns=80)
            # Total 0
            utils.print_progress_bar(0, 0)
            # Invalid inputs
            utils.print_progress_bar("a", "b")
            # Speed/ETA
            utils.print_progress_bar(50, 100, speed=1.0, eta=10)
            # Very long prefix
            utils.print_progress_bar(50, 100, prefix="A" * 100)
            # Unicode error fallback
            with patch("sys.stdout.write", side_effect=[UnicodeEncodeError("utf-8", "", 0, 1, "mock"), None]):
                utils.print_progress_bar(50, 100)

    def test_get_ffmpeg_paths_fallback(self):
        with patch("os.path.exists", return_value=False):
            ffmpeg, ffprobe = utils.get_ffmpeg_paths()
            self.assertEqual(ffmpeg, "ffmpeg")
            self.assertEqual(ffprobe, "ffprobe")

    def test_parse_timestamp_extra(self):
        self.assertEqual(utils.parse_timestamp("00:00:01.500"), 1.5)
        self.assertEqual(utils.parse_timestamp("00:00:01"), 1.0)
        self.assertEqual(utils.parse_timestamp("invalid"), 0.0)
        self.assertEqual(utils.parse_timestamp(None), 0.0)

    def test_process_ffmpeg_line_exception(self):
        with (
            patch("modules.media.ffmpeg_utils.parse_timestamp", side_effect=ValueError("Parse fail")),
            patch("modules.media.ffmpeg_utils.gc.collect") as mock_gc,
        ):
            from modules.media import ffmpeg_utils

            ffmpeg_utils._process_ffmpeg_line("time=00:00:01.00", 0, 100, "Desc")
            mock_gc.assert_called()

    def test_process_ffmpeg_line_no_total_duration(self):
        with (
            patch("modules.media.ffmpeg_utils.parse_timestamp", return_value=1.0),
            patch("modules.media.ffmpeg_utils.print_progress_bar") as mock_bar,
        ):
            from modules.media import ffmpeg_utils

            ffmpeg_utils._process_ffmpeg_line("time=00:00:01.00", 0, 0, "Desc")
            mock_bar.assert_not_called()

    def test_finalize_ffmpeg_progress_raises_called_process_error_on_nonzero_return(self):
        class FakeCalledProcessError(Exception):
            pass

        process = MagicMock(returncode=1)
        process.returncode = 1
        with patch("modules.media.ffmpeg_utils.subprocess.CalledProcessError", FakeCalledProcessError):
            with self.assertRaises(FakeCalledProcessError):
                from modules.media import ffmpeg_utils

                ffmpeg_utils._finalize_ffmpeg_progress(process, ["ffmpeg"], 0.0, 0.0, "desc")

    def test_finalize_ffmpeg_progress_raises_runtime_error_when_called_process_error_invalid(self):
        process = MagicMock(returncode=1)
        process.returncode = 1
        with patch("modules.media.ffmpeg_utils.subprocess.CalledProcessError", "invalid"):
            with self.assertRaises(RuntimeError):
                from modules.media import ffmpeg_utils

                ffmpeg_utils._finalize_ffmpeg_progress(process, ["ffmpeg"], 0.0, 0.0, "desc")

    def test_run_ffmpeg_progress_exception(self):
        with patch("modules.media.ffmpeg_utils.subprocess.Popen", side_effect=OSError("Popen fail")):
            with self.assertRaises(OSError):
                utils.run_ffmpeg_progress(["cmd"], "desc", 100)

    def test_extract_clean_audio_reuse(self):
        with (
            patch("os.path.exists", return_value=True),
            patch("modules.media.ffmpeg_utils.get_audio_duration", return_value=123.45),
            patch("modules.media.ffmpeg_utils.run_ffmpeg_progress") as mock_run_ffmpeg_progress,
            patch("modules.media.ffmpeg_utils.log"),
        ):
            res = utils.extract_clean_audio("video.mp4")
            self.assertTrue(res.endswith("_temp.wav"))
            mock_run_ffmpeg_progress.assert_not_called()

    def test_extract_clean_audio_fail(self):
        with (
            patch("os.path.exists", side_effect=[False, False]),
            patch("modules.media.ffmpeg_utils.get_audio_duration", return_value=123.45),
            patch("modules.media.ffmpeg_utils.run_ffmpeg_progress", side_effect=RuntimeError("Extraction failed")),
            patch("modules.media.ffmpeg_utils.log"),
        ):
            with self.assertRaises(RuntimeError):
                utils.extract_clean_audio("video.mp4")

    def test_extract_clean_audio_invalid_output_raises(self):
        with (
            patch("os.path.exists", side_effect=[False, False, True]),
            patch("modules.media.ffmpeg_utils.get_audio_duration", return_value=10.0),
            patch("modules.media.ffmpeg_utils.run_ffmpeg_progress"),
            patch("os.path.getsize", return_value=1),
            patch("modules.media.ffmpeg_utils.log"),
        ):
            with self.assertRaises(RuntimeError):
                utils.extract_clean_audio("video.mp4")

    def test_cleanup_temp_files_oserror(self):
        with patch("os.listdir", return_value=["test.wav"]), patch("os.remove", side_effect=OSError("Permission denied")):
            utils.cleanup_temp_files(".", "test", "video.mp4")  # Should not raise

    def test_get_cpu_name_exception(self):
        winreg_stub = MagicMock()
        winreg_stub.HKEY_LOCAL_MACHINE = object()
        winreg_stub.OpenKey.side_effect = OSError("Winreg fail")

        with (
            patch("sys.platform", "win32"),
            patch.dict("sys.modules", {"winreg": winreg_stub}),
            patch("modules.media.hardware_utils.winreg", winreg_stub),
        ):
            name = utils.get_cpu_name()
            self.assertIsNotNone(name)

    def test_save_srt_failure_cleanup(self):
        with (
            patch("builtins.open", mock_open()),
            patch("os.replace", side_effect=OSError("Replace fail")),
            patch("os.path.exists", return_value=True),
            patch("os.remove") as mock_remove,
        ):
            with self.assertRaises(OSError):
                utils.save_srt([], "test.srt")
            mock_remove.assert_called()

    def test_check_srt_corruption(self):
        from modules.subtitles import srt_io

        self.assertTrue(srt_io._check_srt_corruption("1", "Not a timestamp"))
        self.assertTrue(srt_io._check_srt_corruption("31401:58:00,000 --> 00:00:02,000"))
        self.assertFalse(srt_io._check_srt_corruption("1", "00:00:00,000 --> 00:00:01,000"))

    def test_validate_srt_edge_cases(self):
        # Empty
        with (
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=100),
            patch("builtins.open", mock_open(read_data="   ")),
        ):
            self.assertFalse(utils.validate_srt("empty.srt"))

        # Missing separator
        with (
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=100),
            patch("builtins.open", mock_open(read_data="1\n00:00:00\nText")),
        ):
            self.assertFalse(utils.validate_srt("no_sep.srt"))

        # Impossible timestamp hour width should be rejected.
        impossible_hours = "1\n31401:58:00,000 --> 00:00:02,000\nText"
        with (
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=100),
            patch("builtins.open", mock_open(read_data=impossible_hours)),
        ):
            self.assertFalse(utils.validate_srt("impossible_hours.srt"))

        # Index follow-up corruption should be rejected.
        invalid_followup = "1\nNot a timestamp\n\n2\n00:00:00,000 --> 00:00:01,000\nText"
        with (
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=100),
            patch("builtins.open", mock_open(read_data=invalid_followup)),
        ):
            self.assertFalse(utils.validate_srt("invalid_followup.srt"))

    def test_validate_srt_oserror(self):
        with patch("os.path.exists", return_value=True), patch("os.path.getsize", side_effect=OSError("io")):
            self.assertFalse(utils.validate_srt("broken.srt"))

    def test_parse_srt_corrupted(self):
        with patch("modules.subtitles.srt_io.validate_srt", return_value=False), patch("logging.getLogger"):
            self.assertEqual(utils.parse_srt("bad.srt"), [])

    def test_parse_srt_garbage_chunks(self):
        content = "NotADigit\n00:00:00,000 --> 00:00:01,000\nText\n\n2\nInvalidTime\nText"
        with patch("modules.subtitles.srt_io.validate_srt", return_value=True), patch("builtins.open", mock_open(read_data=content)):
            segs = utils.parse_srt("garbage.srt")
            self.assertEqual(len(segs), 0)

    def test_parse_srt_invalid_time_range_branch(self):
        content = "1\na --> b --> c\nText"
        with patch("modules.subtitles.srt_io.validate_srt", return_value=True), patch("builtins.open", mock_open(read_data=content)):
            segs = utils.parse_srt("invalid_time.srt")
            self.assertEqual(segs, [])

    def test_format_timestamp_clamps_negative_seconds(self):
        from modules.subtitles import timestamp_utils

        self.assertEqual(timestamp_utils.format_timestamp(-1.25), "00:00:00,000")


if __name__ == "__main__":
    unittest.main()
