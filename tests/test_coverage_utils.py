import unittest
from unittest.mock import MagicMock, patch, mock_open
import os
import sys
import subprocess
import signal

# Ensure modules can be imported
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from modules import utils

class TestCoverageUtils(unittest.TestCase):

    def test_print_banner_with_optimizer(self):
        mock_opt = MagicMock()
        mock_opt.gpu_name = "TestGPU"
        mock_opt.vram_gb = 16
        mock_opt.profile = "HIGH"
        mock_opt.config = {"nllb_batch": 4, "ffmpeg_threads": 8}
        
        with patch("builtins.print"), \
             patch("platform.system", return_value="Windows"), \
             patch("platform.release", return_value="10"), \
             patch("modules.utils.get_cpu_name", return_value="TestCPU"):
            utils.print_banner(mock_opt)

    def test_handle_shutdown_error(self):
        proc = MagicMock()
        proc.poll.return_value = None
        proc.terminate.side_effect = Exception("Kill fail")
        
        with patch("modules.utils.active_subprocesses", [proc]), \
             patch("builtins.print"), \
             patch("sys.exit"):
            utils.handle_shutdown(None, None)
            proc.terminate.assert_called()

    def test_init_console_exception(self):
        with patch("os.name", "nt"), \
             patch("ctypes.windll.kernel32.GetStdHandle", side_effect=Exception("Ctypes fail"), create=True):
            utils.init_console() # Should just pass

    def test_setup_signal_handlers(self):
        with patch("signal.signal") as mock_sig, \
             patch("sys.platform", "linux"):
            utils.setup_signal_handlers()
            self.assertEqual(mock_sig.call_count, 2)

    def test_setup_signal_handlers_windows(self):
        mock_handler = MagicMock()
        with patch("signal.signal"), \
             patch("sys.platform", "win32"), \
             patch("ctypes.WINFUNCTYPE", return_value=lambda x: x, create=True), \
             patch("ctypes.windll.kernel32.SetConsoleCtrlHandler", return_value=True, create=True):
            utils.setup_signal_handlers()

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
            with patch("sys.stdout.write", side_effect=[UnicodeEncodeError('utf-8', u'', 0, 1, 'mock'), None]):
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

    def test_process_ffmpeg_line_exception(self):
        with patch("modules.utils.parse_timestamp", side_effect=Exception("Parse fail")), \
             patch("gc.collect") as mock_gc:
            utils._process_ffmpeg_line("time=00:00:01.00", 0, 100, "Desc")
            mock_gc.assert_called()

    def test_run_ffmpeg_progress_exception(self):
        with patch("subprocess.Popen", side_effect=Exception("Popen fail")):
            with self.assertRaises(Exception):
                utils.run_ffmpeg_progress(["cmd"], "desc", 100)

    def test_extract_clean_audio_reuse(self):
        with patch("os.path.exists", return_value=True), \
             patch("modules.utils.get_audio_duration", return_value=123.45), \
             patch("modules.utils.log"):
            res = utils.extract_clean_audio("video.mp4")
            self.assertTrue(res.endswith("_temp.wav"))

    def test_extract_clean_audio_fail(self):
        with patch("os.path.exists", side_effect=[False, False]), \
             patch("modules.utils.get_audio_duration", return_value=123.45), \
             patch("modules.utils.run_ffmpeg_progress", side_effect=Exception("Extraction failed")), \
             patch("modules.utils.log"):
            with self.assertRaises(Exception):
                utils.extract_clean_audio("video.mp4")

    def test_cleanup_temp_files_oserror(self):
        with patch("os.listdir", return_value=["test.wav"]), \
             patch("os.remove", side_effect=OSError("Permission denied")):
            utils.cleanup_temp_files(".", "test", "video.mp4") # Should not raise

    def test_get_cpu_name_exception(self):
        with patch("sys.platform", "win32"), \
             patch("winreg.OpenKey", side_effect=Exception("Winreg fail"), create=True):
            name = utils.get_cpu_name()
            self.assertIsNotNone(name)

    def test_save_srt_failure_cleanup(self):
        with patch("builtins.open", mock_open()), \
             patch("os.replace", side_effect=Exception("Replace fail")), \
             patch("os.path.exists", return_value=True), \
             patch("os.remove") as mock_remove:
            with self.assertRaises(Exception):
                utils.save_srt([], "test.srt")
            mock_remove.assert_called()

    def test_check_srt_corruption(self):
        self.assertTrue(utils._check_srt_corruption("1", "Not a timestamp"))
        self.assertTrue(utils._check_srt_corruption("31401:58:00,000 --> 00:00:02,000"))
        self.assertFalse(utils._check_srt_corruption("1", "00:00:00,000 --> 00:00:01,000"))

    def test_validate_srt_edge_cases(self):
        # Empty
        with patch("os.path.exists", return_value=True), \
             patch("os.path.getsize", return_value=100), \
             patch("builtins.open", mock_open(read_data="   ")):
            self.assertFalse(utils.validate_srt("empty.srt"))
        
        # Missing separator
        with patch("os.path.exists", return_value=True), \
             patch("os.path.getsize", return_value=100), \
             patch("builtins.open", mock_open(read_data="1\n00:00:00\nText")):
            self.assertFalse(utils.validate_srt("no_sep.srt"))

    def test_parse_srt_corrupted(self):
        with patch("modules.utils.validate_srt", return_value=False), \
             patch("logging.getLogger"):
            self.assertEqual(utils.parse_srt("bad.srt"), [])

    def test_parse_srt_garbage_chunks(self):
        content = "NotADigit\n00:00:00,000 --> 00:00:01,000\nText\n\n2\nInvalidTime\nText"
        with patch("modules.utils.validate_srt", return_value=True), \
             patch("builtins.open", mock_open(read_data=content)):
            segs = utils.parse_srt("garbage.srt")
            self.assertEqual(len(segs), 0)

if __name__ == "__main__":
    unittest.main()
