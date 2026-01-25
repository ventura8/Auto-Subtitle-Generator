import unittest
from unittest.mock import MagicMock, patch, mock_open
import os
import sys

# Ensure modules can be imported
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

# Avoid global sys.modules hacks. conftest.py handles these.


class TestUtils(unittest.TestCase):
    def setUp(self):
        global utils
        from modules import utils

    @patch("os.path.getsize", return_value=100)
    @patch("os.path.exists", return_value=True)
    def test_validate_srt_valid(self, mock_exists, mock_size):
        # Create a dummy valid-looking SRT
        content = "1\n00:00:00,000 --> 00:00:01,000\nHello\n\n"
        with patch("builtins.open", mock_open(read_data=content)):
            self.assertTrue(utils.validate_srt("dummy.srt"))

    @patch("os.path.getsize", return_value=100)
    @patch("os.path.exists", return_value=True)
    def test_validate_srt_invalid(self, mock_exists, mock_size):
        content = "Not an SRT"
        with patch("builtins.open", mock_open(read_data=content)):
            self.assertFalse(utils.validate_srt("dummy.srt"))

    @patch("subprocess.check_output")
    def test_get_audio_duration(self, mock_co):
        mock_co.return_value = b"123.45"
        duration = utils.get_audio_duration("video.mp4")
        self.assertEqual(duration, 123.45)

    @patch("subprocess.check_output", side_effect=Exception("FFprobe fail"))
    def test_get_audio_duration_fail(self, mock_co):
        duration = utils.get_audio_duration("video.mp4")
        self.assertEqual(duration, 0)

    @patch("os.replace")
    @patch("os.remove")
    def test_save_srt(self, mock_remove, mock_replace):
        segments = [MagicMock(start=0, end=1.5, text="Hello")]
        with patch("builtins.open", mock_open()) as m:
            utils.save_srt(segments, "test.srt")
            m.assert_called_with("test.srt.tmp", "w", encoding="utf-8")

    @patch("os.replace")
    @patch("os.remove")
    def test_save_translated_srt(self, mock_remove, mock_replace):
        segments = [MagicMock(start=0, end=1.5)]
        translations = ["Hola"]
        with patch("builtins.open", mock_open()) as m:
            utils.save_translated_srt(segments, translations, "test_es.srt")
            m.assert_called()

    @patch("os.path.exists", return_value=False)
    def test_validate_srt_not_exist(self, mock_exists):
        self.assertFalse(utils.validate_srt("nonexistent.srt"))

    @patch("os.path.getsize", return_value=0)
    @patch("os.path.exists", return_value=True)
    def test_validate_srt_empty(self, mock_exists, mock_size):
        self.assertFalse(utils.validate_srt("empty.srt"))

    def test_parse_timestamp_broken(self):
        # Test invalid formats
        self.assertEqual(utils.parse_timestamp("invalid"), 0.0)
        self.assertEqual(utils.parse_timestamp("00:00:00"), 0.0)  # missing comma

    @patch("os.remove")
    @patch("os.listdir", return_value=["base.wav"])
    @patch("os.path.exists", return_value=True)
    def test_cleanup_temp_files(self, mock_exists, mock_listdir, mock_remove):
        utils.cleanup_temp_files("folder", "base", "video.mp4")
        # Should attempt to remove various temp files
        mock_remove.assert_called()

    @patch("modules.utils.log")
    def test_save_srt_error(self, mock_log):
        # Test exception during write
        with patch("builtins.open", side_effect=OSError("Write Fail")):
            with self.assertRaises(OSError):
                utils.save_srt([], "fail.srt")

    @patch("os.name", "nt")
    def test_init_console_windows(self):
        # Mock ctypes via sys.modules because it's imported LOCALLY
        mock_ctypes = MagicMock()
        mock_kernel32 = MagicMock()
        mock_ctypes.windll.kernel32 = mock_kernel32
        mock_kernel32.GetStdHandle.return_value = 123
        # GetConsoleMode needs to return non-zero (True)
        # And write to the byref arg if possible, but just returning True is enough to enter the if
        mock_kernel32.GetConsoleMode.return_value = 1

        with patch.dict("sys.modules", {"ctypes": mock_ctypes}):
            utils.init_console()

        mock_kernel32.SetConsoleMode.assert_called()

    @patch("modules.utils.active_subprocesses", new_callable=list)
    @patch("sys.exit")
    @patch("sys.platform", "win32")
    @patch("subprocess.call")
    def test_handle_shutdown(self, mock_call, mock_exit, mock_procs):
        # Mock a running process
        proc = MagicMock()
        proc.poll.return_value = None  # Running
        proc.pid = 1234
        mock_procs.append(proc)

        # Inject global list into utils (since it's imported)
        with patch("modules.utils.active_subprocesses", mock_procs):
            utils.handle_shutdown(None, None)

        proc.terminate.assert_called()
        mock_call.assert_called()  # taskkill on windows
        mock_exit.assert_called_with(0)

    def test_print_banner(self):
        # Tests lines 20-78
        mock_opt = MagicMock()
        mock_opt.gpu_name = "TestGPU"
        mock_opt.vram_gb = 12
        mock_opt.profile = "ULTRA"
        mock_opt.config.get.return_value = "8"  # batch/thread

        with patch("builtins.print") as m_print, \
                patch("platform.system", return_value="TestOS"), \
                patch("platform.release", return_value="1.0"), \
                patch("os.cpu_count", return_value=16):

            utils.print_banner(mock_opt)
            # Verify basic output parts
            calls = [str(c) for c in m_print.call_args_list]
            self.assertTrue(any("AI HYBRID" in c for c in calls))
            self.assertTrue(any("TestGPU" in c for c in calls))
            self.assertTrue(any("ULTRA" in c for c in calls))

    def test_get_cpu_name_windows(self):
        # Tests lines 483-495 (Windows Registry)
        # Mock winreg module being available
        mock_winreg = MagicMock()
        mock_winreg.QueryValueEx.return_value = ["Intel Mock CPU", 1]

        with patch.dict("sys.modules", {"winreg": mock_winreg}):
            with patch("sys.platform", "win32"):
                name = utils.get_cpu_name()
                self.assertEqual(name, "Intel Mock CPU")

    def test_run_ffmpeg_progress_logic(self):
        # Test the parsing logic of run_ffmpeg_progress
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0  # Always done, but loop continues until readline is empty
        mock_proc.returncode = 0
        mock_proc.stderr.readline.side_effect = [
            "frame= 100 fps= 25 q=-1.0 size= 100kB time=00:00:01.00 bitrate= 100kbits/s speed=1.0x\n",
            "frame= 200 fps= 25 q=-1.0 size= 200kB time=00:00:02.00 bitrate= 100kbits/s speed=1.0x\n",
            ""
        ]

        with patch("subprocess.Popen", return_value=mock_proc), \
                patch("modules.utils.register_subprocess"), \
                patch("modules.utils.print_progress_bar") as m_bar:

            utils.run_ffmpeg_progress(["ffmpeg", "-ver"], "Processing", total_duration=4.0)
            self.assertTrue(m_bar.call_count >= 2)

    def test_extract_clean_audio(self):
        # Test extract_clean_audio success path
        # First call is to check reuse (False), second is final check (True)
        exists_side_effect = [False, True]
        with patch("os.path.exists", side_effect=exists_side_effect), \
                patch("modules.utils.get_audio_duration", return_value=123.0), \
                patch("modules.utils.run_ffmpeg_progress"), \
                patch("os.path.getsize", return_value=2048):  # valid size

            res = utils.extract_clean_audio("video.mp4")
            self.assertTrue(res.endswith("_temp.wav"))

    def test_save_and_parse_srt(self):
        # Test save_srt and then parse_srt
        from modules.models import Segment
        segs = [Segment(0.0, 1.0, "Hello"), Segment(1.0, 2.5, "World")]

        # We need to use a real temporary file to test read/write cleanly,
        # or use mock_open if we are careful. mock_open is harder for read back what we wrote.
        # Let's use mock_open for write, and a separate mock_open for read?
        # Actually parse_srt reads.

        # Test SAVE
        m_open = mock_open()
        with patch("builtins.open", m_open), patch("os.replace"):
            utils.save_srt(segs, "out.srt")
            # Verify write content
            handle = m_open()
            handle.write.assert_called()
            written = "".join(call.args[0] for call in handle.write.call_args_list)
            self.assertIn("00:00:00,000 --> 00:00:01,000", written)
            self.assertIn("Hello", written)

        # Test PARSE
        fake_srt = "1\n00:00:00,000 --> 00:00:01,000\nHello\n\n2\n00:00:01,000 --> 00:00:02,500\nWorld"
        with patch("builtins.open", mock_open(read_data=fake_srt)), \
                patch("os.path.exists", return_value=True), \
                patch("os.path.getsize", return_value=100):
            parsed = utils.parse_srt("out.srt")
            self.assertEqual(len(parsed), 2)
            self.assertEqual(parsed[0].text, "Hello")
            self.assertEqual(parsed[1].end, 2.5)


if __name__ == "__main__":
    unittest.main()
