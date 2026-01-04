import unittest
from unittest.mock import MagicMock, patch, mock_open
import os
import sys

# Ensure modules can be imported
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import auto_subtitle

class TestCoverageAutoSubtitle(unittest.TestCase):

    def setUp(self):
        # Reset torch handle to allow re-init
        auto_subtitle.torch = None

    @patch("auto_subtitle.print_progress_bar")
    @patch("auto_subtitle.log")
    @patch("sys.exit")
    def test_init_torch_fail(self, mock_exit, mock_log, mock_bar):
        with patch.dict("sys.modules", {"torch": MagicMock(side_effect=ImportError("No torch"))}):
             # We need to trigger the import inside _init_torch_and_hardware
             with patch("builtins.__import__", side_effect=lambda name, *args, **kwargs: 
                        exec("raise ImportError('No torch')") if name == "torch" else MagicMock()):
                 auto_subtitle._init_torch_and_hardware(1, 6)
                 mock_exit.assert_called_with(1)

    @patch("auto_subtitle.print_progress_bar")
    @patch("auto_subtitle.log")
    @patch("sys.exit")
    def test_init_transformers_fail(self, mock_exit, mock_log, mock_bar):
        with patch("builtins.__import__", side_effect=lambda name, *args, **kwargs: 
                   exec("raise ImportError('No transformers')") if name == "transformers" else MagicMock()):
            auto_subtitle._init_nvidia_and_transformers(3, 6)
            mock_exit.assert_called_with(1)

    @patch("auto_subtitle.print_progress_bar")
    @patch("auto_subtitle.log")
    @patch("sys.exit")
    def test_init_whisper_fail(self, mock_exit, mock_log, mock_bar):
        with patch("builtins.__import__", side_effect=lambda name, *args, **kwargs: 
                   exec("raise ImportError('No whisper')") if name == "faster_whisper" else MagicMock()):
            auto_subtitle._init_whisper_and_separator(4, 6)
            mock_exit.assert_called_with(1)

    @patch("auto_subtitle.print_progress_bar")
    @patch("auto_subtitle.log")
    def test_init_separator_skip(self, mock_log, mock_bar):
        with patch("builtins.__import__", side_effect=lambda name, *args, **kwargs: 
                   exec("raise ImportError('No separator')") if "audio_separator" in name else MagicMock()):
            auto_subtitle._init_whisper_and_separator(5, 6)
            mock_log.assert_called()

    def test_init_ai_engine_already_init(self):
        auto_subtitle.torch = MagicMock()
        with patch("builtins.print") as mock_print:
            auto_subtitle.init_ai_engine()
            mock_print.assert_not_called()

    def test_get_nvidia_bin_lib_paths(self):
        with patch("os.path.exists", return_value=True), \
             patch("os.path.isdir", return_value=True), \
             patch("os.listdir", return_value=["item1"]):
            paths = auto_subtitle._get_nvidia_bin_lib_paths("site-packages")
            self.assertTrue(len(paths) > 0)

    @patch("os.add_dll_directory", create=True)
    def test_apply_paths_to_env(self, mock_add):
        with patch("os.environ", {"PATH": ""}):
            auto_subtitle._apply_paths_to_env(["/new/path"])
            self.assertIn("/new/path", os.environ["PATH"])
            # Assuming hasattr(os, 'add_dll_directory') is true on this env
            if hasattr(os, 'add_dll_directory'):
                mock_add.assert_called_with("/new/path")

    def test_load_nvidia_paths_torch_fail(self):
        with patch("site.getsitepackages", return_value=[]), \
             patch("builtins.__import__", side_effect=lambda name, *args, **kwargs: 
                   exec("raise ImportError()") if name == "torch" else MagicMock()):
            auto_subtitle.load_nvidia_paths()

    def test_check_resume_empty_srt(self):
        with patch("os.path.exists", return_value=True), \
             patch("auto_subtitle.utils.parse_srt", return_value=[]), \
             patch("auto_subtitle.log"):
            res = auto_subtitle._check_resume("folder", "base", "vid.mp4", "en")
            self.assertEqual(res, (None, None, None))

    def test_embed_subtitles_empty(self):
        self.assertIsNone(auto_subtitle.embed_subtitles("vid.mp4", []))

    @patch("auto_subtitle.utils.get_audio_duration", side_effect=Exception("Error"))
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
    @patch("auto_subtitle.utils.save_srt", side_effect=Exception("Save fail"))
    @patch("auto_subtitle.log")
    @patch("auto_subtitle.translate_segments")
    @patch("auto_subtitle.embed_subtitles")
    def test_process_video_save_srt_error(self, mock_embed, mock_trans, mock_log, mock_save, mock_ob):
        auto_subtitle.process_video("vid.mp4", MagicMock())
        mock_log.assert_any_call("  [Error] Failed to save source SRT: Save fail", "ERROR")

    @patch("auto_subtitle._obtain_segments", return_value=([MagicMock()], "en", "audio.wav"))
    @patch("auto_subtitle.translate_segments", side_effect=Exception("Trans fail"))
    @patch("auto_subtitle.log")
    def test_process_video_translation_fail(self, mock_log, mock_trans, mock_ob):
        auto_subtitle.process_video("vid.mp4", MagicMock())
        mock_log.assert_any_call("Translation failed: Trans fail", "ERROR")

    def test_get_input_files_exclude_multilang(self):
        with patch("argparse.ArgumentParser.parse_args", return_value=MagicMock(input_path="folder", cpu=False, lang=None, prompt=None)), \
             patch("os.path.isfile", return_value=False), \
             patch("os.path.isdir", return_value=True), \
             patch("os.walk", return_value=[(".", [], ["vid.mp4", "vid_multilang.mp4"])]):
            files, _, _ = auto_subtitle.get_input_files()
            self.assertEqual(len(files), 1)

    @patch("sys.exit")
    def test_get_input_files_not_found(self, mock_exit):
        with patch("argparse.ArgumentParser.parse_args", return_value=MagicMock(input_path="ghost", cpu=False, lang=None, prompt=None)), \
             patch("os.path.isfile", return_value=False), \
             patch("os.path.isdir", return_value=False):
            auto_subtitle.get_input_files()
            mock_exit.assert_called_with(1)

if __name__ == "__main__":
    unittest.main()
