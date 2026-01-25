from modules import transcription
import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Ensure modules can be imported
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)


class TestCoverageTranscription(unittest.TestCase):

    def test_get_separated_vocal_path_success(self):
        with patch("os.listdir", return_value=["vid_(Vocals)_.wav"]), \
                patch("os.path.join", return_value="dir/vid_(Vocals)_.wav"):
            res = transcription._get_separated_vocal_path("vid.mp4")
            self.assertEqual(res, "dir/vid_(Vocals)_.wav")

    def test_get_separated_vocal_path_exception(self):
        with patch("os.listdir", side_effect=OSError()):
            self.assertIsNone(transcription._get_separated_vocal_path("vid.mp4"))

    def test_detect_and_separate_vocals_disabled(self):
        with patch("modules.config.USE_VOCAL_SEPARATION", False):
            self.assertEqual(transcription._detect_and_separate_vocals("vid.mp4", MagicMock()), "vid.mp4")

    def test_detect_and_separate_vocals_resume(self):
        with patch("modules.config.USE_VOCAL_SEPARATION", True), \
                patch("modules.transcription._get_separated_vocal_path", return_value="vocal.wav"), \
                patch("modules.transcription.log") as mock_log:
            res = transcription._detect_and_separate_vocals("vid.mp4", MagicMock())
            self.assertEqual(res, "vocal.wav")
            mock_log.assert_called()

    def test_detect_and_separate_vocals_fail(self):
        mm = MagicMock()
        mm.get_separator.side_effect = Exception("Sep fail")
        with patch("modules.config.USE_VOCAL_SEPARATION", True), \
                patch("modules.transcription._get_separated_vocal_path", return_value=None), \
                patch("modules.utils.extract_clean_audio"), \
                patch("modules.transcription.log") as mock_log:
            res = transcription._detect_and_separate_vocals("vid.mp4", mm)
            self.assertEqual(res, "vid.mp4")
            mock_log.assert_any_call("  [Sep] Warning: Separation failed (Sep fail). Using original audio.", "WARNING")

    def test_filter_hallucinations_branches(self):
        seg = MagicMock(text="Nu uitați să dați like")
        phrases = ["nu uitați să dați like"]
        filtered, count = transcription._filter_hallucinations([seg], phrases)
        self.assertEqual(len(filtered), 0)
        self.assertEqual(count, 1)

    def test_transcribe_video_audio_no_prompt_log(self):
        mm = MagicMock()
        mock_whisper = mm.get_whisper.return_value
        mock_whisper.transcribe.return_value = ([], MagicMock(duration=10, language="en", language_probability=0.9))
        with patch("modules.config.USE_VOCAL_SEPARATION", False), \
                patch("modules.config.INITIAL_PROMPT", None), \
                patch("modules.utils.extract_clean_audio", return_value="audio.wav"), \
                patch("modules.transcription.log") as mock_log:
            transcription.transcribe_video_audio("vid.mp4", mm, forced_prompt=None)
            mock_log.assert_any_call("  [Whisper] Config: No Input Prompt")

    def test_transcribe_video_audio_runtime_error(self):
        mm = MagicMock()
        mock_whisper = mm.get_whisper.return_value
        mock_whisper.transcribe.side_effect = RuntimeError("Other error")
        with patch("modules.config.USE_VOCAL_SEPARATION", False), \
                patch("modules.utils.extract_clean_audio", return_value="audio.wav"), \
                patch("modules.transcription.log"):
            with self.assertRaises(RuntimeError):
                transcription.transcribe_video_audio("vid.mp4", mm)

    def test_transcribe_video_audio_low_conf(self):
        mm = MagicMock()
        mock_whisper = mm.get_whisper.return_value
        mock_whisper.transcribe.return_value = ([], MagicMock(duration=10, language="en", language_probability=0.1))
        with patch("modules.config.USE_VOCAL_SEPARATION", False), \
                patch("modules.utils.extract_clean_audio", return_value="audio.wav"), \
                patch("modules.transcription.log") as mock_log:
            transcription.transcribe_video_audio("vid.mp4", mm)
            mock_log.assert_any_call("  [Warning] Low language confidence (0.10).", "WARNING")

    def test_process_separator_outputs(self):
        output_files = ["dir/vid_(Vocals).wav", "dir/vid_(Instrumental).wav"]
        with patch("os.path.abspath", side_effect=lambda x: x), \
                patch("os.path.exists", return_value=True), \
                patch("os.remove"), \
                patch("os.rename"):
            res = transcription._process_separator_outputs(output_files, "target")
            self.assertIn("vid_(Vocals).wav", res)

    def test_transcribe_video_audio_forced_lang(self):
        mm = MagicMock()
        mock_whisper = mm.get_whisper.return_value
        mock_whisper.transcribe.return_value = ([], MagicMock(duration=10, language="ro", language_probability=0.9))
        with patch("modules.config.USE_VOCAL_SEPARATION", False), \
                patch("modules.utils.extract_clean_audio", return_value="audio.wav"), \
                patch("modules.transcription.log") as mock_log:
            transcription.transcribe_video_audio("vid.mp4", mm, forced_lang="ro")
            mock_log.assert_any_call("  [Whisper] Config: Forced Language='ro'")


if __name__ == "__main__":
    unittest.main()
