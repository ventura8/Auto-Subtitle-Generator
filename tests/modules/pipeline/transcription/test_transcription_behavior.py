import importlib
import unittest
from unittest.mock import MagicMock, patch

transcription = importlib.import_module("modules.pipeline.transcription")


class TestCoverageTranscription(unittest.TestCase):
    def test_detect_and_separate_vocals_disabled(self):
        with patch("modules.configuration.config.USE_VOCAL_SEPARATION", False):
            self.assertEqual(transcription._detect_and_separate_vocals("vid.mp4", MagicMock()), "vid.mp4")

    def test_detect_and_separate_vocals_resume(self):
        with (
            patch("modules.configuration.config.USE_VOCAL_SEPARATION", True),
            patch("modules.pipeline.transcription._get_separated_vocal_path", return_value="vocal.wav"),
            patch("modules.pipeline.transcription.log") as mock_log,
        ):
            res = transcription._detect_and_separate_vocals("vid.mp4", MagicMock())
            self.assertEqual(res, "vocal.wav")
            mock_log.assert_called()

    def test_detect_and_separate_vocals_fail(self):
        mm = MagicMock()
        mm.get_separator.side_effect = RuntimeError("Sep fail")
        with (
            patch("modules.configuration.config.USE_VOCAL_SEPARATION", True),
            patch("modules.pipeline.transcription._get_separated_vocal_path", return_value=None),
            patch("modules.utils.extract_clean_audio"),
            patch("modules.workdir.ensure_work_dir", return_value="vid.asg-temp"),
            patch("modules.pipeline.transcription.create_private_dir", return_value="vid.asg-temp/.asg-tmp-x"),
            patch("modules.workdir.remove_scratch_dir", return_value=True) as mock_remove_scratch,
            patch("modules.pipeline.transcription.log") as mock_log,
        ):
            res = transcription._detect_and_separate_vocals("vid.mp4", mm)
            self.assertEqual(res, "vid.mp4")
            mock_log.assert_any_call("  [Sep] Warning: Separation failed (Sep fail). Using original audio.", "WARNING")
            # The private separator scratch directory is always removed, even on failure.
            mock_remove_scratch.assert_called_once_with("vid.asg-temp/.asg-tmp-x")

    def test_detect_and_separate_vocals_missing_optional_backend(self):
        with (
            patch("modules.configuration.config.USE_VOCAL_SEPARATION", True),
            patch("modules.pipeline.transcription._get_separated_vocal_path", return_value=None),
            patch("modules.pipeline.transcription._run_vocal_separation", side_effect=ImportError("audio_separator")),
            patch("modules.pipeline.transcription.log") as mock_log,
        ):
            self.assertEqual(transcription._detect_and_separate_vocals("vid.mp4", MagicMock()), "vid.mp4")
            mock_log.assert_any_call("  [Sep] Warning: Separation failed (audio_separator). Using original audio.", "WARNING")

    def test_filter_hallucinations_branches(self):
        seg = MagicMock(text="thanks for watching")
        phrases = ["thanks for watching"]
        filtered, count = transcription._filter_hallucinations([seg], phrases)
        self.assertEqual(len(filtered), 0)
        self.assertEqual(count, 1)

    def test_transcribe_video_audio_no_prompt_log(self):
        mm = MagicMock()
        mock_whisper = mm.get_whisper.return_value
        mock_whisper.transcribe.return_value = ([], MagicMock(duration=10, language="en", language_probability=0.9))
        with (
            patch("modules.configuration.config.USE_VOCAL_SEPARATION", False),
            patch("modules.configuration.config.INITIAL_PROMPT", None),
            patch("modules.utils.extract_clean_audio", return_value="audio.wav"),
            patch("modules.pipeline.transcription.log") as mock_log,
        ):
            transcription.transcribe_video_audio("vid.mp4", mm, forced_prompt=None)
            mock_log.assert_any_call("  [Whisper] Config: No Input Prompt")

    def test_transcribe_video_audio_runtime_error(self):
        mm = MagicMock()
        mock_whisper = mm.get_whisper.return_value
        mock_whisper.transcribe.side_effect = RuntimeError("Other error")
        with (
            patch("modules.configuration.config.USE_VOCAL_SEPARATION", False),
            patch("modules.utils.extract_clean_audio", return_value="audio.wav"),
            patch("modules.pipeline.transcription.log"),
        ):
            with self.assertRaises(RuntimeError):
                transcription.transcribe_video_audio("vid.mp4", mm)

    def test_transcribe_video_audio_low_conf(self):
        mm = MagicMock()
        mock_whisper = mm.get_whisper.return_value
        mock_whisper.transcribe.return_value = ([], MagicMock(duration=10, language="en", language_probability=0.1))
        with (
            patch("modules.configuration.config.USE_VOCAL_SEPARATION", False),
            patch("modules.utils.extract_clean_audio", return_value="audio.wav"),
            patch("modules.pipeline.transcription.log") as mock_log,
        ):
            transcription.transcribe_video_audio("vid.mp4", mm)
            mock_log.assert_any_call("  [Warning] Low language confidence (0.10).", "WARNING")

    def test_transcribe_video_audio_forced_lang(self):
        mm = MagicMock()
        mock_whisper = mm.get_whisper.return_value
        mock_whisper.transcribe.return_value = ([], MagicMock(duration=10, language="ro", language_probability=0.9))
        with (
            patch("modules.configuration.config.USE_VOCAL_SEPARATION", False),
            patch("modules.utils.extract_clean_audio", return_value="audio.wav"),
            patch("modules.pipeline.transcription.log") as mock_log,
        ):
            transcription.transcribe_video_audio("vid.mp4", mm, forced_lang="ro")
            mock_log.assert_any_call("  [Whisper] Config: Forced Language='ro'")


if __name__ == "__main__":
    unittest.main()
