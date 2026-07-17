import importlib
import unittest
from unittest.mock import MagicMock, patch

transcription = importlib.import_module("modules.pipeline.transcription")


class TestCoverageTranscription(unittest.TestCase):
    def test_get_separated_vocal_path_success(self):
        with patch("os.listdir", return_value=["vid_(Vocals)_.wav"]), patch("os.path.join", return_value="dir/vid_(Vocals)_.wav"):
            res = transcription._get_separated_vocal_path("vid.mp4")
            self.assertEqual(res, "dir/vid_(Vocals)_.wav")

    def test_get_separated_vocal_path_exception(self):
        with patch("os.listdir", side_effect=OSError()):
            self.assertIsNone(transcription._get_separated_vocal_path("vid.mp4"))

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
            patch("modules.pipeline.transcription.log") as mock_log,
        ):
            res = transcription._detect_and_separate_vocals("vid.mp4", mm)
            self.assertEqual(res, "vid.mp4")
            mock_log.assert_any_call("  [Sep] Warning: Separation failed (Sep fail). Using original audio.", "WARNING")

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

    def test_process_separator_outputs(self):
        output_files = ["dir/vid_(Vocals).wav", "dir/vid_(Instrumental).wav"]

        def exists_side_effect(path):
            if path == "dir/vid_(Instrumental).wav":
                return True
            if "target" in path and "vid_(Vocals).wav" in path:
                return False
            return "vid_(Vocals).wav" in path

        with (
            patch("os.path.abspath", side_effect=lambda x: x),
            patch("os.path.exists", side_effect=exists_side_effect),
            patch("os.rename") as mock_rename,
        ):
            res = transcription._process_separator_outputs(output_files, "target")
            self.assertIn("vid_(Vocals).wav", res)
            mock_rename.assert_called_once()
            rename_args = mock_rename.call_args[0]
            self.assertEqual(rename_args[0], "dir/vid_(Vocals).wav")
            self.assertTrue(rename_args[1].endswith("vid_(Vocals).wav"))

    def test_process_separator_outputs_returns_none_when_only_instrumental(self):
        with (
            patch("os.path.abspath", side_effect=lambda x: x),
            patch("os.path.exists", return_value=True),
            patch("os.rename") as mock_rename,
        ):
            res = transcription._process_separator_outputs(["dir/vid_(Instrumental).wav"], "target")
            self.assertIsNone(res)
            mock_rename.assert_not_called()

    def test_process_separator_outputs_ignores_relative_instrumental_from_target_dir(self):
        output_files = [
            "vid_(Vocals).wav",
            "vid_(Instrumental).wav",
        ]

        def exists_side_effect(path):
            # Relative files do not exist from CWD in this scenario.
            if path in {"vid_(Vocals).wav", "vid_(Instrumental).wav"}:
                return False
            # Target-dir-resolved files exist.
            if path in {"target/vid_(Vocals).wav", "target/vid_(Instrumental).wav"}:
                return True
            if path in {"target\\vid_(Vocals).wav", "target\\vid_(Instrumental).wav"}:
                return True
            return False

        with (
            patch("os.path.abspath", side_effect=lambda x: x),
            patch("os.path.isabs", return_value=False),
            patch("os.path.exists", side_effect=exists_side_effect),
            patch("os.rename") as mock_rename,
        ):
            res = transcription._process_separator_outputs(output_files, "target")
            self.assertIsNotNone(res)
            assert res is not None
            self.assertTrue(res.endswith("vid_(Vocals).wav"))
            self.assertIn(mock_rename.call_count, {0, 1})

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
