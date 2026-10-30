import importlib
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from modules.configuration import asr_settings
from modules.media.input_binding import InputRefusedError

transcription = importlib.import_module("modules.pipeline.transcription")


class TestCoverageTranscription(unittest.TestCase):
    def setUp(self):
        # The legacy Whisper path is under test; the shipped default routes per language.
        asr_settings.set_cli_override("whisper")
        self.addCleanup(asr_settings.set_cli_override, None)
        # extract_clean_audio is mocked, but its result is really staged, so it must exist.
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.work_audio = os.path.join(self._tmp.name, "vid_temp.wav")
        with open(self.work_audio, "wb") as handle:
            handle.write(b"RIFF")

    def test_detect_and_separate_vocals_disabled(self):
        stage = MagicMock()
        with patch("modules.configuration.config.USE_VOCAL_SEPARATION", False):
            self.assertIsNone(transcription._detect_and_separate_vocals("vid.mp4", MagicMock(), stage))
        stage.import_file.assert_not_called()

    def test_detect_and_separate_vocals_resume(self):
        stage = MagicMock()
        stage.import_file.return_value = "/stage/vocal.wav"
        with (
            patch("modules.configuration.config.USE_VOCAL_SEPARATION", True),
            patch("modules.pipeline.transcription._get_separated_vocal_path", return_value="vocal.wav"),
            patch("modules.pipeline.transcription.log") as mock_log,
        ):
            res = transcription._detect_and_separate_vocals("vid.mp4", MagicMock(), stage)
            self.assertEqual(res, ("vocal.wav", "/stage/vocal.wav"))
            stage.import_file.assert_called_once_with("vocal.wav")
            mock_log.assert_called()

    def test_detect_and_separate_vocals_refused_resume_falls_back(self):
        stage = MagicMock()
        stage.import_file.side_effect = InputRefusedError("link")
        with (
            patch("modules.configuration.config.USE_VOCAL_SEPARATION", True),
            patch("modules.pipeline.transcription._get_separated_vocal_path", return_value="vocal.wav"),
            patch("modules.pipeline.transcription.log") as mock_log,
        ):
            self.assertIsNone(transcription._detect_and_separate_vocals("vid.mp4", MagicMock(), stage))
            mock_log.assert_any_call("  [Sep] Warning: Separation failed (link). Using original audio.", "WARNING")

    def test_detect_and_separate_vocals_fail(self):
        mm = MagicMock()
        mm.get_separator.side_effect = RuntimeError("Sep fail")
        with (
            patch("modules.configuration.config.USE_VOCAL_SEPARATION", True),
            patch("modules.pipeline.transcription._get_separated_vocal_path", return_value=None),
            patch("modules.utils.extract_clean_audio", return_value=self.work_audio),
            patch("modules.workdir.ensure_work_dir", return_value=self._tmp.name),
            patch("modules.pipeline.transcription.log") as mock_log,
            transcription.audio_stage.open_stage() as stage,
        ):
            res = transcription._detect_and_separate_vocals("vid.mp4", mm, stage)
            self.assertIsNone(res)
            mock_log.assert_any_call("  [Sep] Warning: Separation failed (Sep fail). Using original audio.", "WARNING")
            # Nothing was written into the work directory by the failed separation.
            self.assertEqual(os.listdir(self._tmp.name), ["vid_temp.wav"])
            # The separator was pointed at a stage sub-directory.
            self.assertEqual(os.path.dirname(mm.get_separator.call_args.kwargs["output_dir"]), stage.root)

    def test_detect_and_separate_vocals_missing_optional_backend(self):
        with (
            patch("modules.configuration.config.USE_VOCAL_SEPARATION", True),
            patch("modules.pipeline.transcription._get_separated_vocal_path", return_value=None),
            patch("modules.pipeline.transcription._run_vocal_separation", side_effect=ImportError("audio_separator")),
            patch("modules.pipeline.transcription.log") as mock_log,
        ):
            self.assertIsNone(transcription._detect_and_separate_vocals("vid.mp4", MagicMock(), MagicMock()))
            mock_log.assert_any_call("  [Sep] Warning: Separation failed (audio_separator). Using original audio.", "WARNING")

    def test_prepare_audio_falls_back_to_staged_plain_audio(self):
        mm = MagicMock()
        stage = MagicMock()
        stage.import_file.return_value = "/stage/vid_temp.wav"
        with (
            patch("modules.configuration.config.USE_VOCAL_SEPARATION", True),
            patch("modules.pipeline.transcription._detect_and_separate_vocals", return_value=None),
            patch("modules.utils.extract_clean_audio", return_value=self.work_audio),
        ):
            self.assertEqual(transcription._prepare_audio("vid.mp4", mm, stage), (self.work_audio, "/stage/vid_temp.wav"))
        mm.offload_separator.assert_called_once()
        stage.import_file.assert_called_once_with(self.work_audio)

    def test_prepare_audio_uses_separated_vocals(self):
        mm = MagicMock()
        stage = MagicMock()
        vocals = ("work/v.wav", "/stage/v.wav")
        with (
            patch("modules.configuration.config.USE_VOCAL_SEPARATION", True),
            patch("modules.pipeline.transcription._detect_and_separate_vocals", return_value=vocals),
            patch("modules.utils.extract_clean_audio") as extract,
        ):
            self.assertEqual(transcription._prepare_audio("vid.mp4", mm, stage), vocals)
        extract.assert_not_called()
        stage.import_file.assert_not_called()

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
            patch("modules.utils.extract_clean_audio", return_value=self.work_audio),
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
            patch("modules.utils.extract_clean_audio", return_value=self.work_audio),
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
            patch("modules.utils.extract_clean_audio", return_value=self.work_audio),
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
            patch("modules.utils.extract_clean_audio", return_value=self.work_audio),
            patch("modules.pipeline.transcription.log") as mock_log,
        ):
            transcription.transcribe_video_audio("vid.mp4", mm, forced_lang="ro")
            mock_log.assert_any_call("  [Whisper] Config: Forced Language='ro'")


if __name__ == "__main__":
    unittest.main()
