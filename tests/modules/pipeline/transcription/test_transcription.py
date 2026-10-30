import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from modules.configuration import asr_settings


class TestTranscription(unittest.TestCase):
    def setUp(self):
        global transcription
        from modules.pipeline import transcription

        # The legacy Whisper path is under test; the shipped default routes per language.
        asr_settings.set_cli_override("whisper")
        self.addCleanup(asr_settings.set_cli_override, None)

        original_vram_gb = transcription.OPTIMIZER.vram_gb
        original_cpu_cores = transcription.OPTIMIZER.cpu_cores
        original_whisper_beam = transcription.OPTIMIZER.config["whisper_beam"]

        self.addCleanup(setattr, transcription.OPTIMIZER, "vram_gb", original_vram_gb)
        self.addCleanup(setattr, transcription.OPTIMIZER, "cpu_cores", original_cpu_cores)
        self.addCleanup(transcription.OPTIMIZER.config.__setitem__, "whisper_beam", original_whisper_beam)

        # Ensure OPTIMIZER has real values
        transcription.OPTIMIZER.vram_gb = 0
        transcription.OPTIMIZER.cpu_cores = 8
        transcription.OPTIMIZER.config["whisper_beam"] = 5

        # extract_clean_audio is mocked, but its result is really staged, so it must exist.
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.work_audio = os.path.join(self._tmp.name, "video_temp.wav")
        with open(self.work_audio, "wb") as handle:
            handle.write(b"RIFF-work-audio")
        patcher = patch("modules.pipeline.transcription.utils.extract_clean_audio", return_value=self.work_audio)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _assert_read_staged_copy(self, path):
        """The path Whisper read is a private staged copy of the work audio, not the work file."""
        self.assertNotEqual(path, self.work_audio)
        self.assertNotEqual(os.path.dirname(path), self._tmp.name)
        self.assertEqual(os.path.basename(path), "video_temp.wav")

    @patch("modules.configuration.config.USE_VOCAL_SEPARATION", False)
    @patch("modules.models.ModelManager")
    def test_transcribe_video_audio_success(self, mock_mm):
        mock_whisper = mock_mm.return_value.get_whisper.return_value
        seen = {}

        def transcribe(path, **_kwargs):
            with open(path, "rb") as handle:
                seen[path] = handle.read()
            return (
                [MagicMock(start=0, end=1, text="Hello", avg_logprob=-0.1)],
                MagicMock(language="en", language_probability=0.99, duration=10.0),
            )

        mock_whisper.transcribe.side_effect = transcribe

        segments, lang, transcribe_path = transcription.transcribe_video_audio("video.mp4", mock_mm.return_value)

        self.assertEqual(len(segments), 1)
        self.assertEqual(lang, "en")
        # The work-directory path is still what callers get back ...
        self.assertEqual(transcribe_path, self.work_audio)
        # ... but Whisper read a staged copy of the same bytes, removed once transcription ended.
        ((staged, data),) = seen.items()
        self._assert_read_staged_copy(staged)
        self.assertEqual(data, b"RIFF-work-audio")
        self.assertFalse(os.path.exists(os.path.dirname(staged)))

    @patch("modules.configuration.config.USE_VOCAL_SEPARATION", False)
    @patch("modules.models.ModelManager")
    def test_transcribe_video_audio_oom_retry(self, mock_mm):
        mock_whisper = mock_mm.return_value.get_whisper.return_value
        # Raise OOM once, then succeed
        mock_whisper.transcribe.side_effect = [
            RuntimeError("CUDA out of memory"),
            (
                [MagicMock(start=0, end=1, text="Hello", avg_logprob=-0.1)],
                MagicMock(language="en", language_probability=0.99, duration=10.0),
            ),
        ]

        segments, lang, _ = transcription.transcribe_video_audio("video.mp4", mock_mm.return_value)
        self.assertEqual(len(segments), 1)
        # Should have called twice
        self.assertEqual(mock_whisper.transcribe.call_count, 2)
        self.assertEqual(mock_whisper.transcribe.call_args_list[1].kwargs["beam_size"], 2)
        # Both attempts read the same staged copy.
        first, second = (call.args[0] for call in mock_whisper.transcribe.call_args_list)
        self.assertEqual(first, second)
        self._assert_read_staged_copy(first)


if __name__ == "__main__":
    unittest.main()
