import unittest
from unittest.mock import MagicMock, patch


class TestTranscription(unittest.TestCase):
    def setUp(self):
        global transcription
        from modules.pipeline import transcription

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

    @patch("modules.pipeline.transcription.utils.extract_clean_audio", return_value="vocals.wav")
    @patch("modules.models.ModelManager")
    def test_transcribe_video_audio_success(self, mock_mm, mock_extract):
        mock_whisper = mock_mm.return_value.get_whisper.return_value
        mock_whisper.transcribe.return_value = (
            [MagicMock(start=0, end=1, text="Hello", avg_logprob=-0.1)],
            MagicMock(language="en", language_probability=0.99, duration=10.0),
        )

        segments, lang, _ = transcription.transcribe_video_audio("video.mp4", mock_mm.return_value)

        self.assertEqual(len(segments), 1)
        self.assertEqual(lang, "en")
        mock_whisper.transcribe.assert_called()

    @patch("modules.pipeline.transcription.utils.extract_clean_audio", return_value="vocals.wav")
    @patch("modules.models.ModelManager")
    def test_transcribe_video_audio_oom_retry(self, mock_mm, mock_extract):
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


if __name__ == "__main__":
    unittest.main()
