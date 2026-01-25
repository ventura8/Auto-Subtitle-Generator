import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure modules can be imported
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)


class TestTranscription(unittest.TestCase):
    def setUp(self):
        global transcription
        from modules import transcription

        # Ensure OPTIMIZER has real values
        transcription.OPTIMIZER.vram_gb = 0
        transcription.OPTIMIZER.cpu_cores = 8
        transcription.OPTIMIZER.config["whisper_beam"] = 5

    @patch("modules.transcription.utils.extract_clean_audio", return_value="vocals.wav")
    @patch("modules.models.ModelManager")
    def test_transcribe_video_audio_success(self, mock_mm, mock_extract):
        mock_whisper = mock_mm.return_value.get_whisper.return_value
        mock_whisper.transcribe.return_value = (
            [MagicMock(start=0, end=1, text="Hello", avg_logprob=-0.1)],
            MagicMock(language="en", language_probability=0.99, duration=10.0)
        )

        segments, lang, _ = transcription.transcribe_video_audio("video.mp4", mock_mm.return_value)

        self.assertEqual(len(segments), 1)
        self.assertEqual(lang, "en")
        mock_whisper.transcribe.assert_called()

    @patch("modules.transcription.utils.extract_clean_audio", return_value="vocals.wav")
    @patch("modules.models.ModelManager")
    def test_transcribe_video_audio_oom_retry(self, mock_mm, mock_extract):
        mock_whisper = mock_mm.return_value.get_whisper.return_value
        # Raise OOM once, then succeed
        mock_whisper.transcribe.side_effect = [
            RuntimeError("CUDA out of memory"),
            ([MagicMock(start=0, end=1, text="Hello", avg_logprob=-0.1)],
             MagicMock(language="en", language_probability=0.99, duration=10.0))
        ]

        segments, lang, _ = transcription.transcribe_video_audio("video.mp4", mock_mm.return_value)
        self.assertEqual(len(segments), 1)
        # Should have called twice
        self.assertEqual(mock_whisper.transcribe.call_count, 2)


if __name__ == "__main__":
    unittest.main()
