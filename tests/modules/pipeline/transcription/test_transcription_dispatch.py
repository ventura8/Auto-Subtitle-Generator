"""Tests for ASR engine dispatch, language vote, routing and Whisper fallback in transcription.py."""

import os
import unittest
from unittest.mock import MagicMock, call, patch

from modules.asr import pipeline as asr_pipeline
from modules.asr.language_id import LanguageVote
from modules.configuration import asr_settings
from modules.models import Segment
from modules.pipeline import transcription

WORK_DIR = os.path.join("folder", "vid.asg-temp")
AUDIO = os.path.join(WORK_DIR, "vid_temp.wav")
SPANS = [(0.0, 10.0), (12.0, 20.0)]


def _vote(language="ro", probability=0.9, share=1.0):
    """A language vote as language_id.detect_language returns it."""
    return LanguageVote(language, probability, share, {language: probability})


def _whisper_result(language="ro"):
    """Whisper transcribe() return value with one segment."""
    info = MagicMock(duration=20.0, language=language, language_probability=0.95)
    return [MagicMock(start=0.0, end=1.0, text="Whisper text", avg_logprob=-0.1)], info


class _DispatchCase(unittest.TestCase):
    """Common patches: no real audio, VAD, language ID or model is ever touched."""

    def setUp(self):
        self.addCleanup(asr_settings.set_cli_override, None)
        self.addCleanup(asr_settings.reset)
        asr_settings.reset()
        self.mm = MagicMock()
        self.mm.get_whisper.return_value.transcribe.return_value = _whisper_result()
        self.source = MagicMock()
        self.log = self._patch("modules.pipeline.transcription.log")
        self._patch("modules.pipeline.transcription._prepare_audio", return_value=AUDIO)
        self._patch("modules.pipeline.transcription.utils.print_progress_bar")
        self._patch("modules.pipeline.transcription.log_vram")
        self.open_source = self._patch("modules.pipeline.transcription.audio_source.open_pcm_source", return_value=self.source)
        self.spans = self._patch("modules.pipeline.transcription.audio_source.iter_speech_spans", return_value=list(SPANS))
        self.detect = self._patch("modules.pipeline.transcription.language_id.detect_language", return_value=_vote())
        self.asr_transcribe = self._patch("modules.pipeline.transcription.asr_pipeline.transcribe")
        self.asr_transcribe.return_value = asr_pipeline.AsrResult([(12.0, 14.0, "Doi."), (0.0, 2.0, "Unu.")], 18.0, 18.0, {})
        self.shortfall = self._patch("modules.pipeline.transcription.asr_gpu_shortfall", return_value=None)
        self.batch = self._patch("modules.pipeline.transcription.apply_dynamic_asr_batch", return_value=4)

    def _patch(self, target, **kwargs):
        """Start a patch that is undone after the test."""
        patcher = patch(target, **kwargs)
        mocked = patcher.start()
        self.addCleanup(patcher.stop)
        return mocked

    def _run(self, engine, forced_lang=None):
        """Transcribe ``vid.mp4`` with ``engine`` requested on the command line."""
        asr_settings.set_cli_override(engine)
        return transcription.transcribe_video_audio(os.path.join("folder", "vid.mp4"), self.mm, forced_lang)

    def _warnings(self):
        """All WARNING messages logged so far."""
        return [entry.args[0] for entry in self.log.call_args_list if entry.args[1:] == ("WARNING",)]


class TestWhisperEngine(_DispatchCase):
    def test_default_whisper_never_decodes_or_votes(self):
        segments, language, path = self._run(None)
        self.assertEqual((language, path), ("ro", AUDIO))
        self.assertEqual(len(segments), 1)
        self.open_source.assert_not_called()
        self.detect.assert_not_called()
        self.mm.offload_whisper.assert_called_once_with()
        self.mm.offload_asr.assert_called_once_with()

    def test_force_detected_language_passes_the_vote_to_whisper(self):
        asr_settings._STATE["force_detected_language"] = True
        self._run("whisper")
        self.assertEqual(self.mm.get_whisper.return_value.transcribe.call_args.kwargs["language"], "ro")
        self.source.close.assert_called_once_with()
        self.mm.get_asr.assert_not_called()

    def test_romanian_whisper_text_is_folded(self):
        self.mm.get_whisper.return_value.transcribe.return_value = (
            [Segment(0.0, 1.0, "Aşa"), Segment(1.0, 2.0, "da")],
            MagicMock(duration=2.0, language="ro", language_probability=0.9),
        )
        segments, _language, _path = self._run(None)
        self.assertEqual([segment.text for segment in segments], ["Așa", "da"])


class TestNvidiaEngines(_DispatchCase):
    def test_canary_with_forced_language_skips_the_vote(self):
        segments, language, _path = self._run("canary", forced_lang="RO")
        self.assertEqual(language, "ro")
        self.detect.assert_not_called()
        self.assertEqual(self.mm.method_calls[:2], [call.offload_whisper(), call.get_asr("canary")])
        self.assertEqual([segment.text for segment in segments], ["Unu.", "Doi."])
        request = self.asr_transcribe.call_args.args[3]
        self.assertEqual((request.language, request.batch_size), ("ro", 4))
        self.open_source.assert_called_once_with(AUDIO, os.path.join(WORK_DIR, "vid_asr16k.wav"))
        self.source.close.assert_called_once_with()

    def test_parakeet_with_detected_language(self):
        self._run("parakeet")
        self.detect.assert_called_once_with(self.mm.get_whisper.return_value, self.source, SPANS)
        self.mm.get_asr.assert_called_once_with("parakeet")
        self.shortfall.assert_not_called()

    def test_unsupported_language_falls_back_to_whisper(self):
        self.detect.return_value = _vote("ja")
        self._run("canary")
        self.mm.get_asr.assert_not_called()
        self.assertEqual(self.mm.get_whisper.return_value.transcribe.call_args.kwargs["language"], "ja")
        self.assertIn("  [ASR] Using Whisper instead of canary: canary does not support 'ja'.", self._warnings())

    def test_invalid_forced_language_is_voted_instead(self):
        self._run("canary", forced_lang="Romanian")
        self.detect.assert_called_once()
        self.assertTrue(any("'Romanian' is not a two-letter" in message for message in self._warnings()))

    def test_no_speech_means_no_language_and_whisper(self):
        self.spans.return_value = []
        self._run("canary")
        self.detect.assert_not_called()
        self.assertIn("  [ASR] Using Whisper instead of canary: no language detected.", self._warnings())
        self.assertIsNone(self.mm.get_whisper.return_value.transcribe.call_args.kwargs["language"])

    def test_unscored_vote_and_low_confidence_are_reported(self):
        self.detect.return_value = None
        self._run("canary")
        self.assertIn("  [ASR] Language ID could not score the audio; leaving the language to Whisper.", self._warnings())
        self.detect.return_value = _vote(probability=0.3)
        self._run("canary")
        self.assertIn("  [Warning] Low language confidence (0.30).", self._warnings())

    def test_small_whisper_model_warns_about_language_id(self):
        with patch("modules.configuration.config.WHISPER_MODEL_SIZE", "small"):
            self._run("canary")
        self.assertTrue(any("Language ID runs on Whisper 'small'" in message for message in self._warnings()))

    def test_load_failure_falls_back_to_whisper(self):
        self.mm.get_asr.side_effect = OSError("no such revision")
        segments, _language, _path = self._run("canary")
        self.assertEqual(segments[0].text, "Whisper text")
        self.assertIn("  [ASR] Could not load Canary (no such revision); falling back to Whisper.", self._warnings())

    def test_engine_failure_guard_redoes_the_file_with_whisper(self):
        self.asr_transcribe.return_value = asr_pipeline.AsrResult([], 100.0, 5.0, {"empty": 9})
        segments, language, _path = self._run("canary")
        self.assertEqual((segments[0].text, language), ("Whisper text", "ro"))
        self.mm.offload_asr.assert_called()
        self.assertTrue(any("redoing it with Whisper in 'ro'" in message for message in self._warnings()))
        self.assertEqual(self.mm.get_whisper.return_value.transcribe.call_args.kwargs["language"], "ro")

    def test_pipeline_errors_propagate_and_offload(self):
        self.asr_transcribe.side_effect = RuntimeError("CUDA error: device-side assert")
        with self.assertRaises(RuntimeError):
            self._run("canary")
        self.log.assert_any_call("Transcription failed: CUDA error: device-side assert", "ERROR")
        self.mm.offload_whisper.assert_called()
        self.mm.offload_asr.assert_called()
        self.source.close.assert_called_once_with()

    def test_filters_and_hallucinations_are_summarised(self):
        self.asr_transcribe.return_value = asr_pipeline.AsrResult(
            [(0.0, 2.0, "Thanks for watching"), (3.0, 5.0, "Bună.")], 10.0, 10.0, {"low confidence": 2, "empty": 0}
        )
        with patch("modules.configuration.config.HALLUCINATION_PHRASES", ["thanks for watching"]):
            segments, _language, _path = self._run("canary")
        self.assertEqual([segment.text for segment in segments], ["Bună."])
        self.assertIn("  [Canary] Filtered: 1 known phrase, 2 low confidence.", self._warnings())


class TestAutoEngine(_DispatchCase):
    def test_routed_language_uses_canary(self):
        self._run("auto")
        self.mm.get_asr.assert_called_once_with("canary")
        self.shortfall.assert_called_once_with("canary")

    def test_unrouted_language_reuses_whisper_without_warning(self):
        self.detect.return_value = _vote("en")
        self._run("auto")
        self.mm.get_asr.assert_not_called()
        self.assertEqual(self.mm.get_whisper.call_count, 2)
        self.mm.offload_whisper.assert_called_once_with()
        self.assertEqual(self._warnings(), [])

    def test_forced_unrouted_language_never_opens_the_audio(self):
        self._run("auto", forced_lang="en")
        self.open_source.assert_not_called()
        self.assertEqual(self.mm.get_whisper.return_value.transcribe.call_args.kwargs["language"], "en")

    def test_vram_shortfall_prefers_whisper_on_the_gpu(self):
        self.shortfall.return_value = "only 1.0 GB of VRAM free for nvidia/canary-1b-v2"
        self._run("auto")
        self.mm.get_asr.assert_not_called()
        self.assertIn("  [ASR] Using Whisper instead of Canary: only 1.0 GB of VRAM free for nvidia/canary-1b-v2.", self._warnings())


class TestWorkDirGuard(_DispatchCase):
    def test_audio_outside_the_work_directory_is_refused(self):
        with patch("modules.pipeline.transcription._prepare_audio", return_value=os.path.join("elsewhere", "a.wav")):
            with self.assertRaisesRegex(ValueError, "must come from the work directory"):
                self._run("canary")
        self.open_source.assert_not_called()


if __name__ == "__main__":
    unittest.main()
