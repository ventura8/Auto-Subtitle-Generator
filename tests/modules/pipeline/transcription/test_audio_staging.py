"""transcribe_video_audio hands every model a private staged copy of the work-directory audio.

The work directory sits inside the input folder, whose owner can keep writing
into it while a model loads. These tests swap or plant links there *after* the
audio was staged (from inside the mocked model loads) and prove that no reader
or writer ever goes through them.
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from modules import workdir
from modules.asr import pipeline as asr_pipeline
from modules.asr.language_id import LanguageVote
from modules.configuration import asr_settings
from modules.media import audio_stage
from modules.media.input_binding import InputRefusedError
from modules.pipeline import transcription

ORIGINAL = b"\x00ORIGINAL-AUDIO" * 32
VICTIM = b"VICTIM-RECORDING"


def _whisper_result():
    """Whisper transcribe() return value with one segment."""
    info = MagicMock(duration=20.0, language="en", language_probability=0.95)
    return [MagicMock(start=0.0, end=1.0, text="Hello", avg_logprob=-0.1)], info


class _StagingCase(unittest.TestCase):
    """A real input folder and work directory; FFmpeg, the separator and the models are stand-ins."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(asr_settings.set_cli_override, None)
        self.addCleanup(asr_settings.reset)
        asr_settings.reset()
        self.folder = self._tmp.name
        self.video = os.path.join(self.folder, "movie.mp4")
        self._write(self.video, b"video")
        self.work_dir = workdir.ensure_work_dir(self.folder, "movie")
        self.temp_wav = self._write(os.path.join(self.work_dir, "movie_temp.wav"), ORIGINAL)
        self.victim = self._write(os.path.join(self.folder, "victim.wav"), VICTIM)
        self.read_by_whisper = []
        self.mm = MagicMock()
        self.mm.get_whisper.return_value.transcribe.side_effect = self._whisper_reads
        for target, kwargs in (
            ("modules.pipeline.transcription.utils.extract_clean_audio", {"return_value": self.temp_wav}),
            ("modules.pipeline.transcription.utils.print_progress_bar", {}),
            ("modules.pipeline.transcription._probe_duration", {"return_value": 20.0}),
        ):
            patcher = patch(target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)
        log_patcher = patch("modules.pipeline.transcription.log")
        self.log = log_patcher.start()
        self.addCleanup(log_patcher.stop)

    def _write(self, path, payload):
        with open(path, "wb") as handle:
            handle.write(payload)
        return path

    def _read(self, path):
        with open(path, "rb") as handle:
            return handle.read()

    def _whisper_reads(self, path, **_kwargs):
        """Record the path Whisper was handed and the bytes it held at that moment."""
        self.read_by_whisper.append((path, self._read(path)))
        return _whisper_result()

    def _relink(self, link, target=None):
        """Replace ``link`` with a symlink to the victim (or ``target``)."""
        if os.path.lexists(link):
            os.remove(link)
        try:
            os.symlink(target or self.victim, link)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"symlinks unavailable: {error}")

    def _assert_outside_input_folder(self, path):
        folder = os.path.realpath(self.folder)
        self.assertNotEqual(os.path.commonpath([os.path.realpath(path), folder]), folder)


class TestWhisperReadsTheStagedCopy(_StagingCase):
    def test_temp_wav_swapped_during_model_load_is_never_read(self):
        asr_settings.set_cli_override("whisper")
        self.mm.get_whisper.side_effect = lambda: self._relink(self.temp_wav) or self.mm.get_whisper.return_value
        with patch("modules.configuration.config.USE_VOCAL_SEPARATION", False):
            _segments, _language, path = transcription.transcribe_video_audio(self.video, self.mm)
        ((staged, data),) = self.read_by_whisper
        self.assertEqual(data, ORIGINAL)
        self._assert_outside_input_folder(staged)
        self.assertFalse(os.path.lexists(staged), "the stage is removed after transcription")
        self.assertEqual(path, self.temp_wav)
        self.assertEqual(self._read(self.victim), VICTIM)

    def test_linked_temp_wav_is_refused_before_any_model_loads(self):
        asr_settings.set_cli_override("whisper")
        self._relink(self.temp_wav)
        with (
            patch("modules.configuration.config.USE_VOCAL_SEPARATION", False),
            self.assertRaises(InputRefusedError),
        ):
            transcription.transcribe_video_audio(self.video, self.mm)
        self.mm.get_whisper.assert_not_called()


class TestNvidiaReadsTheStagedCopy(_StagingCase):
    def test_pcm_source_opens_the_staged_copy_and_the_transcode_stays_in_the_work_dir(self):
        asr_settings.set_cli_override("canary")
        opened = []

        def open_pcm_source(path, transcode_path):
            opened.append((path, self._read(path), transcode_path))
            return MagicMock()

        real_resolve = transcription.routing.resolve

        def swap_then_resolve(*args):
            # Routing runs after staging and before the PCM source opens: swap the work file now.
            self._relink(self.temp_wav)
            return real_resolve(*args)

        with (
            patch("modules.pipeline.transcription.routing.resolve", side_effect=swap_then_resolve),
            patch("modules.configuration.config.USE_VOCAL_SEPARATION", False),
            patch("modules.pipeline.transcription.audio_source.open_pcm_source", side_effect=open_pcm_source),
            patch("modules.pipeline.transcription.audio_source.iter_speech_spans", return_value=[(0.0, 5.0)]),
            patch("modules.pipeline.transcription.language_id.detect_language", return_value=LanguageVote("ro", 0.9, 1.0, {"ro": 0.9})),
            patch(
                "modules.pipeline.transcription.asr_pipeline.transcribe",
                return_value=asr_pipeline.AsrResult([(0.0, 2.0, "Unu.")], 5.0, 5.0, {}),
            ),
            patch("modules.pipeline.transcription.asr_gpu_shortfall", return_value=None),
            patch("modules.pipeline.transcription.apply_dynamic_asr_batch", return_value=1),
            patch("modules.pipeline.transcription.log_vram"),
        ):
            segments, _language, path = transcription.transcribe_video_audio(self.video, self.mm, "ro")
        ((staged, data, transcode),) = opened
        self.assertEqual(data, ORIGINAL)
        self._assert_outside_input_folder(staged)
        self.assertEqual(transcode, os.path.join(self.work_dir, "movie_asr16k.wav"))
        self.assertEqual((segments[0].text, path), ("Unu.", self.temp_wav))
        self.assertEqual(self._read(self.victim), VICTIM)


class TestSeparationThroughTheStage(_StagingCase):
    def _manager_with_separator(self, during_load=None):
        """A separator that writes ``SEP:<input bytes>`` stems into the directory it is handed."""
        separator = MagicMock()
        seen = {}

        def separate(audio_input):
            seen["input"] = audio_input
            stem = os.path.join(seen["output_dir"], "movie_temp_(Vocals)_model.wav")
            return [self._write(stem, b"SEP:" + self._read(audio_input))]

        def get_separator(output_dir=None):
            seen["output_dir"] = output_dir
            if during_load:
                during_load()
            return separator

        separator.separate.side_effect = separate
        self.mm.get_separator.side_effect = get_separator
        return seen

    def _transcribe_with_separation(self):
        asr_settings.set_cli_override("whisper")
        with (
            patch("modules.configuration.config.USE_VOCAL_SEPARATION", True),
            patch("modules.configuration.config.SEPARATION_CHUNK_MINUTES", 0),
        ):
            return transcription.transcribe_video_audio(self.video, self.mm)

    def test_separator_reads_and_writes_only_inside_the_stage(self):
        seen = self._manager_with_separator()
        _segments, _language, path = self._transcribe_with_separation()
        stage_root = os.path.dirname(seen["output_dir"])
        self.assertTrue(os.path.basename(stage_root).startswith(audio_stage.STAGE_PREFIX))
        self.assertEqual(seen["input"], os.path.join(stage_root, "movie_temp.wav"))
        self._assert_outside_input_folder(seen["output_dir"])
        self.assertFalse(os.path.lexists(stage_root))
        # Whisper read the staged vocals; the work-dir copy is kept for resume and returned.
        ((staged, data),) = self.read_by_whisper
        self.assertEqual(staged, os.path.join(stage_root, "movie_temp_(Vocals)_model.wav"))
        self.assertEqual(data, b"SEP:" + ORIGINAL)
        self.assertEqual(path, os.path.join(self.work_dir, "movie_temp_(Vocals)_model.wav"))
        self.assertEqual(self._read(path), b"SEP:" + ORIGINAL)
        self.mm.offload_separator.assert_called_once_with()

    def test_temp_wav_swapped_during_separator_load_is_never_read(self):
        seen = self._manager_with_separator(during_load=lambda: self._relink(self.temp_wav))
        self._transcribe_with_separation()
        self._assert_outside_input_folder(seen["input"])
        ((_staged, data),) = self.read_by_whisper
        self.assertEqual(data, b"SEP:" + ORIGINAL)
        self.assertEqual(self._read(self.victim), VICTIM)

    def test_link_planted_at_the_vocal_stem_name_is_never_written_through(self):
        stem = os.path.join(self.work_dir, "movie_temp_(Vocals)_model.wav")
        self._manager_with_separator(during_load=lambda: self._relink(stem))
        # Keeping the stem for resume is refused; transcription still uses the staged vocals.
        _segments, _language, path = self._transcribe_with_separation()
        self.assertEqual(self._read(self.victim), VICTIM)
        self.assertTrue(os.path.islink(stem))
        self.assertEqual(path, stem)
        ((_staged, data),) = self.read_by_whisper
        self.assertEqual(data, b"SEP:" + ORIGINAL)
        warnings = [c.args[0] for c in self.log.call_args_list if c.args[1:] == ("WARNING",)]
        self.assertTrue(any("Not keeping" in message and "symlink" in message.lower() for message in warnings))

    def test_resumed_vocals_swapped_after_staging_are_never_read(self):
        stem = self._write(os.path.join(self.work_dir, "movie_temp_(Vocals)_model.wav"), b"RESUMED-VOCALS")
        asr_settings.set_cli_override("whisper")
        self.mm.get_whisper.side_effect = lambda: self._relink(stem) or self.mm.get_whisper.return_value
        with patch("modules.configuration.config.USE_VOCAL_SEPARATION", True):
            _segments, _language, path = transcription.transcribe_video_audio(self.video, self.mm)
        self.mm.get_separator.assert_not_called()
        self.assertEqual(path, stem)
        ((_staged, data),) = self.read_by_whisper
        self.assertEqual(data, b"RESUMED-VOCALS")
        self.assertEqual(self._read(self.victim), VICTIM)


if __name__ == "__main__":
    unittest.main()
