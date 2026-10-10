"""Tests for the streaming PCM readers and block VAD in modules.asr.audio_source.

numpy and soundfile belong to the ``ml`` dependency group, which the CI test
jobs do not install, so the readers are exercised through fakes: a fake "WAV"
is a text file holding ``rate channels frames`` and audio is a plain list.
One class runs against the real libraries when they are importable.
"""

import importlib
import importlib.util
import itertools
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from modules.asr import audio_source
from modules.safe_io import SymlinkRefusedError

SR = audio_source.SAMPLE_RATE
_REAL_AUDIO_LIBS = all(importlib.util.find_spec(name) is not None for name in ("numpy", "soundfile"))


def _write_fake_wav(path, seconds, rate=SR, channels=1):
    """Write a fake WAV header file the fake soundfile module understands."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{rate} {channels} {int(seconds * rate)}")


def _read_header(path):
    """Parse a fake WAV header; libsndfile-style RuntimeError for anything else."""
    with open(path, encoding="utf-8", errors="replace") as handle:
        parts = handle.read().split()
    try:
        rate, channels, frames = (int(part) for part in parts)
    except ValueError as error:
        raise RuntimeError(f"Error opening {path!r}: Format not recognised.") from error
    return rate, channels, frames


class _FakeHandle:
    """Seekable fake ``SoundFile`` whose sample ``n`` has the value ``n``."""

    def __init__(self, frames):
        self.frames = frames
        self.position = 0
        self.read_kwargs = None
        self.closed = False

    def seek(self, frame):
        """Move to ``frame``."""
        self.position = frame

    def read(self, count, **kwargs):
        """Return ``count`` samples from the current position."""
        self.read_kwargs = kwargs
        end = min(self.position + count, self.frames)
        samples = list(range(self.position, end))
        self.position = end
        return samples

    def close(self):
        """Mark the handle closed."""
        self.closed = True


class _FakeSoundFile:
    """Stand-in for the ``soundfile`` module over fake WAV header files."""

    handles: list = []

    @staticmethod
    def info(path):
        """Return libsndfile-style metadata for a fake WAV."""
        rate, channels, frames = _read_header(path)
        return SimpleNamespace(samplerate=rate, channels=channels, duration=frames / rate)

    @classmethod
    def SoundFile(cls, path, mode="r"):
        """Open a fake WAV for reading."""
        if mode != "r":
            raise ValueError(mode)
        handle = _FakeHandle(_read_header(path)[2])
        cls.handles.append(handle)
        return handle


def _speech_audio(seconds, speech):
    """Return silence of ``seconds`` with 1.0 samples over each ``(start, end)`` in ``speech``."""
    audio = [0.0] * int(seconds * SR)
    for start, end in speech:
        first, last = int(round(start * SR)), int(round(end * SR))
        audio[first:last] = [1.0] * (last - first)
    return audio


def _runs(audio):
    """Return ``(start, end)`` sample runs of non-zero audio."""
    runs, index = [], 0
    for speech, group in itertools.groupby(audio, key=bool):
        length = sum(1 for _sample in group)
        if speech:
            runs.append((index, index + length))
        index += length
    return runs


class _FakeVad:
    """Stand-in for ``faster_whisper.vad``: speech is wherever the audio is non-zero."""

    def __init__(self):
        self.calls = []

    @staticmethod
    def VadOptions(**kwargs):
        """Record the options as a namespace."""
        return SimpleNamespace(**kwargs)

    def get_speech_timestamps(self, audio, options):
        """Return non-zero runs at least ``min_speech_duration_ms`` long, in samples."""
        self.calls.append((len(audio), options))
        min_samples = getattr(options, "min_speech_duration_ms", 0) * SR / 1000
        return [{"start": start, "end": end} for start, end in _runs(audio) if end - start >= min_samples]


def _rounded(spans):
    """Round span times to milliseconds for comparison."""
    return [(round(start, 3), round(end, 3)) for start, end in spans]


class TestAccessorsAndSettings(unittest.TestCase):
    def test_lazy_modules(self):
        with patch("modules.asr.audio_source.importlib.import_module", return_value="module") as load:
            self.assertEqual(audio_source._soundfile(), "module")
        load.assert_called_once_with("soundfile")
        self.assertIs(audio_source._vad_module(), sys.modules["faster_whisper.vad"])

    def test_decode_audio_uses_faster_whisper_at_16k(self):
        with patch.object(sys.modules["faster_whisper"], "decode_audio", return_value="pcm") as decode:
            self.assertEqual(audio_source._decode_audio("a.wav"), "pcm")
        decode.assert_called_once_with("a.wav", sampling_rate=SR)

    def test_asr_vad_settings(self):
        settings = audio_source.asr_vad_settings(15.0, 500)
        self.assertEqual(settings, audio_source.VadSettings(0.5, 250, 500, 300, 15.0))


class TestArraySource(unittest.TestCase):
    def test_read_clamps_to_the_source(self):
        source = audio_source._ArraySource(list(range(SR)))
        self.assertEqual(source.duration, 1.0)
        self.assertEqual(len(source.read(-1.0, 0.5)), SR // 2)
        self.assertEqual(source.read(0.9, 5.0)[0], 0.9 * SR)
        self.assertEqual(len(source.read(0.9, 5.0)), SR // 10)
        self.assertEqual(len(source.read(0.8, 0.2)), 0)

    def test_close_drops_the_samples(self):
        source = audio_source._ArraySource([1.0] * SR)
        source.close()
        self.assertEqual(len(source.read(0.0, 1.0)), 0)

    def test_base_class_requires_a_reader(self):
        source = audio_source.PcmSource(SR)
        source.close()
        with self.assertRaises(NotImplementedError):
            source.read(0.0, 1.0)


class _AudioDirTestCase(unittest.TestCase):
    """A temporary work directory plus the fake soundfile module."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name
        self.audio = os.path.join(self.dir, "stem.wav")
        self.transcode = os.path.join(self.dir, "stem_asr16k.wav")
        patcher = patch("modules.asr.audio_source._soundfile", return_value=_FakeSoundFile)
        patcher.start()
        self.addCleanup(patcher.stop)


class TestOpenPcmSource(_AudioDirTestCase):
    def test_16k_mono_file_is_read_by_seeking(self):
        _write_fake_wav(self.audio, 1.0)
        source = audio_source.open_pcm_source(self.audio, self.transcode)
        self.assertIsInstance(source, audio_source._SoundFileSource)
        self.assertEqual(source.duration, 1.0)
        self.assertEqual(source.read(0.5, 0.5 + 4 / SR), [8000, 8001, 8002, 8003])
        handle = _FakeSoundFile.handles[-1]
        self.assertEqual(handle.read_kwargs, {"dtype": "float32", "always_2d": False})
        source.close()
        self.assertTrue(handle.closed)

    def test_short_other_format_is_decoded_in_memory(self):
        _write_fake_wav(self.audio, 0.5, rate=8000, channels=2)
        with patch("modules.asr.audio_source._decode_audio", return_value=[0.0] * SR) as decode:
            source = audio_source.open_pcm_source(self.audio, self.transcode)
        decode.assert_called_once_with(self.audio)
        self.assertIsInstance(source, audio_source._ArraySource)
        self.assertFalse(os.path.exists(self.transcode))

    def test_unreadable_file_falls_back_to_ffprobe_duration(self):
        with open(self.audio, "wb") as handle:
            handle.write(b"not audio")
        with (
            patch("modules.asr.audio_source.ffmpeg_utils.get_audio_duration", return_value=3.0) as probe,
            patch("modules.asr.audio_source._decode_audio", return_value=[0.0] * (3 * SR)),
        ):
            source = audio_source.open_pcm_source(self.audio, self.transcode)
        probe.assert_called_once_with(self.audio)
        self.assertEqual(source.duration, 3.0)


def _fake_transcoder(_source, target, start, length, mono_16k=False):
    """Emulate ``write_audio_window`` by writing a 16 kHz mono fake WAV of ``length`` s."""
    if not mono_16k or start != 0:
        raise AssertionError("expected a whole-file mono 16 kHz transcode")
    _write_fake_wav(target, length)


class TestTranscodeOnce(_AudioDirTestCase):
    def setUp(self):
        super().setUp()
        _write_fake_wav(self.audio, 2.0, rate=44100, channels=2)
        patcher = patch("modules.asr.audio_source.WHOLE_DECODE_MAX_SECONDS", 1.0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _open(self, writer=_fake_transcoder):
        """Open the long input with ``writer`` standing in for FFmpeg; return (source, writer mock, log mock)."""
        with (
            patch("modules.asr.audio_source.ffmpeg_utils.write_audio_window", side_effect=writer) as write,
            patch("modules.asr.audio_source.log") as log,
        ):
            source = audio_source.open_pcm_source(self.audio, self.transcode)
        return source, write, log

    def test_long_input_is_transcoded_once_into_the_work_dir(self):
        source, write, log = self._open()
        self.assertIsInstance(source, audio_source._SoundFileSource)
        self.assertEqual(source.duration, 2.0)
        self.assertEqual(write.call_args.args[0], self.audio)
        self.assertEqual(write.call_args.args[3], 2.0)
        self.assertEqual(log.call_args.args[1], "WARNING")
        self.assertEqual(sorted(os.listdir(self.dir)), ["stem.wav", "stem_asr16k.wav"])

    def test_finished_transcode_is_reused(self):
        _write_fake_wav(self.transcode, 2.2)
        source, write, log = self._open()
        self.assertAlmostEqual(source.duration, 2.2)
        write.assert_not_called()
        log.assert_not_called()

    def test_transcode_of_the_wrong_length_is_redone(self):
        _write_fake_wav(self.transcode, 1.0)
        source, write, _log = self._open()
        write.assert_called_once()
        self.assertEqual(source.duration, 2.0)

    def test_transcode_in_the_wrong_format_is_redone(self):
        _write_fake_wav(self.transcode, 2.0, rate=44100)
        _source, write, _log = self._open()
        write.assert_called_once()

    def test_failed_transcode_leaves_nothing_behind(self):
        with self.assertRaises(RuntimeError):
            self._open(writer=RuntimeError("ffmpeg failed"))
        self.assertEqual(os.listdir(self.dir), ["stem.wav"])

    def test_symlinked_transcode_is_refused(self):
        decoy = os.path.join(self.dir, "decoy.wav")
        _write_fake_wav(decoy, 2.0)
        os.symlink(decoy, self.transcode)
        with self.assertRaises(SymlinkRefusedError):
            self._open()
        self.assertTrue(os.path.islink(self.transcode))
        self.assertEqual(sorted(os.listdir(self.dir)), ["decoy.wav", "stem.wav", "stem_asr16k.wav"])


@unittest.skipUnless(_REAL_AUDIO_LIBS, "numpy and soundfile come with the ml dependency group")
class TestRealSoundFile(unittest.TestCase):
    def test_real_16k_mono_wav_is_read_by_seeking(self):
        numpy = importlib.import_module("numpy")
        soundfile = importlib.import_module("soundfile")
        ramp = numpy.linspace(-1.0, 1.0, SR, dtype=numpy.float32)
        with tempfile.TemporaryDirectory() as work_dir:
            path = os.path.join(work_dir, "x_temp.wav")
            soundfile.write(path, ramp, SR, subtype="FLOAT", format="WAV")
            source = audio_source.open_pcm_source(path, os.path.join(work_dir, "unused.wav"))
            window = source.read(0.5, 0.75)
            source.close()
        self.assertEqual(window.dtype, numpy.float32)
        self.assertEqual(window.ndim, 1)
        self.assertEqual(window.tolist(), ramp[SR // 2 : SR * 3 // 4].tolist())


class TestIterSpeechSpans(unittest.TestCase):
    def _spans(self, seconds, speech, max_speech=15.0, block=10.0):
        """Run the block VAD over synthetic audio; return (rounded spans, fake VAD)."""
        fake = _FakeVad()
        source = audio_source._ArraySource(_speech_audio(seconds, speech))
        with patch("modules.asr.audio_source._vad_module", return_value=fake):
            spans = audio_source.iter_speech_spans(source, audio_source.asr_vad_settings(max_speech, 500), block_seconds=block)
        return _rounded(spans), fake

    def test_span_crossing_a_block_edge_is_carried_whole(self):
        spans, fake = self._spans(30.0, [(5.0, 15.0)], max_speech=20.0)
        self.assertEqual(spans, [(5.0, 15.0)])
        self.assertEqual([length for length, _options in fake.calls], [10 * SR, 15 * SR, 11 * SR])

    def test_carried_span_is_still_split_at_the_cap(self):
        spans, _fake = self._spans(30.0, [(5.0, 15.0)], max_speech=6.0)
        self.assertEqual(spans, [(5.0, 10.0), (10.0, 15.0)])

    def test_vad_options_come_from_the_settings(self):
        _spans, fake = self._spans(5.0, [(1.0, 2.0)])
        options = fake.calls[0][1]
        self.assertEqual(
            vars(options),
            {
                "threshold": 0.5,
                "min_speech_duration_ms": 250,
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 300,
                "max_speech_duration_s": 15.0,
            },
        )

    def test_short_onset_cut_by_an_edge_is_recovered(self):
        spans, fake = self._spans(20.0, [(9.9, 10.5)])
        self.assertEqual(spans, [(9.9, 10.5)])
        self.assertEqual(fake.calls[1][0], 11 * SR)

    def test_next_block_starts_after_the_last_closed_span(self):
        spans, fake = self._spans(20.0, [(2.0, 9.5)])
        self.assertEqual(spans, [(2.0, 9.5)])
        self.assertEqual(fake.calls[1][0], int(10.5 * SR))

    def test_span_as_long_as_a_block_is_not_carried(self):
        spans, fake = self._spans(25.0, [(0.0, 25.0)], max_speech=30.0)
        self.assertEqual(spans, [(0.0, 25.0)])
        self.assertEqual([length for length, _options in fake.calls], [10 * SR, 10 * SR, 5 * SR])

    def test_silence_and_empty_sources_have_no_spans(self):
        self.assertEqual(self._spans(12.0, [])[0], [])
        self.assertEqual(self._spans(0.0, [])[0], [])


class TestTidySpans(unittest.TestCase):
    def test_short_gaps_merge_within_the_cap(self):
        self.assertEqual(audio_source._tidy_spans([(1.2, 2.0), (0.0, 1.0)], 15.0), [(0.0, 2.0)])

    def test_long_gaps_and_the_cap_keep_spans_apart(self):
        self.assertEqual(audio_source._tidy_spans([(0.0, 1.0), (1.5, 2.0)], 15.0), [(0.0, 1.0), (1.5, 2.0)])
        self.assertEqual(audio_source._tidy_spans([(0.0, 5.0), (5.1, 9.0)], 8.0), [(0.0, 5.0), (5.1, 9.0)])

    def test_slivers_are_dropped(self):
        self.assertEqual(audio_source._tidy_spans([(0.0, 0.2), (3.0, 4.0)], 15.0), [(3.0, 4.0)])

    def test_overlong_spans_split_into_equal_pieces(self):
        pieces = audio_source._tidy_spans([(0.0, 20.0)], 8.0)
        self.assertEqual(_rounded(pieces), [(0.0, 6.667), (6.667, 13.333), (13.333, 20.0)])
        self.assertEqual(pieces[-1][1], 20.0)


class TestInternalPauses(unittest.TestCase):
    def test_gaps_between_speech_parts_are_absolute(self):
        fake = _FakeVad()
        audio = _speech_audio(1.0, [(0.0, 0.1), (0.2, 0.4)])
        with patch("modules.asr.audio_source._vad_module", return_value=fake):
            pauses = audio_source.internal_pauses(audio, 10.0)
        self.assertEqual(_rounded(pauses), [(10.1, 10.2)])
        self.assertEqual(vars(fake.calls[0][1]), {"threshold": 0.5, "min_silence_duration_ms": 150, "speech_pad_ms": 30})

    def test_touching_parts_leave_no_pause(self):
        fake = SimpleNamespace(
            VadOptions=_FakeVad.VadOptions,
            get_speech_timestamps=lambda _audio, _options: [{"start": 0, "end": 1600}, {"start": 1600, "end": 3200}],
        )
        with patch("modules.asr.audio_source._vad_module", return_value=fake):
            self.assertEqual(audio_source.internal_pauses([0.0] * 3200, 0.0), [])


if __name__ == "__main__":
    unittest.main()
