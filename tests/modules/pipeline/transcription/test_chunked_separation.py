"""Chunked vocal separation: long inputs are separated window by window, resumably, and joined."""

import importlib
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from modules import safe_io, workdir

transcription = importlib.import_module("modules.pipeline.transcription")


class TestChunkWindows(unittest.TestCase):
    def test_short_audio_is_one_window(self):
        self.assertEqual(transcription._chunk_windows(100.0, 1800), [(0.0, 100.0)])

    def test_exact_multiple_has_no_empty_tail(self):
        self.assertEqual(transcription._chunk_windows(3600.0, 1800), [(0.0, 1800), (1800.0, 1800)])

    def test_four_hours_at_thirty_minutes_is_eight_windows(self):
        windows = transcription._chunk_windows(4 * 3600.0, 1800)
        self.assertEqual(len(windows), 8)
        self.assertEqual(windows[0], (0.0, 1800))
        self.assertEqual(windows[-1], (7 * 1800.0, 1800))

    def test_short_tail_is_folded_into_previous_window(self):
        windows = transcription._chunk_windows(4 * 3600.0 + 5, 1800)
        self.assertEqual(len(windows), 8)
        self.assertEqual(windows[-1], (7 * 1800.0, 1805.0))
        self.assertAlmostEqual(sum(length for _, length in windows), 4 * 3600.0 + 5)

    def test_long_tail_keeps_its_own_window(self):
        windows = transcription._chunk_windows(1800.0 + 120, 1800)
        self.assertEqual(windows, [(0.0, 1800), (1800.0, 120.0)])

    def test_single_short_window_is_never_folded_away(self):
        self.assertEqual(transcription._chunk_windows(30.0, 1800), [(0.0, 30.0)])


class TestChunkSeconds(unittest.TestCase):
    def test_reads_config_in_minutes(self):
        with patch("modules.configuration.config.SEPARATION_CHUNK_MINUTES", 30):
            self.assertEqual(transcription._separation_chunk_seconds(), 1800)

    def test_zero_or_missing_disables_chunking(self):
        with patch("modules.configuration.config.SEPARATION_CHUNK_MINUTES", 0):
            self.assertEqual(transcription._separation_chunk_seconds(), 0)
        with patch("modules.configuration.config.SEPARATION_CHUNK_MINUTES", None):
            self.assertEqual(transcription._separation_chunk_seconds(), 0)


class _ChunkCase(unittest.TestCase):
    """Real work directory and scratch directory; FFmpeg and the separator are stand-ins that write files."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.folder = self._tmp.name
        self.work_dir = workdir.ensure_work_dir(self.folder, "movie")
        self.scratch_dir = safe_io.create_private_dir(self.work_dir, "movie_vocals")
        self.audio = self._write(os.path.join(self.work_dir, "movie_temp.wav"), b"\x00" * 4096)
        self.separator = MagicMock()
        self.separator.separate.side_effect = self._fake_separate
        self.job = {
            "separator": self.separator,
            "audio_path": self.audio,
            "scratch_dir": self.scratch_dir,
            "work_dir": self.work_dir,
            "base_name": "movie",
        }
        self.durations = {}
        self.addCleanup(patch.stopall)
        patch("modules.pipeline.transcription.ffmpeg_utils.write_audio_window", side_effect=self._fake_window).start()
        patch("modules.pipeline.transcription.ffmpeg_utils.concat_audio_files", side_effect=self._fake_concat).start()
        patch("modules.pipeline.transcription._probe_duration", side_effect=lambda p: self.durations.get(os.path.normpath(p), 0.0)).start()
        patch("modules.pipeline.transcription.log").start()

    def _work_entries(self):
        """Work-dir entries other than the fixture's own separator scratch directory."""
        return sorted(e for e in os.listdir(self.work_dir) if e != os.path.basename(self.scratch_dir))

    def _write(self, path, payload=b"\x00" * 64):
        with open(path, "wb") as handle:
            handle.write(payload)
        return path

    def _fake_window(self, source, target, start, length, mono_16k=False):
        self._write(target, f"window {start} {length} mono={mono_16k}".encode())
        self.durations[os.path.normpath(target)] = length

    def _fake_separate(self, chunk_input):
        stem = os.path.join(self.scratch_dir, os.path.basename(chunk_input).replace(".wav", "_(Vocals)_model.wav"))
        return [self._write(stem)]

    def _fake_concat(self, sources, target, list_path):
        self._write(list_path, b"list")
        joined = b""
        for source in sources:
            with open(source, "rb") as handle:
                joined += handle.read()
        self._write(target, joined)


class TestChunkedSeparation(_ChunkCase):
    def test_separates_each_window_and_joins_into_one_16k_stem(self):
        self.durations[os.path.normpath(self.audio)] = 150.0
        final = transcription._run_chunked_separation(self.job, 150.0, 60)

        self.assertEqual(final, os.path.join(self.work_dir, "movie_temp_(Vocals)_chunked.wav"))
        self.assertTrue(os.path.isfile(final))
        # 150 s at 60 s -> (0,60) (60,60) (120,30): the 30 s tail is < 60 so it folds into the second window.
        self.assertEqual(self.separator.separate.call_count, 2)
        # Chunk stems are removed after the join; only the source WAV and the final stem remain.
        self.assertEqual(self._work_entries(), ["movie_temp.wav", "movie_temp_(Vocals)_chunked.wav"])
        # Nothing was left inside the scratch directory either.
        self.assertEqual(os.listdir(self.scratch_dir), [])

    def test_chunk_input_is_padded_and_stem_is_trimmed_to_the_window(self):
        with patch("modules.pipeline.transcription.ffmpeg_utils.write_audio_window") as mock_window:
            mock_window.side_effect = self._fake_window
            transcription._separate_chunk(self.job, 1, 3, (60.0, 60.0))
        cut, trim = mock_window.call_args_list
        # Input cut: 2 s of context before and after the window, at the source format.
        self.assertEqual(cut.args[2:4], (58.0, 64.0))
        self.assertFalse(cut.kwargs.get("mono_16k", False))
        # Output trim: skip the leading pad, keep exactly the window, downmix to 16 kHz mono.
        self.assertEqual(trim.args[2:4], (2.0, 60.0))
        self.assertTrue(trim.kwargs["mono_16k"])
        self.assertNotEqual(trim.args[1], os.path.join(self.work_dir, "movie_sepchunk_001.wav"), "written via scratch, promoted after")
        self.assertTrue(os.path.isfile(os.path.join(self.work_dir, "movie_sepchunk_001.wav")))

    def test_first_chunk_has_no_leading_pad(self):
        with patch("modules.pipeline.transcription.ffmpeg_utils.write_audio_window") as mock_window:
            mock_window.side_effect = self._fake_window
            transcription._separate_chunk(self.job, 0, 3, (0.0, 60.0))
        cut, trim = mock_window.call_args_list
        self.assertEqual(cut.args[2:4], (0.0, 62.0))
        self.assertEqual(trim.args[2:4], (0.0, 60.0))

    def test_finished_chunk_from_an_earlier_run_is_resumed_not_recomputed(self):
        done = self._write(os.path.join(self.work_dir, "movie_sepchunk_000.wav"))
        self.durations[os.path.normpath(done)] = 60.0
        self.durations[os.path.normpath(self.audio)] = 120.0

        transcription._run_chunked_separation(self.job, 120.0, 60)

        # Only the second window went through the separator.
        self.assertEqual(self.separator.separate.call_count, 1)
        self.assertIn("chunk_001.wav", self.separator.separate.call_args[0][0])

    def test_truncated_leftover_chunk_is_redone(self):
        stale = self._write(os.path.join(self.work_dir, "movie_sepchunk_000.wav"))
        self.durations[os.path.normpath(stale)] = 12.0  # cut off mid-write by a power loss
        self.durations[os.path.normpath(self.audio)] = 120.0

        transcription._run_chunked_separation(self.job, 120.0, 60)
        self.assertEqual(self.separator.separate.call_count, 2)

    def test_symlinked_chunk_stem_is_never_resumed(self):
        target = self._write(os.path.join(self.folder, "victim.wav"))
        link = os.path.join(self.work_dir, "movie_sepchunk_000.wav")
        try:
            os.symlink(target, link)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"symlinks unavailable: {error}")
        self.durations[os.path.normpath(link)] = 60.0
        self.assertFalse(transcription._is_finished_chunk(link, 60.0))

    def test_separator_without_vocal_stem_raises_and_cleans_chunk_input(self):
        self.separator.separate.side_effect = lambda chunk_input: [self._write(os.path.join(self.scratch_dir, "x_(Instrumental).wav"))]
        with self.assertRaises(RuntimeError):
            transcription._separate_chunk(self.job, 0, 1, (0.0, 30.0))
        self.assertFalse(os.path.exists(os.path.join(self.scratch_dir, "chunk_000.wav")))
        self.assertFalse(os.path.exists(os.path.join(self.work_dir, "movie_sepchunk_000.wav")))

    def test_failed_trim_discards_reservation_and_raw_stem(self):
        raw = self._write(os.path.join(self.scratch_dir, "raw_(Vocals).wav"))
        with patch("modules.pipeline.transcription.ffmpeg_utils.write_audio_window", side_effect=RuntimeError("ffmpeg died")):
            with self.assertRaises(RuntimeError):
                transcription._write_chunk_stem(raw, os.path.join(self.work_dir, "movie_sepchunk_000.wav"), 2.0, 60.0, self.work_dir)
        self.assertFalse(os.path.exists(raw))
        self.assertEqual(self._work_entries(), ["movie_temp.wav"])

    def test_failed_join_discards_reservation_and_list(self):
        stems = [self._write(os.path.join(self.work_dir, f"movie_sepchunk_00{i}.wav")) for i in range(2)]
        with patch("modules.pipeline.transcription.ffmpeg_utils.concat_audio_files", side_effect=RuntimeError("concat died")):
            with self.assertRaises(RuntimeError):
                transcription._join_chunk_stems(stems, os.path.join(self.work_dir, "final.wav"), self.job)
        self.assertFalse(os.path.exists(os.path.join(self.work_dir, "final.wav")))
        self.assertFalse(os.path.exists(os.path.join(self.scratch_dir, "stems.list")))
        self.assertEqual([e for e in self._work_entries() if e.startswith(safe_io.SCRATCH_PREFIX)], [], "no leaked reservation")


class TestSeparateDispatch(_ChunkCase):
    def test_short_audio_uses_the_single_pass(self):
        self.durations[os.path.normpath(self.audio)] = 600.0
        with (
            patch("modules.configuration.config.SEPARATION_CHUNK_MINUTES", 30),
            patch("modules.pipeline.transcription._run_chunked_separation") as chunked,
        ):
            transcription._separate(self.job)
        chunked.assert_not_called()
        self.separator.separate.assert_called_once_with(self.audio)

    def test_long_audio_uses_chunks(self):
        self.durations[os.path.normpath(self.audio)] = 4 * 3600.0
        with (
            patch("modules.configuration.config.SEPARATION_CHUNK_MINUTES", 30),
            patch("modules.pipeline.transcription._run_chunked_separation", return_value="stem") as chunked,
        ):
            self.assertEqual(transcription._separate(self.job), "stem")
        chunked.assert_called_once_with(self.job, 4 * 3600.0, 1800)

    def test_chunking_disabled_always_single_pass(self):
        self.durations[os.path.normpath(self.audio)] = 4 * 3600.0
        with (
            patch("modules.configuration.config.SEPARATION_CHUNK_MINUTES", 0),
            patch("modules.pipeline.transcription._run_chunked_separation") as chunked,
        ):
            transcription._separate(self.job)
        chunked.assert_not_called()

    def test_unprobeable_duration_falls_back_to_single_pass(self):
        with (
            patch("modules.configuration.config.SEPARATION_CHUNK_MINUTES", 30),
            patch("modules.pipeline.transcription._run_chunked_separation") as chunked,
        ):
            transcription._separate(self.job)
        chunked.assert_not_called()

    def test_chunked_result_is_found_by_resume_scan(self):
        # The joined stem must match what _get_separated_vocal_path looks for on the next run.
        final = self._write(os.path.join(self.work_dir, "movie_temp_(Vocals)_chunked.wav"))
        self.durations[os.path.normpath(final)] = 133.0
        self.durations[os.path.normpath(self.audio)] = 133.0
        self.assertEqual(transcription._get_separated_vocal_path(os.path.join(self.folder, "movie.mp4")), final)


class TestRunVocalSeparationChunkedEndToEnd(_ChunkCase):
    def test_run_vocal_separation_delegates_and_removes_scratch(self):
        self.durations[os.path.normpath(self.audio)] = 150.0
        manager = MagicMock()
        manager.get_separator.return_value = self.separator
        # Use a separator that writes into whatever output_dir it is handed.
        created = {}

        def get_separator(output_dir=None):
            created["dir"] = output_dir
            self.scratch_dir = output_dir
            self.job["scratch_dir"] = output_dir
            return self.separator

        manager.get_separator.side_effect = get_separator
        with (
            patch("modules.configuration.config.SEPARATION_CHUNK_MINUTES", 1),
            patch("modules.pipeline.transcription.utils.extract_clean_audio", return_value=self.audio),
        ):
            result = transcription._run_vocal_separation(os.path.join(self.folder, "movie.mp4"), manager)
        self.assertEqual(result, os.path.join(self.work_dir, "movie_temp_(Vocals)_chunked.wav"))
        self.assertFalse(os.path.lexists(created["dir"]), "separator scratch directory removed afterwards")
        self.assertEqual(self.separator.separate.call_count, 2)


if __name__ == "__main__":
    unittest.main()
