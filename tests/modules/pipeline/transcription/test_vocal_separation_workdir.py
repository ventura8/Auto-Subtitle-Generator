"""Vocal separation must keep every stem inside the per-video work directory and resume only valid tracks."""

import importlib
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from modules import safe_io, workdir

transcription = importlib.import_module("modules.pipeline.transcription")


class TestVocalSeparationWorkDir(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.folder = self._tmp.name
        self.video = os.path.join(self.folder, "movie.mp4")
        with open(self.video, "wb") as handle:
            handle.write(b"\x00" * 16)
        self.work_dir = workdir.ensure_work_dir(self.folder, "movie")
        self.temp_wav = os.path.join(self.work_dir, "movie_temp.wav")
        with open(self.temp_wav, "wb") as handle:
            handle.write(b"\x00" * 2048)

    def _write(self, path, payload=b"\x00" * 64):
        with open(path, "wb") as handle:
            handle.write(payload)
        return path

    def _folder_entries(self):
        return sorted(entry for entry in os.listdir(self.folder) if entry != "movie.mp4")

    # --- resume -----------------------------------------------------------

    def test_resume_finds_valid_vocal_track_in_work_dir(self):
        vocal = self._write(os.path.join(self.work_dir, "movie_temp_(Vocals)_model.wav"))
        with patch("modules.pipeline.transcription.utils.get_audio_duration", return_value=60.0):
            self.assertEqual(transcription._get_separated_vocal_path(self.video), vocal)

    def test_resume_ignores_vocals_left_beside_video_by_old_releases(self):
        self._write(os.path.join(self.folder, "movie_temp_(Vocals)_model.wav"))
        with patch("modules.pipeline.transcription.utils.get_audio_duration", return_value=60.0):
            self.assertIsNone(transcription._get_separated_vocal_path(self.video))

    def test_resume_discards_truncated_vocal_track(self):
        vocal = self._write(os.path.join(self.work_dir, "movie_temp_(Vocals)_model.wav"))

        def duration(path):
            return 12.0 if "Vocals" in path else 60.0

        with (
            patch("modules.pipeline.transcription.utils.get_audio_duration", side_effect=duration),
            patch("modules.pipeline.transcription.log") as mock_log,
        ):
            self.assertIsNone(transcription._get_separated_vocal_path(self.video))
        self.assertFalse(os.path.exists(vocal))
        self.assertIn("Discarding incomplete vocal track", mock_log.call_args[0][0])

    def test_resume_without_work_dir(self):
        os.remove(self.temp_wav)
        os.rmdir(self.work_dir)
        self.assertIsNone(transcription._get_separated_vocal_path(self.video))

    def test_valid_vocal_track_rejects_symlink_and_unprobeable_files(self):
        vocal = self._write(os.path.join(self.work_dir, "movie_temp_(Vocals)_model.wav"))
        with patch("modules.pipeline.transcription.utils.get_audio_duration", side_effect=OSError("ffprobe missing")):
            self.assertFalse(transcription._is_valid_vocal_track(vocal, self.temp_wav))
        with patch("modules.pipeline.transcription.utils.get_audio_duration", return_value=0.0):
            self.assertFalse(transcription._is_valid_vocal_track(vocal, self.temp_wav))
        with patch("modules.pipeline.transcription.os.path.islink", return_value=True):
            self.assertFalse(transcription._is_valid_vocal_track(vocal, self.temp_wav))

    def test_valid_vocal_track_without_source_wav_only_needs_positive_duration(self):
        vocal = self._write(os.path.join(self.work_dir, "movie_temp_(Vocals)_model.wav"))
        with patch("modules.pipeline.transcription.utils.get_audio_duration", return_value=5.0):
            self.assertTrue(transcription._is_valid_vocal_track(vocal, os.path.join(self.work_dir, "missing.wav")))

    def test_discard_file_ignores_missing(self):
        transcription._discard_file(os.path.join(self.work_dir, "missing.wav"))

    # --- separation run ---------------------------------------------------

    def _separator_that_writes(self, names):
        """Return a model manager whose separator writes ``names`` into its output_dir."""
        manager = MagicMock()

        def get_separator(output_dir=None):
            separator = MagicMock()

            def separate(_audio_input):
                written = []
                for name in names:
                    written.append(self._write(os.path.join(output_dir, name)))
                return written

            separator.separate.side_effect = separate
            manager.last_output_dir = output_dir
            return separator

        manager.get_separator.side_effect = get_separator
        return manager

    def test_run_vocal_separation_moves_only_vocals_into_work_dir(self):
        manager = self._separator_that_writes(["movie_temp_(Vocals)_model.wav", "movie_temp_(Instrumental)_model.wav"])
        with (
            patch("modules.pipeline.transcription.utils.extract_clean_audio", return_value=self.temp_wav),
            patch("modules.pipeline.transcription.log"),
        ):
            vocal = transcription._run_vocal_separation(self.video, manager)

        self.assertEqual(vocal, os.path.join(self.work_dir, "movie_temp_(Vocals)_model.wav"))
        self.assertTrue(os.path.isfile(vocal))
        # The separator wrote into a private scratch dir inside the work dir, which is gone now.
        self.assertTrue(manager.last_output_dir.startswith(os.path.join(self.work_dir, safe_io.SCRATCH_PREFIX)))
        self.assertFalse(os.path.lexists(manager.last_output_dir))
        self.assertEqual(sorted(os.listdir(self.work_dir)), ["movie_temp.wav", "movie_temp_(Vocals)_model.wav"])
        self.assertEqual(self._folder_entries(), ["movie.asg-temp"])

    def test_run_vocal_separation_returns_none_without_vocals_and_cleans_scratch(self):
        manager = self._separator_that_writes(["movie_temp_(Instrumental)_model.wav"])
        with (
            patch("modules.pipeline.transcription.utils.extract_clean_audio", return_value=self.temp_wav),
            patch("modules.pipeline.transcription.log"),
        ):
            self.assertIsNone(transcription._run_vocal_separation(self.video, manager))
        self.assertEqual(sorted(os.listdir(self.work_dir)), ["movie_temp.wav"])

    def test_run_vocal_separation_failure_leaves_no_scratch_behind(self):
        manager = MagicMock()
        manager.get_separator.return_value.separate.side_effect = RuntimeError("boom")
        with (
            patch("modules.pipeline.transcription.utils.extract_clean_audio", return_value=self.temp_wav),
            patch("modules.pipeline.transcription.log"),
        ):
            with self.assertRaises(RuntimeError):
                transcription._run_vocal_separation(self.video, manager)
        self.assertEqual(sorted(os.listdir(self.work_dir)), ["movie_temp.wav"])

    def test_run_vocal_separation_replaces_stale_vocal_output(self):
        stale = self._write(os.path.join(self.work_dir, "movie_temp_(Vocals)_model.wav"), b"stale")
        manager = self._separator_that_writes(["movie_temp_(Vocals)_model.wav"])
        with (
            patch("modules.pipeline.transcription.utils.extract_clean_audio", return_value=self.temp_wav),
            patch("modules.pipeline.transcription.log"),
        ):
            vocal = transcription._run_vocal_separation(self.video, manager)
        self.assertEqual(vocal, stale)
        with open(vocal, "rb") as handle:
            self.assertNotEqual(handle.read(), b"stale")

    def test_move_separator_output_refuses_symlink_destination(self):
        source = self._write(os.path.join(self.work_dir, "src.wav"))
        destination = os.path.join(self.work_dir, "movie_temp_(Vocals)_model.wav")
        try:
            os.symlink(self.video, destination)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"symlinks unavailable: {error}")
        with self.assertRaises(safe_io.SymlinkRefusedError):
            transcription._move_separator_output(source, destination)
        self.assertTrue(os.path.isfile(source))

    def test_move_separator_output_noop_for_missing_or_same_path(self):
        transcription._move_separator_output(os.path.join(self.work_dir, "missing.wav"), os.path.join(self.work_dir, "x.wav"))
        same = self._write(os.path.join(self.work_dir, "same.wav"))
        transcription._move_separator_output(same, same)
        self.assertTrue(os.path.isfile(same))

    def test_resolve_separator_output_path_variants(self):
        absolute = self._write(os.path.join(self.work_dir, "abs.wav"))
        self.assertEqual(transcription._resolve_separator_output_path(absolute, self.work_dir), absolute)
        relative_missing = "movie_temp_(Vocals)_model.wav"
        self.assertEqual(
            transcription._resolve_separator_output_path(relative_missing, self.work_dir),
            os.path.join(self.work_dir, relative_missing),
        )

    def test_resolve_separator_output_path_never_prefers_cwd(self):
        # A same-named file in the process CWD must never be consumed: the name
        # resolves against the scratch directory even when the CWD file exists.
        name = "movie_temp_(Vocals)_model.wav"
        with patch("modules.pipeline.transcription.os.path.exists", return_value=True):
            self.assertEqual(
                transcription._resolve_separator_output_path(name, self.work_dir),
                os.path.join(self.work_dir, name),
            )

    def test_resolve_separator_output_path_rejects_paths_outside_scratch_dir(self):
        outside_absolute = os.path.join(self.folder, "outside_(Vocals).wav")
        traversal = os.path.join("..", "outside_(Vocals).wav")
        self.assertIsNone(transcription._resolve_separator_output_path(outside_absolute, self.work_dir))
        self.assertIsNone(transcription._resolve_separator_output_path(traversal, self.work_dir))

    def test_process_separator_outputs_skips_escaping_stem_without_moving_it(self):
        # A stem reported outside the scratch directory must be left untouched.
        victim = self._write(os.path.join(self.folder, "victim_(Vocals).wav"), b"victim")
        scratch = safe_io.create_private_dir(self.work_dir, "movie_vocals")
        with patch("modules.pipeline.transcription.log") as mock_log:
            result = transcription._process_separator_outputs([victim], scratch, self.work_dir)
        self.assertIsNone(result)
        self.assertTrue(os.path.isfile(victim))
        with open(victim, "rb") as handle:
            self.assertEqual(handle.read(), b"victim")
        self.assertIn("outside the scratch directory", mock_log.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
