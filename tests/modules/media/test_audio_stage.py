"""Tests for the private audio stage the model readers get their copies from."""

import os
import stat
import tempfile
import unittest
from unittest.mock import patch

from modules.media import audio_stage
from modules.media.input_binding import InputRefusedError, set_input_root


class AudioStageTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(set_input_root, None)
        self.folder = self._tmp.name
        self.work_dir = os.path.join(self.folder, "movie.asg-temp")
        os.mkdir(self.work_dir)
        self.work_audio = self._write(os.path.join(self.work_dir, "movie_temp.wav"), b"WORK-AUDIO")

    def _write(self, path, payload):
        with open(path, "wb") as handle:
            handle.write(payload)
        return path

    def _read(self, path):
        with open(path, "rb") as handle:
            return handle.read()

    def _symlink_or_skip(self, target, link):
        try:
            os.symlink(target, link)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"symlinks unavailable: {error}")

    def test_failed_copy_removes_the_partial_file_so_a_retry_works(self):
        with audio_stage.open_stage() as stage:
            with patch("modules.media.audio_stage.shutil.copyfileobj", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    stage.import_file(self.work_audio)
            self.assertEqual(os.listdir(stage.root), [])
            self.assertEqual(self._read(stage.import_file(self.work_audio)), b"WORK-AUDIO")

    def test_name_collision_never_removes_the_existing_copy(self):
        other_dir = os.path.join(self.folder, "other")
        os.mkdir(other_dir)
        other = self._write(os.path.join(other_dir, "movie_temp.wav"), b"OTHER")
        with audio_stage.open_stage() as stage:
            first = stage.import_file(self.work_audio)
            with self.assertRaises(FileExistsError):
                stage.import_file(other)
            self.assertEqual(self._read(first), b"WORK-AUDIO")

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFOs are POSIX only")
    def test_a_fifo_is_refused_without_blocking(self):
        fifo = os.path.join(self.work_dir, "movie_sepchunk_000.wav")
        os.mkfifo(fifo)
        with audio_stage.open_stage() as stage:
            with self.assertRaises(InputRefusedError):
                stage.import_file(fifo)

    def test_stage_is_a_private_directory_outside_the_input_folder(self):
        with audio_stage.open_stage() as stage:
            self.assertTrue(os.path.basename(stage.root).startswith(audio_stage.STAGE_PREFIX))
            self.assertTrue(os.path.isdir(stage.root))
            folder = os.path.realpath(self.folder)
            self.assertNotEqual(os.path.commonpath([os.path.realpath(stage.root), folder]), folder)
            if os.name == "posix":
                self.assertEqual(stat.S_IMODE(os.stat(stage.root).st_mode), 0o700)
        self.assertFalse(os.path.lexists(stage.root))

    def test_import_copies_the_bytes_and_caches_per_path(self):
        with audio_stage.open_stage() as stage:
            copy = stage.import_file(self.work_audio)
            self.assertEqual(copy, os.path.join(stage.root, "movie_temp.wav"))
            self.assertEqual(self._read(copy), b"WORK-AUDIO")
            # The work file changing afterwards does not reach the staged copy, and a
            # second import (also by a relative spelling) returns the same copy.
            self._write(self.work_audio, b"CHANGED")
            relative = os.path.relpath(self.work_audio)
            with patch("modules.media.audio_stage.open_bound_input") as bound:
                self.assertEqual(stage.import_file(relative), copy)
            bound.assert_not_called()
            self.assertEqual(self._read(copy), b"WORK-AUDIO")

    def test_import_refuses_a_symlink_and_never_reads_its_target(self):
        victim = self._write(os.path.join(self.folder, "victim.wav"), b"VICTIM")
        link = os.path.join(self.work_dir, "movie_asr16k.wav")
        self._symlink_or_skip(victim, link)
        with audio_stage.open_stage() as stage:
            with self.assertRaises(InputRefusedError):
                stage.import_file(link)
            self.assertEqual(os.listdir(stage.root), [])
        self.assertEqual(self._read(victim), b"VICTIM")

    def test_import_refuses_a_linked_work_directory_below_the_input_root(self):
        elsewhere = os.path.join(self.folder, "elsewhere")
        os.mkdir(elsewhere)
        self._write(os.path.join(elsewhere, "movie_temp.wav"), b"VICTIM")
        linked_dir = os.path.join(self.folder, "linked.asg-temp")
        self._symlink_or_skip(elsewhere, linked_dir)
        set_input_root(self.folder)
        linked_audio = os.path.join(linked_dir, "movie_temp.wav")
        with audio_stage.open_stage() as stage:
            with self.assertRaises(InputRefusedError):
                stage.import_file(linked_audio)

    def test_import_refuses_a_directory(self):
        with audio_stage.open_stage() as stage, self.assertRaises(InputRefusedError):
            stage.import_file(self.work_dir)

    def test_import_failure_is_not_cached(self):
        missing = os.path.join(self.work_dir, "missing.wav")
        with audio_stage.open_stage() as stage:
            with self.assertRaises(InputRefusedError):
                stage.import_file(missing)
            self._write(missing, b"LATE")
            self.assertEqual(self._read(stage.import_file(missing)), b"LATE")

    def test_path_keeps_only_the_file_name(self):
        with audio_stage.open_stage() as stage:
            self.assertEqual(stage.path(os.path.join("..", "x", "a.wav")), os.path.join(stage.root, "a.wav"))

    def test_new_dir_is_a_fresh_sub_directory_of_the_stage(self):
        with audio_stage.open_stage() as stage:
            first = stage.new_dir("separator")
            second = stage.new_dir("separator")
            self.assertNotEqual(first, second)
            for created in (first, second):
                self.assertEqual(os.path.dirname(created), stage.root)
                self.assertTrue(os.path.basename(created).startswith("separator-"))
                self.assertTrue(os.path.isdir(created))

    def _crash_inside_a_stage(self, seen_roots):
        with audio_stage.open_stage() as stage:
            seen_roots.append(stage.root)
            stage.import_file(self.work_audio)
            os.mkdir(os.path.join(stage.new_dir("separator"), "nested"))
            raise RuntimeError("model crashed")

    def test_stage_is_removed_even_when_the_block_raises(self):
        seen_roots = []
        with self.assertRaisesRegex(RuntimeError, "model crashed"):
            self._crash_inside_a_stage(seen_roots)
        self.assertEqual(len(seen_roots), 1)
        self.assertFalse(os.path.lexists(seen_roots[0]))
        # The work directory it copied from is left alone.
        self.assertEqual(self._read(self.work_audio), b"WORK-AUDIO")


if __name__ == "__main__":
    unittest.main()
