"""HeldDirectory: every entry is reached through the directory held at validation time."""

import glob
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from modules import held_dir
from modules.held_dir import HeldDirectory
from modules.safe_io import DIR_FD_SUPPORTED, SCRATCH_PREFIX, SymlinkRefusedError


def _write(path, text):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _read(path):
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _write_data(handle):
    handle.write("data")


def _write_through(held, path, action):
    """Open a text writer on ``path`` and run ``action`` on its handle before it promotes."""
    with held.text_writer(path) as handle:
        action(handle)


class _HeldDirCase(unittest.TestCase):
    """Shared scenarios; subclasses pick the descriptor or the re-verified identity mode."""

    use_dir_fd = DIR_FD_SUPPORTED

    def setUp(self):
        self.root = os.path.realpath(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)
        self.work_dir = os.path.join(self.root, "movie.asg-temp")
        os.mkdir(self.work_dir)
        self.victim = os.path.join(self.root, "victim.json")
        _write(self.victim, "VICTIM")
        self.held = self.make_held(self.work_dir)

    def make_held(self, path):
        with patch("modules.held_dir.DIR_FD_SUPPORTED", self.use_dir_fd):
            held = HeldDirectory(path)
        self.addCleanup(held.close)
        return held

    def entry(self, name):
        return os.path.join(self.work_dir, name)

    def scratch_entries(self):
        return glob.glob(os.path.join(glob.escape(self.work_dir), f"{SCRATCH_PREFIX}*"))

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFOs are POSIX only")
    def test_read_text_refuses_a_fifo_without_blocking(self):
        fifo = self.entry("job.json")
        os.mkfifo(fifo)
        with self.assertRaises(SymlinkRefusedError):
            self.held.read_text(fifo)

    def test_entry_name_accepts_only_direct_children(self):
        self.assertEqual(self.held.entry_name(self.entry("a.json")), "a.json")
        nested = os.path.join(self.work_dir, "sub", "a.json")
        with self.assertRaises(SymlinkRefusedError):
            self.held.entry_name(nested)
        with self.assertRaises(SymlinkRefusedError):
            self.held.entry_name(self.victim)

    def test_exists_reports_files_links_and_missing_entries(self):
        _write(self.entry("a.json"), "x")
        os.symlink(os.path.join(self.root, "missing"), self.entry("dangling"))

        self.assertTrue(self.held.exists(self.entry("a.json")))
        self.assertTrue(self.held.exists(self.entry("dangling")))
        self.assertFalse(self.held.exists(self.entry("absent.json")))

    def test_read_text_returns_a_regular_file(self):
        _write(self.entry("a.json"), "[1]")

        self.assertEqual(self.held.read_text(self.entry("a.json")), "[1]")

    def test_read_text_refuses_a_symlink(self):
        link = self.entry("a.json")
        os.symlink(self.victim, link)

        with self.assertRaises(OSError):
            self.held.read_text(link)

    def test_read_text_refuses_a_directory(self):
        sub = self.entry("sub")
        os.mkdir(sub)

        with self.assertRaises(SymlinkRefusedError):
            self.held.read_text(sub)

    def test_remove_unlinks_the_link_not_its_target(self):
        os.symlink(self.victim, self.entry("a.json"))

        self.held.remove(self.entry("a.json"))
        self.held.remove(self.entry("absent.json"))

        self.assertFalse(os.path.lexists(self.entry("a.json")))
        self.assertEqual(_read(self.victim), "VICTIM")

    def test_text_writer_replaces_the_destination_atomically(self):
        _write(self.entry("out.json"), "old")

        with self.held.text_writer(self.entry("out.json")) as handle:
            handle.write("new")
            self.assertEqual(_read(self.entry("out.json")), "old")
            self.assertEqual(len(self.scratch_entries()), 1)

        self.assertEqual(_read(self.entry("out.json")), "new")
        self.assertEqual(self.scratch_entries(), [])

    def test_text_writer_failure_discards_the_scratch_entry(self):
        def write_then_fail(handle):
            handle.write("partial")
            raise ValueError("boom")

        target = self.entry("out.json")
        with self.assertRaises(ValueError):
            _write_through(self.held, target, write_then_fail)

        self.assertEqual(os.listdir(self.work_dir), [])

    def test_text_writer_refuses_a_replaced_scratch_entry_and_leaves_the_impostor(self):
        def replace_scratch_with_link(handle):
            handle.write("data")
            (scratch,) = self.scratch_entries()
            os.remove(scratch)
            os.symlink(self.victim, scratch)

        target = self.entry("out.json")
        with self.assertRaises(SymlinkRefusedError):
            _write_through(self.held, target, replace_scratch_with_link)

        self.assertFalse(os.path.lexists(self.entry("out.json")))
        self.assertEqual(len(self.scratch_entries()), 1)
        self.assertEqual(_read(self.victim), "VICTIM")

    def test_text_writer_tolerates_a_vanished_scratch_entry(self):
        def remove_scratch(handle):
            handle.write("data")
            (scratch,) = self.scratch_entries()
            os.remove(scratch)

        target = self.entry("out.json")
        with self.assertRaises(FileNotFoundError):
            _write_through(self.held, target, remove_scratch)

        self.assertEqual(os.listdir(self.work_dir), [])

    def test_text_writer_never_follows_a_link_planted_at_the_scratch_name(self):
        token = "cd" * 8
        planted = self.entry(f"{SCRATCH_PREFIX}{token}.tmp")
        os.symlink(self.victim, planted)
        target = self.entry("out.json")

        with patch("modules.held_dir.secrets.token_hex", return_value=token):
            with self.assertRaises(FileExistsError):
                _write_through(self.held, target, _write_data)

        self.assertTrue(os.path.islink(planted))
        self.assertEqual(_read(self.victim), "VICTIM")

    def test_close_is_idempotent(self):
        self.held.close()
        self.held.close()

        self.assertIsNone(self.held.fd)

    def test_constructor_refuses_a_symlinked_directory(self):
        link = os.path.join(self.root, "link.asg-temp")
        os.symlink(self.work_dir, link)

        with self.assertRaises(SymlinkRefusedError):
            self.make_held(link)

    def test_constructor_refuses_a_regular_file(self):
        with self.assertRaises(SymlinkRefusedError):
            self.make_held(self.victim)


@unittest.skipUnless(DIR_FD_SUPPORTED, "dir_fd binding is POSIX only")
class TestHeldDirectoryWithDescriptor(_HeldDirCase):
    use_dir_fd = True

    def _swap_work_dir_for_link(self):
        """Move the held directory away and put a link to another directory at its name."""
        elsewhere = os.path.join(self.root, "elsewhere")
        os.mkdir(elsewhere)
        moved = self.work_dir + ".moved"
        os.rename(self.work_dir, moved)
        os.symlink(elsewhere, self.work_dir)
        return elsewhere, moved

    def test_holds_an_open_descriptor(self):
        self.assertIsNotNone(self.held.fd)

    def test_writes_follow_the_held_directory_not_its_old_name(self):
        elsewhere, moved = self._swap_work_dir_for_link()

        with self.held.text_writer(self.entry("out.json")) as handle:
            handle.write("data")

        self.assertEqual(os.listdir(elsewhere), [])
        self.assertEqual(_read(os.path.join(moved, "out.json")), "data")

    def test_reads_follow_the_held_directory_not_its_old_name(self):
        _write(self.entry("in.json"), "mine")
        elsewhere, _moved = self._swap_work_dir_for_link()
        _write(os.path.join(elsewhere, "in.json"), "VICTIM")

        self.assertEqual(self.held.read_text(self.entry("in.json")), "mine")

    def test_directory_swapped_while_opening_is_refused(self):
        other = os.stat(self.root)
        with patch("modules.held_dir.os.fstat", return_value=other):
            with self.assertRaises(SymlinkRefusedError):
                self.make_held(self.work_dir)


class TestHeldDirectoryWithoutDescriptor(_HeldDirCase):
    """The identity-checked mode used where ``dir_fd`` is unavailable (Windows)."""

    use_dir_fd = False

    def test_holds_no_descriptor(self):
        self.assertIsNone(self.held.fd)

    def test_replaced_directory_is_refused_before_any_operation(self):
        os.rename(self.work_dir, self.work_dir + ".moved")
        os.mkdir(self.work_dir)
        entry = self.entry("a.json")

        with self.assertRaises(SymlinkRefusedError):
            self.held.exists(entry)
        with self.assertRaises(SymlinkRefusedError):
            self.held.remove(entry)

    def test_directory_replaced_by_a_link_is_refused_before_writing(self):
        elsewhere = os.path.join(self.root, "elsewhere")
        os.mkdir(elsewhere)
        os.rename(self.work_dir, self.work_dir + ".moved")
        os.symlink(elsewhere, self.work_dir)
        target = self.entry("out.json")

        with self.assertRaises(SymlinkRefusedError):
            _write_through(self.held, target, _write_data)

        self.assertEqual(os.listdir(elsewhere), [])

    def test_read_refuses_a_file_swapped_between_check_and_open(self):
        entry = self.entry("a.json")
        _write(entry, "mine")
        other = os.lstat(self.victim)

        with patch.object(HeldDirectory, "_stat", return_value=other):
            with self.assertRaises(SymlinkRefusedError):
                self.held.read_text(entry)

    def test_promote_refuses_a_link_planted_at_the_destination(self):
        target = self.entry("out.json")

        def plant_link_at_destination(handle):
            handle.write("data")
            os.symlink(self.victim, target)

        with self.assertRaises(SymlinkRefusedError):
            _write_through(self.held, target, plant_link_at_destination)

        self.assertEqual(_read(self.victim), "VICTIM")
        self.assertEqual(self.scratch_entries(), [])


class TestIsExpectedRegular(unittest.TestCase):
    def test_regular_file_without_a_prior_stat_is_accepted(self):
        self.assertTrue(held_dir._is_expected_regular(None, os.stat(__file__)))

    def test_non_regular_object_is_refused(self):
        self.assertFalse(held_dir._is_expected_regular(None, os.stat(os.path.dirname(__file__))))


if __name__ == "__main__":
    unittest.main()
