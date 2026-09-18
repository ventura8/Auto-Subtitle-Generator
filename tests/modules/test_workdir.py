"""Tests for the per-video work directory: creation, link refusal, and shallow purge."""

import importlib
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

from modules import safe_io, workdir


class TestWorkDir(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.folder = self._tmp.name
        self.work_dir = workdir.work_dir_path(self.folder, "movie")

    def _symlink(self, target, link, target_is_directory=False):
        try:
            os.symlink(target, link, target_is_directory=target_is_directory)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"symlinks unavailable: {error}")

    # --- binding to the input ---------------------------------------------

    def _video(self, payload=b"\x00" * 16):
        path = os.path.join(self.folder, "movie.mp4")
        with open(path, "wb") as handle:
            handle.write(payload)
        return path

    def _stem(self):
        path = os.path.join(self.work_dir, "movie_temp.wav")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("x")
        return path

    def test_bind_creates_and_stamps_a_fresh_work_dir(self):
        video = self._video()
        self.assertEqual(workdir.bind_work_dir_to_source(self.folder, "movie", video), workdir.BIND_FRESH)
        stamp_path = os.path.join(self.work_dir, workdir.SOURCE_STAMP_NAME)
        self.assertTrue(os.path.isfile(stamp_path))
        self.assertEqual(workdir._read_stamp(stamp_path), workdir.source_stamp(video))

    def test_bind_keeps_state_for_the_same_input(self):
        video = self._video()
        workdir.bind_work_dir_to_source(self.folder, "movie", video)
        stem = self._stem()
        with patch("modules.workdir.log") as mock_log:
            self.assertEqual(workdir.bind_work_dir_to_source(self.folder, "movie", video), workdir.BIND_SAME)
        self.assertTrue(os.path.isfile(stem))
        mock_log.assert_not_called()

    def test_bind_discards_state_made_for_a_different_input(self):
        video = self._video()
        workdir.bind_work_dir_to_source(self.folder, "movie", video)
        stem = self._stem()
        video = self._video(b"\x01" * 32)
        with patch("modules.workdir.log") as mock_log:
            self.assertEqual(workdir.bind_work_dir_to_source(self.folder, "movie", video), workdir.BIND_CHANGED)
        self.assertFalse(os.path.exists(stem))
        self.assertIn("changed since the last run", mock_log.call_args[0][0])
        self.assertEqual(workdir._read_stamp(os.path.join(self.work_dir, workdir.SOURCE_STAMP_NAME)), workdir.source_stamp(video))

    def test_bind_without_a_readable_input_touches_nothing(self):
        self.assertIsNone(workdir.bind_work_dir_to_source(self.folder, "movie", os.path.join(self.folder, "missing.mp4")))
        self.assertFalse(os.path.exists(self.work_dir))

    def test_bind_treats_a_planted_stamp_link_or_junk_as_mismatch(self):
        video = self._video()
        os.mkdir(self.work_dir)
        stamp_path = os.path.join(self.work_dir, workdir.SOURCE_STAMP_NAME)
        with open(stamp_path, "w", encoding="utf-8") as handle:
            handle.write("[1, 2]")
        self.assertIsNone(workdir._read_stamp(stamp_path))
        os.remove(stamp_path)
        self._symlink(video, stamp_path)
        self.assertIsNone(workdir._read_stamp(stamp_path))
        with patch("modules.workdir.log"):
            workdir.bind_work_dir_to_source(self.folder, "movie", video)
        self.assertFalse(os.path.islink(stamp_path))
        self.assertEqual(workdir._read_stamp(stamp_path), workdir.source_stamp(video))

    def test_bind_refuses_to_reuse_a_stale_dir_it_could_not_discard(self):
        video = self._video()
        os.mkdir(self.work_dir)
        with patch("modules.workdir.purge_work_dir", return_value=False), patch("modules.workdir.log"):
            with self.assertRaises(OSError):
                workdir.bind_work_dir_to_source(self.folder, "movie", video)

    # --- path + creation --------------------------------------------------

    def test_work_dir_path_is_beside_the_video(self):
        self.assertEqual(self.work_dir, os.path.join(self.folder, "movie.asg-temp"))
        self.assertEqual(workdir.work_dir_path("", "movie"), os.path.join(".", "movie.asg-temp"))

    def test_ensure_work_dir_creates_private_directory_once(self):
        self.assertFalse(workdir.work_dir_exists(self.folder, "movie"))
        created = workdir.ensure_work_dir(self.folder, "movie")
        self.assertEqual(created, self.work_dir)
        self.assertTrue(os.path.isdir(self.work_dir))
        self.assertTrue(workdir.work_dir_exists(self.folder, "movie"))
        if sys.platform != "win32":
            # Private to the owner regardless of umask (never read or change the umask itself).
            self.assertEqual(os.stat(self.work_dir).st_mode & 0o077, 0)
        # Second call reuses the directory (resume) instead of failing.
        self.assertEqual(workdir.ensure_work_dir(self.folder, "movie"), self.work_dir)

    def test_ensure_work_dir_refuses_symlink(self):
        elsewhere = os.path.join(self.folder, "elsewhere")
        os.mkdir(elsewhere)
        self._symlink(elsewhere, self.work_dir, target_is_directory=True)
        with self.assertRaises(safe_io.SymlinkRefusedError):
            workdir.ensure_work_dir(self.folder, "movie")
        self.assertFalse(workdir.work_dir_exists(self.folder, "movie"))

    def test_ensure_work_dir_refuses_regular_file_at_name(self):
        with open(self.work_dir, "w", encoding="utf-8") as handle:
            handle.write("not a directory")
        with self.assertRaises(safe_io.SymlinkRefusedError):
            workdir.ensure_work_dir(self.folder, "movie")

    def test_ensure_work_dir_handles_creation_race(self):
        # mkdir loses a race to another process that created the same directory.
        real_mkdir = os.mkdir

        def racing_mkdir(path, mode):
            real_mkdir(path, mode)
            raise FileExistsError(path)

        with patch("modules.workdir.os.mkdir", side_effect=racing_mkdir):
            self.assertEqual(workdir.ensure_work_dir(self.folder, "movie"), self.work_dir)

    # --- purge ------------------------------------------------------------

    def test_purge_missing_work_dir_is_success(self):
        self.assertTrue(workdir.purge_work_dir(self.folder, "movie"))

    def test_purge_removes_files_and_scratch_dirs_only(self):
        workdir.ensure_work_dir(self.folder, "movie")
        for name in ("movie_temp.wav", "movie.manifest.json", ".temp_output.movie.fr.json.abc.tmp", "movie.source_lang.txt"):
            with open(os.path.join(self.work_dir, name), "w", encoding="utf-8") as handle:
                handle.write("x")
        # A crashed atomic write leaves a scratch directory with one file inside.
        scratch = safe_io.create_private_dir(self.work_dir, "movie_vocals")
        with open(os.path.join(scratch, "output-abc.wav"), "w", encoding="utf-8") as handle:
            handle.write("partial")

        self.assertTrue(workdir.purge_work_dir(self.folder, "movie"))
        self.assertFalse(os.path.lexists(self.work_dir))
        # Nothing else in the input folder was touched.
        self.assertEqual(os.listdir(self.folder), [])

    def test_purge_leaves_unknown_subdirectory_and_reports(self):
        workdir.ensure_work_dir(self.folder, "movie")
        foreign = os.path.join(self.work_dir, "user-stuff")
        os.mkdir(foreign)
        with open(os.path.join(foreign, "keep.txt"), "w", encoding="utf-8") as handle:
            handle.write("keep")
        with open(os.path.join(self.work_dir, "movie_temp.wav"), "w", encoding="utf-8") as handle:
            handle.write("x")

        with patch("modules.workdir.log") as mock_log:
            self.assertFalse(workdir.purge_work_dir(self.folder, "movie"))
        self.assertTrue(os.path.isfile(os.path.join(foreign, "keep.txt")))
        self.assertFalse(os.path.exists(os.path.join(self.work_dir, "movie_temp.wav")))
        self.assertIn("user-stuff", mock_log.call_args[0][0])

    def test_purge_never_recurses_into_planted_scratch_dir_with_subdir(self):
        workdir.ensure_work_dir(self.folder, "movie")
        planted = os.path.join(self.work_dir, f"{safe_io.SCRATCH_PREFIX}planted")
        os.makedirs(os.path.join(planted, "nested"))
        with open(os.path.join(planted, "nested", "victim.txt"), "w", encoding="utf-8") as handle:
            handle.write("victim")

        with patch("modules.workdir.log"):
            self.assertFalse(workdir.purge_work_dir(self.folder, "movie"))
        self.assertTrue(os.path.isfile(os.path.join(planted, "nested", "victim.txt")))

    def test_purge_refuses_symlinked_work_dir(self):
        elsewhere = os.path.join(self.folder, "elsewhere")
        os.mkdir(elsewhere)
        with open(os.path.join(elsewhere, "precious.txt"), "w", encoding="utf-8") as handle:
            handle.write("precious")
        self._symlink(elsewhere, self.work_dir, target_is_directory=True)

        with patch("modules.workdir.log"):
            self.assertFalse(workdir.purge_work_dir(self.folder, "movie"))
        self.assertTrue(os.path.isfile(os.path.join(elsewhere, "precious.txt")))

    def test_purge_unlinks_planted_symlink_without_following_it(self):
        workdir.ensure_work_dir(self.folder, "movie")
        victim = os.path.join(self.folder, "victim.txt")
        with open(victim, "w", encoding="utf-8") as handle:
            handle.write("victim")
        self._symlink(victim, os.path.join(self.work_dir, "movie_temp.wav"))

        self.assertTrue(workdir.purge_work_dir(self.folder, "movie"))
        self.assertTrue(os.path.isfile(victim))
        with open(victim, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "victim")

    def test_remove_scratch_dir_missing_or_not_directory(self):
        self.assertTrue(workdir.remove_scratch_dir(os.path.join(self.folder, "missing")))
        regular = os.path.join(self.folder, "file")
        with open(regular, "w", encoding="utf-8") as handle:
            handle.write("x")
        self.assertFalse(workdir.remove_scratch_dir(regular))

    def test_remove_scratch_dir_stops_at_unclassifiable_child(self):
        # A child that cannot be classified as a file or link stops cleanup
        # rather than being removed blindly.
        scratch = safe_io.create_private_dir(self.folder, "movie_vocals")
        child = os.path.join(scratch, "a")
        with open(child, "w", encoding="utf-8") as handle:
            handle.write("x")
        with patch("modules.workdir._is_unlinkable", return_value=False):
            self.assertFalse(workdir.remove_scratch_dir(scratch))
        self.assertTrue(os.path.isfile(child))

    # --- cleanup binding --------------------------------------------------

    def test_binding_rejects_directory_replaced_after_validation(self):
        # Identity recorded before cleanup no longer matches -> abort.
        workdir.ensure_work_dir(self.folder, "movie")
        stale_identity = (-1, -1)
        self.assertFalse(workdir._binding_is_valid(self.work_dir, None, stale_identity))

    def test_binding_rejects_missing_identity(self):
        self.assertFalse(workdir._binding_is_valid(self.work_dir, None, None))

    def test_purge_aborts_when_directory_changes_during_cleanup(self):
        workdir.ensure_work_dir(self.folder, "movie")
        with open(os.path.join(self.work_dir, "movie_temp.wav"), "w", encoding="utf-8") as handle:
            handle.write("x")
        with patch("modules.workdir._binding_is_valid", return_value=False), patch("modules.workdir.log") as mock_log:
            self.assertFalse(workdir.purge_work_dir(self.folder, "movie"))
        # Nothing was touched.
        self.assertTrue(os.path.isfile(os.path.join(self.work_dir, "movie_temp.wav")))
        self.assertIn("changed during cleanup", mock_log.call_args[0][0])

    def test_remove_scratch_dir_aborts_when_binding_invalid(self):
        scratch = safe_io.create_private_dir(self.folder, "movie_vocals")
        child = os.path.join(scratch, "a")
        with open(child, "w", encoding="utf-8") as handle:
            handle.write("x")
        with patch("modules.workdir._binding_is_valid", return_value=False):
            self.assertFalse(workdir.remove_scratch_dir(scratch))
        self.assertTrue(os.path.isfile(child))

    def test_directory_identity_rejects_links_and_non_directories(self):
        self.assertIsNone(workdir._directory_identity(os.path.join(self.folder, "missing")))
        regular = os.path.join(self.folder, "file")
        with open(regular, "w", encoding="utf-8") as handle:
            handle.write("x")
        self.assertIsNone(workdir._directory_identity(regular))
        link = os.path.join(self.folder, "link")
        self._symlink(self.folder, link, target_is_directory=True)
        self.assertIsNone(workdir._directory_identity(link))

    def test_directory_identity_matches_itself(self):
        workdir.ensure_work_dir(self.folder, "movie")
        self.assertIsNotNone(workdir._directory_identity(self.work_dir))
        self.assertEqual(workdir._directory_identity(self.work_dir), workdir._directory_identity(self.work_dir))

    def test_listdir_target_prefers_descriptor(self):
        self.assertEqual(workdir._listdir_target("some/path", None), "some/path")
        # A descriptor of 0 is valid and must not fall back to the pathname.
        self.assertEqual(workdir._listdir_target("some/path", 0), 0)

    def test_dir_fd_probe_is_true_on_a_real_posix_capability_set(self):
        # Exactly what CPython's os.py registers on Linux: stat (fstatat) is in
        # the set, lstat never is, even though os.lstat accepts dir_fd. The
        # probe must not depend on lstat or the binding is dead everywhere.
        posix_dir_fd = {
            os.access,
            os.chmod,
            os.stat,
            os.utime,
            os.link,
            os.mkdir,
            os.open,
            os.readlink,
            os.rename,
            os.symlink,
            os.unlink,
            os.rmdir,
        }
        self.assertNotIn(os.lstat, posix_dir_fd)
        with patch.object(os, "supports_dir_fd", posix_dir_fd), patch.object(os, "supports_fd", {os.listdir}):
            self.assertTrue(workdir._probe_dir_fd_support())

    def test_dir_fd_probe_is_false_when_a_required_call_is_missing(self):
        with patch.object(os, "supports_dir_fd", {os.open, os.stat, os.unlink}), patch.object(os, "supports_fd", {os.listdir}):
            self.assertFalse(workdir._probe_dir_fd_support(), "rmdir missing")
        with patch.object(os, "supports_dir_fd", {os.open, os.stat, os.unlink, os.rmdir}), patch.object(os, "supports_fd", set()):
            self.assertFalse(workdir._probe_dir_fd_support(), "listdir-by-fd missing")

    def test_open_dir_handle_returns_none_when_unsupported(self):
        workdir.ensure_work_dir(self.folder, "movie")
        with patch("modules.workdir._DIR_FD_SUPPORTED", False):
            self.assertIsNone(workdir._open_dir_handle(self.work_dir))

    def test_open_dir_handle_returns_none_for_missing_directory(self):
        self.assertIsNone(workdir._open_dir_handle(os.path.join(self.folder, "missing")))

    def test_close_handle_tolerates_none_and_bad_descriptor(self):
        workdir._close_handle(None)
        workdir._close_handle(-1)

    # --- POSIX dir_fd branches (simulated so they are covered on Windows too) ---

    def test_open_dir_handle_uses_nofollow_flags_when_supported(self):
        with (
            patch("modules.workdir._DIR_FD_SUPPORTED", True),
            patch("modules.workdir.os.open", return_value=7) as mock_open,
        ):
            self.assertEqual(workdir._open_dir_handle("some/dir"), 7)
        mock_open.assert_called_once_with("some/dir", workdir._DIR_OPEN_FLAGS)

    def test_open_dir_handle_returns_none_when_open_fails(self):
        with (
            patch("modules.workdir._DIR_FD_SUPPORTED", True),
            patch("modules.workdir.os.open", side_effect=OSError("replaced by a link")),
        ):
            self.assertIsNone(workdir._open_dir_handle("some/dir"))

    def test_binding_accepts_descriptor_still_naming_the_directory(self):
        identity = (11, 22)
        with patch("modules.workdir.os.fstat", return_value=os.stat_result((0, 22, 11, 0, 0, 0, 0, 0, 0, 0))):
            self.assertTrue(workdir._binding_is_valid("some/dir", 7, identity))

    def test_binding_rejects_descriptor_pointing_elsewhere(self):
        identity = (11, 22)
        with patch("modules.workdir.os.fstat", return_value=os.stat_result((0, 99, 11, 0, 0, 0, 0, 0, 0, 0))):
            self.assertFalse(workdir._binding_is_valid("some/dir", 7, identity))

    def test_binding_rejects_unstatable_descriptor(self):
        with patch("modules.workdir.os.fstat", side_effect=OSError("bad fd")):
            self.assertFalse(workdir._binding_is_valid("some/dir", 7, (11, 22)))

    def test_nested_scratch_dir_uses_parent_binding_not_the_pathname(self):
        # With a held parent descriptor the nested directory is opened relative
        # to it and removed relative to it -- never re-resolved by pathname.
        parent_fd = 7
        with (
            patch("modules.workdir.os.open", return_value=9) as mock_open,
            patch("modules.workdir._empty_bound_dir", return_value=True),
            patch("modules.workdir._close_handle") as mock_close,
            patch("modules.workdir._rmdir_with_retry", return_value=True) as mock_rmdir,
            patch("modules.workdir.remove_scratch_dir") as mock_by_pathname,
        ):
            ok = workdir._remove_nested_scratch_dir(".asg-tmp-x", "/work/.asg-tmp-x", parent_fd)
        self.assertTrue(ok)
        mock_open.assert_called_once_with(".asg-tmp-x", workdir._DIR_OPEN_FLAGS, dir_fd=parent_fd)
        mock_rmdir.assert_called_once_with(".asg-tmp-x", parent_fd)
        mock_close.assert_called_once_with(9)
        mock_by_pathname.assert_not_called()

    def test_nested_scratch_dir_refuses_link_or_non_directory(self):
        # O_NOFOLLOW | O_DIRECTORY makes the open fail for a planted link.
        with (
            patch("modules.workdir.os.open", side_effect=OSError("ELOOP")),
            patch("modules.workdir._rmdir_with_retry") as mock_rmdir,
        ):
            self.assertFalse(workdir._remove_nested_scratch_dir(".asg-tmp-x", "/work/.asg-tmp-x", 7))
        mock_rmdir.assert_not_called()

    def test_nested_scratch_dir_treats_vanished_entry_as_removed(self):
        with patch("modules.workdir.os.open", side_effect=FileNotFoundError("gone")):
            self.assertTrue(workdir._remove_nested_scratch_dir(".asg-tmp-x", "/work/.asg-tmp-x", 7))

    def test_nested_scratch_dir_keeps_non_empty_directory_and_closes_handle(self):
        with (
            patch("modules.workdir.os.open", return_value=9),
            patch("modules.workdir._empty_bound_dir", return_value=False),
            patch("modules.workdir._close_handle") as mock_close,
            patch("modules.workdir._rmdir_with_retry") as mock_rmdir,
        ):
            self.assertFalse(workdir._remove_nested_scratch_dir(".asg-tmp-x", "/work/.asg-tmp-x", 7))
        mock_close.assert_called_once_with(9)
        mock_rmdir.assert_not_called()

    def test_nested_scratch_dir_falls_back_to_pathname_without_a_descriptor(self):
        with patch("modules.workdir.remove_scratch_dir", return_value=True) as mock_by_pathname:
            self.assertTrue(workdir._remove_nested_scratch_dir(".asg-tmp-x", "/work/.asg-tmp-x", None))
        mock_by_pathname.assert_called_once_with("/work/.asg-tmp-x")

    def test_binding_refuses_pathname_fallback_where_dir_fd_is_supported(self):
        # On POSIX a failed open must abort cleanup rather than silently
        # falling back to unbound pathname operations.
        workdir.ensure_work_dir(self.folder, "movie")
        identity = workdir._directory_identity(self.work_dir)
        with patch("modules.workdir._DIR_FD_SUPPORTED", True):
            self.assertFalse(workdir._binding_is_valid(self.work_dir, None, identity))

    def test_unlink_and_rmdir_retry_then_give_up(self):
        with patch("modules.workdir.time.sleep") as mock_sleep:
            with patch("modules.workdir.os.unlink", side_effect=PermissionError("locked")):
                self.assertFalse(workdir._unlink_with_retry("x"))
            with patch("modules.workdir.os.rmdir", side_effect=PermissionError("locked")):
                self.assertFalse(workdir._rmdir_with_retry("x"))
        self.assertEqual(mock_sleep.call_count, 2 * (workdir._REMOVE_ATTEMPTS - 1))

    def test_unlink_and_rmdir_treat_missing_as_done(self):
        self.assertTrue(workdir._unlink_with_retry(os.path.join(self.folder, "nope")))
        self.assertTrue(workdir._rmdir_with_retry(os.path.join(self.folder, "nope")))

    def test_purge_reports_rmdir_failure(self):
        workdir.ensure_work_dir(self.folder, "movie")
        with patch("modules.workdir._rmdir_with_retry", return_value=False), patch("modules.workdir.log") as mock_log:
            self.assertFalse(workdir.purge_work_dir(self.folder, "movie"))
        self.assertIn("Could not remove", mock_log.call_args[0][0])

    def test_remove_work_entry_unstatable(self):
        with patch("modules.workdir.os.lstat", side_effect=OSError("gone")):
            self.assertFalse(workdir._remove_work_entry("x", os.path.join(self.folder, "x")))

    def test_safe_listdir_unreadable(self):
        self.assertEqual(workdir._safe_listdir(os.path.join(self.folder, "missing")), [])

    def test_is_link_uses_isjunction_when_available(self):
        with (
            patch("modules.workdir.os.path.islink", return_value=False),
            patch("modules.workdir.os.path.isjunction", return_value=True, create=True),
        ):
            self.assertTrue(workdir._is_link("x"))


class TestLeakGuardCleanup(unittest.TestCase):
    """The conftest leak guard must remove leaked files, not just directories."""

    def test_remove_leaked_entries_removes_files_and_directories(self):
        conftest = importlib.import_module("conftest")
        with tempfile.TemporaryDirectory() as folder:
            leaked_dir = os.path.join(folder, "movie.asg-temp")
            os.mkdir(leaked_dir)
            with open(os.path.join(leaked_dir, "inner.wav"), "w", encoding="utf-8") as handle:
                handle.write("x")
            leaked_file = os.path.join(folder, ".asg-tmp-stray")
            with open(leaked_file, "w", encoding="utf-8") as handle:
                handle.write("x")

            with patch.object(conftest, "_p", folder):
                conftest._remove_leaked_entries(["movie.asg-temp", ".asg-tmp-stray"])

            self.assertFalse(os.path.lexists(leaked_dir))
            self.assertFalse(os.path.lexists(leaked_file), "a leaked file must be removed, not skipped by rmtree")

    def test_remove_leaked_entries_tolerates_already_gone(self):
        conftest = importlib.import_module("conftest")
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(conftest, "_p", folder):
                conftest._remove_leaked_entries(["movie.asg-temp", ".asg-tmp-gone"])

    # --- retry helpers ----------------------------------------------------


if __name__ == "__main__":
    unittest.main()
