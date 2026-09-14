import os
import tempfile
import unittest
from unittest.mock import patch

from modules import safe_io, utils
from modules.models import Segment
from modules.subtitles.srt_io import save_srt


class TestSafeIO(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = self.tmp.name
        self.victim = os.path.join(self.folder, "victim.txt")
        with open(self.victim, "w", encoding="utf-8") as fh:
            fh.write("keep me")

    def tearDown(self):
        self.tmp.cleanup()

    def _plant_symlink(self, name, target=None):
        link = os.path.join(self.folder, name)
        try:
            os.symlink(target or self.victim, link)
        except OSError as exc:
            if os.name == "nt":
                self.skipTest(f"symlink creation unavailable on this Windows host: {exc}")
            raise
        return link

    def _assert_victim_untouched(self):
        with open(self.victim, "r", encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "keep me")

    def _folder_entries(self):
        return sorted(e for e in os.listdir(self.folder) if e != "victim.txt")

    # --- atomic_text_writer -------------------------------------------------

    def test_atomic_text_writer_writes_plain_file(self):
        path = os.path.join(self.folder, "movie.common_input.json")
        with safe_io.atomic_text_writer(path) as fh:
            fh.write("{}")
        with open(path, "r", encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "{}")
        self.assertEqual(self._folder_entries(), ["movie.common_input.json"])

    def test_atomic_text_writer_honours_umask_without_changing_it(self):
        path = os.path.join(self.folder, "movie.en.srt")
        with patch("modules.safe_io.os.umask") as mock_umask:
            with safe_io.atomic_text_writer(path) as fh:
                fh.write("x")
        mock_umask.assert_not_called()
        if os.name != "nt":
            expected = 0o666 & ~_current_umask()
            self.assertEqual(os.stat(path).st_mode & 0o777, expected)

    def test_atomic_text_writer_refuses_symlink_destination(self):
        link = self._plant_symlink("movie.common_input.json")
        with self.assertRaises(safe_io.SymlinkRefusedError):
            with safe_io.atomic_text_writer(link) as fh:
                fh.write("pwned")
        self._assert_victim_untouched()
        self.assertTrue(os.path.islink(link))
        self.assertEqual(self._folder_entries(), ["movie.common_input.json"])

    def test_atomic_text_writer_discards_scratch_on_error(self):
        path = os.path.join(self.folder, "movie.manifest.json")
        with self.assertRaises(RuntimeError):
            with safe_io.atomic_text_writer(path) as fh:
                fh.write("partial")
                raise RuntimeError("boom")
        self.assertFalse(os.path.exists(path))
        self.assertEqual(self._folder_entries(), [])

    def test_atomic_text_writer_survives_scratch_dir_swap(self):
        # A live attacker renames the private scratch directory away and plants
        # a symlink (to a directory holding a symlink to the victim) in its
        # place while the writer holds the handle. Content must go through the
        # descriptor, and promotion must be refused.
        path = os.path.join(self.folder, "movie.en.srt")
        with self.assertRaises(safe_io.SymlinkRefusedError):
            with safe_io.atomic_text_writer(path) as fh:
                fh.write("pwned")
                self._swap_scratch_dir_for_symlink()
        self._assert_victim_untouched()
        self.assertFalse(os.path.exists(path))

    def _swap_scratch_dir_for_symlink(self):
        scratch_dirs = [e for e in os.listdir(self.folder) if e.startswith(safe_io.SCRATCH_PREFIX)]
        self.assertEqual(len(scratch_dirs), 1)
        scratch_dir = os.path.join(self.folder, scratch_dirs[0])
        moved = scratch_dir + ".moved"
        os.rename(scratch_dir, moved)
        decoy_dir = os.path.join(self.folder, "decoy-dir")
        os.mkdir(decoy_dir)
        for name in os.listdir(moved):
            os.symlink(self.victim, os.path.join(decoy_dir, name))
        self._plant_symlink(scratch_dirs[0], decoy_dir)
        self.addCleanup(self._remove_tree_no_follow, moved)
        self.addCleanup(self._remove_tree_no_follow, decoy_dir)

    @staticmethod
    def _remove_tree_no_follow(path):
        if os.path.islink(path) or not os.path.isdir(path):
            return
        for name in os.listdir(path):
            os.remove(os.path.join(path, name))
        os.rmdir(path)

    # --- save_srt ------------------------------------------------------------

    def test_save_srt_ignores_planted_tmp_symlink(self):
        # A predictable "<srt>.tmp" symlink must never be followed.
        self._plant_symlink("movie.en.srt.tmp")
        target = os.path.join(self.folder, "movie.en.srt")
        save_srt([Segment(0.0, 1.0, "hi")], target)
        self._assert_victim_untouched()
        with open(target, "r", encoding="utf-8") as fh:
            self.assertIn("hi", fh.read())

    # --- reserve / promote / discard ----------------------------------------

    def test_reserve_temp_path_uses_private_scratch_dir(self):
        final = os.path.join(self.folder, "movie_temp.wav")
        reservation = safe_io.reserve_temp_path(final)
        self.assertEqual(os.fspath(reservation), reservation.path)
        self.assertEqual(os.path.dirname(reservation.dir_path), self.folder)
        self.assertTrue(os.path.basename(reservation.dir_path).startswith(f"{safe_io.SCRATCH_PREFIX}movie_temp-"))
        self.assertTrue(reservation.path.endswith(".wav"))
        self.assertIsNone(reservation.fd)
        if os.name != "nt":
            self.assertEqual(os.stat(reservation.dir_path).st_mode & 0o777, 0o700)
        safe_io.discard_temp_path(reservation)
        self.assertEqual(self._folder_entries(), [])

    def test_scratch_tag_is_bounded_and_collision_resistant(self):
        long_a = "x" * 240
        long_b = "x" * 239 + "y"
        self.assertNotEqual(safe_io.scratch_tag(long_a), safe_io.scratch_tag(long_b))
        self.assertLess(len(safe_io.scratch_tag(long_a)), 40)
        reservation = safe_io.reserve_temp_path(os.path.join(self.folder, f"{long_a}.wav"))
        self.assertTrue(os.path.exists(reservation.path))
        safe_io.discard_temp_path(reservation)
        self.assertEqual(self._folder_entries(), [])

    def test_promote_temp_path_refuses_symlink(self):
        link = self._plant_symlink("movie_temp.wav")
        reservation = safe_io.reserve_temp_path(link)
        with self.assertRaises(safe_io.SymlinkRefusedError):
            safe_io.promote_temp_path(reservation, link)
        self._assert_victim_untouched()
        self.assertTrue(os.path.islink(link))
        # A failed promotion leaves the reservation intact for the caller to discard.
        self.assertTrue(os.path.isfile(reservation.path))
        safe_io.discard_temp_path(reservation)
        self.assertEqual(self._folder_entries(), ["movie_temp.wav"])

    def test_promote_temp_path_refuses_replaced_scratch_dir(self):
        # Attacker renames the 0700 directory away and recreates a plain
        # directory + regular file with the same names (no symlink involved).
        final = os.path.join(self.folder, "movie_temp.wav")
        reservation = safe_io.reserve_temp_path(final)
        moved = reservation.dir_path + ".moved"
        os.rename(reservation.dir_path, moved)
        os.mkdir(reservation.dir_path)
        with open(reservation.path, "w", encoding="utf-8") as fh:
            fh.write("attacker content")
        self.addCleanup(self._remove_tree_no_follow, moved)
        self.addCleanup(self._remove_tree_no_follow, reservation.dir_path)
        with self.assertRaises(safe_io.SymlinkRefusedError):
            safe_io.promote_temp_path(reservation, final)
        self.assertFalse(os.path.exists(final))
        # The attacker's replacement directory must not be deleted by us.
        self.assertTrue(os.path.isdir(reservation.dir_path))
        self.assertTrue(os.path.isfile(reservation.path))

    def test_discard_temp_path_never_deletes_replaced_directory(self):
        final = os.path.join(self.folder, "movie_temp.wav")
        reservation = safe_io.reserve_temp_path(final)
        moved = reservation.dir_path + ".moved"
        os.rename(reservation.dir_path, moved)
        os.mkdir(reservation.dir_path)
        bystander = os.path.join(reservation.dir_path, "unrelated.txt")
        with open(bystander, "w", encoding="utf-8") as fh:
            fh.write("bystander")
        self.addCleanup(self._remove_tree_no_follow, moved)
        self.addCleanup(self._remove_tree_no_follow, reservation.dir_path)
        safe_io.discard_temp_path(reservation)
        self.assertTrue(os.path.isfile(bystander))
        # Our real scratch file (in the moved dir) is still removed when bound by dir_fd.
        if safe_io.DIR_FD_SUPPORTED:
            self.assertEqual(os.listdir(moved), [])

    def test_promote_temp_path_moves_and_cleans_scratch_dir(self):
        final = os.path.join(self.folder, "movie_multilang.mp4")
        reservation = safe_io.reserve_temp_path(final)
        with open(reservation.path, "wb") as fh:
            fh.write(b"video")
        safe_io.promote_temp_path(reservation, final)
        with open(final, "rb") as fh:
            self.assertEqual(fh.read(), b"video")
        self.assertEqual(self._folder_entries(), ["movie_multilang.mp4"])
        if os.name != "nt":
            self.assertEqual(os.stat(final).st_mode & 0o777, 0o666 & ~_current_umask())

    # --- pathname fallback (platforms without dir_fd support, e.g. Windows) ---

    def test_pathname_fallback_round_trip_and_tamper_detection(self):
        final = os.path.join(self.folder, "movie.en.srt")
        with patch("modules.safe_io.DIR_FD_SUPPORTED", False):
            with safe_io.atomic_text_writer(final) as fh:
                fh.write("ok")
            with open(final, "r", encoding="utf-8") as fh:
                self.assertEqual(fh.read(), "ok")

            reservation = safe_io.reserve_temp_path(os.path.join(self.folder, "movie_temp.wav"))
            self.assertIsNone(reservation.dir_fd)
            os.remove(reservation.path)
            with open(reservation.path, "w", encoding="utf-8") as fh:
                fh.write("replacement")  # new inode at the same pathname
            with self.assertRaises(safe_io.SymlinkRefusedError):
                safe_io.promote_temp_path(reservation, os.path.join(self.folder, "movie_temp.wav"))
            safe_io.discard_temp_path(reservation)  # refuses to delete the replacement, leaves it
            self.assertTrue(os.path.isfile(reservation.path))
            os.remove(reservation.path)
            os.rmdir(reservation.dir_path)
        self.assertEqual(self._folder_entries(), ["movie.en.srt"])

    def test_reserve_cleans_up_when_file_creation_fails(self):
        with patch("modules.safe_io._create_exclusive", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                safe_io.reserve_temp_path(os.path.join(self.folder, "movie_temp.wav"))
        self.assertEqual(self._folder_entries(), [])

    def test_name_reservation_retries_then_gives_up(self):
        with patch("modules.safe_io.secrets.token_hex", return_value="fixed"):
            first = safe_io.reserve_temp_path(os.path.join(self.folder, "movie_temp.wav"))
            self.addCleanup(safe_io.discard_temp_path, first)
            with self.assertRaises(FileExistsError):
                safe_io.reserve_temp_path(os.path.join(self.folder, "movie_temp.wav"))
            with patch("modules.safe_io._create_private_dir", return_value=first.dir_path):
                with self.assertRaises(FileExistsError):
                    safe_io.reserve_temp_path(os.path.join(self.folder, "movie_temp.wav"))

    # --- cleanup_temp_files --------------------------------------------------

    def test_cleanup_temp_files_never_touches_scratch_dirs(self):
        # A planted directory (or junction) carrying a scratch-looking name must
        # never be walked or removed by the per-video cleanup scan.
        planted = os.path.join(self.folder, f"{safe_io.SCRATCH_PREFIX}{safe_io.scratch_tag('movie_temp')}-deadbeef")
        os.mkdir(planted)
        with open(os.path.join(planted, "precious.txt"), "w", encoding="utf-8") as fh:
            fh.write("precious")
        with open(os.path.join(self.folder, "movie_temp.wav"), "w", encoding="utf-8") as fh:
            fh.write("")
        utils.cleanup_temp_files(self.folder, "movie", "movie.mp4")
        self.assertTrue(os.path.isfile(os.path.join(planted, "precious.txt")))
        self.assertFalse(os.path.exists(os.path.join(self.folder, "movie_temp.wav")))


def _current_umask():
    value = os.umask(0)
    os.umask(value)
    return value


if __name__ == "__main__":
    unittest.main()
