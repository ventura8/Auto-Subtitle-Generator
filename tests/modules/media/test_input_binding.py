"""Tests for descriptor-bound input handling."""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

from modules.media import input_binding
from modules.media.input_binding import InputRefusedError, bind_input, media_source, rewind_inputs, set_input_root, stat_input


class InputBindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = self.tmp.name
        self.video = os.path.join(self.folder, "clip.mp4")
        with open(self.video, "wb") as handle:
            handle.write(b"original")
        self.secret = os.path.join(self.folder, "secret.mp4")
        with open(self.secret, "wb") as handle:
            handle.write(b"secret")
        self.addCleanup(set_input_root, None)

    def _symlink_or_skip(self, target, link):
        try:
            os.symlink(target, link)
        except (OSError, NotImplementedError) as e:
            raise unittest.SkipTest(f"symlinks unavailable: {e}")

    def test_unbound_path_passes_through(self):
        self.assertEqual(media_source(self.video), (self.video, ()))

    def test_bound_input_reads_the_opened_file_not_the_pathname(self):
        with bind_input(self.video) as bound:
            source, pass_fds = media_source(self.video)
            if sys.platform == "win32":
                self.assertEqual((source, pass_fds), (self.video, ()))
            else:
                self.assertEqual(source, f"/dev/fd/{bound.fd}")
                self.assertEqual(pass_fds, (bound.fd,))
            bound_id = os.fstat(bound.fd).st_ino
            if sys.platform == "win32":
                # The held handle denies rename/delete, so the name cannot be re-pointed at all.
                with self.assertRaises(PermissionError):
                    os.remove(self.video)
            else:
                # Swap the name for a symlink to another file mid-run: the descriptor is unaffected.
                os.remove(self.video)
                self._symlink_or_skip(self.secret, self.video)
            self.assertEqual(os.fstat(bound.fd).st_ino, bound_id)
            self.assertEqual(os.read(bound.fd, 8), b"original")
        self.assertIsNone(bound.fd)
        self.assertEqual(media_source(self.video), (self.video, ()))

    def test_symlink_input_is_refused(self):
        link = os.path.join(self.folder, "planted.mp4")
        self._symlink_or_skip(self.secret, link)
        with self.assertRaises(InputRefusedError):
            with bind_input(link):
                pass
        with self.assertRaises(InputRefusedError):
            input_binding._open_regular_fallback(link)

    def test_missing_and_non_regular_inputs_are_refused(self):
        def _bind(path):
            with bind_input(path):
                pass

        missing = os.path.join(self.folder, "ghost.mp4")
        with self.assertRaises(InputRefusedError):
            _bind(missing)
        with self.assertRaises(InputRefusedError):
            _bind(self.folder)

    def test_non_strict_bind_yields_none_and_leaves_no_binding(self):
        missing = os.path.join(self.folder, "ghost.mp4")
        with bind_input(missing, strict=False) as bound:
            self.assertIsNone(bound)
            self.assertEqual(media_source(missing), (missing, ()))

    def test_nested_bind_reuses_outer_descriptor(self):
        with bind_input(self.video) as outer:
            with bind_input(self.video) as inner:
                self.assertIs(inner, outer)
            # Inner exit must not close the outer descriptor.
            os.fstat(outer.fd)
            self.assertEqual(media_source(self.video)[0], media_source(outer.path)[0])

    def test_fallback_open_accepts_regular_file(self):
        fd = input_binding._open_regular_fallback(self.video)
        try:
            self.assertEqual(os.read(fd, 8), b"original")
        finally:
            os.close(fd)

    def test_fallback_refuses_identity_change_between_lstat_and_open(self):
        real = os.lstat(self.video)
        swapped = os.stat_result((real.st_mode, real.st_ino + 1, real.st_dev, 1, 0, 0, 8, 0, 0, 0))
        with patch("os.fstat", return_value=swapped):
            with self.assertRaises(InputRefusedError):
                input_binding._open_regular_fallback(self.video)

    def test_binding_is_released_on_error_inside_block(self):
        def _raise_inside_binding():
            with bind_input(self.video):
                raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            _raise_inside_binding()
        self.assertEqual(media_source(self.video), (self.video, ()))

    def test_stat_input_uses_the_bound_descriptor(self):
        with bind_input(self.video) as bound:
            self.assertEqual(stat_input(self.video).st_ino, os.fstat(bound.fd).st_ino)
        self.assertEqual(stat_input(self.video).st_ino, os.stat(self.video).st_ino)

    def test_rewind_inputs_resets_offset_before_each_child(self):
        with bind_input(self.video) as bound:
            os.read(bound.fd, 4)
            self.assertEqual(os.lseek(bound.fd, 0, os.SEEK_CUR), 4)
            rewind_inputs(media_source(self.video)[1])
            self.assertEqual(os.lseek(bound.fd, 0, os.SEEK_CUR), 0 if sys.platform != "win32" else 4)

    def test_swapped_parent_directory_below_root_is_refused(self):
        victim_dir = os.path.join(self.folder, "victim")
        os.mkdir(victim_dir)
        os.rename(self.secret, os.path.join(victim_dir, "movie.mp4"))
        root = os.path.join(self.folder, "drop")
        os.mkdir(os.path.join(root))
        os.mkdir(os.path.join(root, "clips"))
        clip = os.path.join(root, "clips", "movie.mp4")
        with open(clip, "wb") as handle:
            handle.write(b"original")
        set_input_root(root)
        with bind_input(clip) as bound:
            self.assertEqual(os.read(bound.fd, 8), b"original")
        # After collection, clips/ is renamed away and a link to the victim directory takes its place.
        os.rename(os.path.join(root, "clips"), os.path.join(root, "clips.bak"))
        self._symlink_or_skip(victim_dir, os.path.join(root, "clips"))
        self.assertTrue(os.path.isfile(clip))
        with self.assertRaises(InputRefusedError):
            with bind_input(clip):
                pass
        if input_binding._DIR_FD_SUPPORTED:
            with self.assertRaises(InputRefusedError):
                input_binding._open_regular_fallback(clip)

    def test_path_outside_root_falls_back_to_parent_open(self):
        set_input_root(os.path.join(self.folder, "elsewhere"))
        with bind_input(self.video) as bound:
            self.assertEqual(os.read(bound.fd, 8), b"original")

    def test_is_link_detects_symlinks_and_junctions(self):
        link = os.path.join(self.folder, "link.mp4")
        self._symlink_or_skip(self.video, link)
        self.assertTrue(input_binding.is_link(link))
        self.assertFalse(input_binding.is_link(self.video))
        with patch("os.path.islink", return_value=False), patch("os.path.isjunction", return_value=True, create=True):
            self.assertTrue(input_binding.is_link(self.folder))

    def test_dotdot_prefixed_child_name_stays_inside_the_root(self):
        root = os.path.join(self.folder, "drop")
        os.makedirs(os.path.join(root, "..clips"))
        clip = os.path.join(root, "..clips", "movie.mp4")
        with open(clip, "wb") as handle:
            handle.write(b"original")
        set_input_root(root)
        self.assertEqual(input_binding._split_trusted(clip), (os.path.realpath(root), ["..clips", "movie.mp4"]))
        with bind_input(clip) as bound:
            self.assertEqual(os.read(bound.fd, 8), b"original")

    def test_root_replaced_by_a_link_after_selection_is_refused(self):
        root = os.path.join(self.folder, "drop")
        os.mkdir(root)
        clip = os.path.join(root, "movie.mp4")
        with open(clip, "wb") as handle:
            handle.write(b"original")
        set_input_root(root)
        os.rename(root, root + ".bak")
        victim_dir = os.path.join(self.folder, "victim")
        os.mkdir(victim_dir)
        os.rename(self.secret, os.path.join(victim_dir, "movie.mp4"))
        self._symlink_or_skip(victim_dir, root)
        self.assertTrue(os.path.isfile(clip))
        with self.assertRaises(InputRefusedError):
            with bind_input(clip):
                pass
        with self.assertRaises(InputRefusedError):
            input_binding._open_regular_fallback(clip)

    def test_selected_root_that_is_itself_a_symlink_is_honoured(self):
        # A user may legitimately drag a symlinked folder: the target is resolved at selection time.
        real_root = os.path.join(self.folder, "real-root")
        os.mkdir(real_root)
        clip_real = os.path.join(real_root, "movie.mp4")
        with open(clip_real, "wb") as handle:
            handle.write(b"original")
        linked_root = os.path.join(self.folder, "linked-root")
        self._symlink_or_skip(real_root, linked_root)
        set_input_root(linked_root)
        with bind_input(os.path.join(linked_root, "movie.mp4")) as bound:
            self.assertEqual(os.read(bound.fd, 8), b"original")

    def test_fallback_platform_hands_ffmpeg_a_private_copy(self):
        with patch.object(input_binding, "_IS_POSIX", False), patch.object(input_binding, "_DIR_FD_SUPPORTED", False):
            with bind_input(self.video) as bound:
                source, pass_fds = media_source(self.video)
                self.assertEqual(pass_fds, ())
                self.assertNotEqual(source, self.video)
                self.assertTrue(os.path.basename(os.path.dirname(source)).startswith(input_binding._PRIVATE_COPY_PREFIX))
                # The copy lives in the user's temp dir, never inside the (attacker-controlled) input tree.
                self.assertEqual(os.path.realpath(os.path.dirname(os.path.dirname(source))), os.path.realpath(tempfile.gettempdir()))
                self.assertNotEqual(os.path.commonpath([source, self.folder]), self.folder)
                with open(source, "rb") as handle:
                    self.assertEqual(handle.read(), b"original")
                # Repeated lookups reuse the same copy; a nested bind shares it.
                self.assertEqual(media_source(self.video)[0], source)
                with bind_input(self.video) as inner:
                    self.assertEqual(inner.media_source()[0], source)
                self.assertTrue(os.path.exists(source))
                copy_dir = os.path.dirname(source)
            self.assertFalse(os.path.exists(copy_dir))
            self.assertIsNone(bound.fd)

    def test_fallback_copy_is_unaffected_by_a_parent_swap_after_it_is_made(self):
        root = os.path.join(self.folder, "drop")
        os.makedirs(os.path.join(root, "clips"))
        clip = os.path.join(root, "clips", "movie.mp4")
        with open(clip, "wb") as handle:
            handle.write(b"original")
        set_input_root(root)
        with patch.object(input_binding, "_IS_POSIX", False), patch.object(input_binding, "_DIR_FD_SUPPORTED", False):
            with bind_input(clip):
                source, _ = media_source(clip)
                # Attacker swaps clips/ for a link to their own tree and plants a same-named file there.
                os.rename(os.path.join(root, "clips"), os.path.join(root, "clips.bak"))
                attacker = os.path.join(self.folder, "attacker")
                os.mkdir(attacker)
                self._symlink_or_skip(attacker, os.path.join(root, "clips"))
                with open(os.path.join(attacker, "movie.mp4"), "wb") as handle:
                    handle.write(b"planted!")
                self.assertEqual(media_source(clip)[0], source)
                with open(source, "rb") as handle:
                    self.assertEqual(handle.read(), b"original")
            self.assertFalse(os.path.exists(os.path.dirname(source)))


if __name__ == "__main__":
    unittest.main()
