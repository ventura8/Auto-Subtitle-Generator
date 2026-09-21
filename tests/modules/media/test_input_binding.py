"""Tests for descriptor-bound input handling."""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

from modules.media import input_binding
from modules.media.input_binding import InputRefusedError, bind_input, media_source


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
        with self.assertRaises(InputRefusedError):
            with bind_input(os.path.join(self.folder, "ghost.mp4")):
                pass
        with self.assertRaises(InputRefusedError):
            with bind_input(self.folder):
                pass

    def test_non_strict_bind_yields_none_and_leaves_no_binding(self):
        missing = os.path.join(self.folder, "ghost.mp4")
        with bind_input(missing, strict=False) as bound:
            self.assertIsNone(bound)
            self.assertEqual(media_source(missing), (missing, ()))

    def test_nested_bind_reuses_outer_descriptor(self):
        with bind_input(self.video) as outer:
            with bind_input(self.video) as inner:
                self.assertEqual(inner.fd, outer.fd)
                self.assertFalse(inner._owns_fd)
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
        with self.assertRaises(RuntimeError):
            with bind_input(self.video):
                raise RuntimeError("boom")
        self.assertEqual(media_source(self.video), (self.video, ()))


if __name__ == "__main__":
    unittest.main()
