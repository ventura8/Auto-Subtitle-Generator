"""Separator checkpoint purging in :mod:`modules.runtime.model_cache`.

Split out of ``test_models.py``: these exercise the cache-purge helpers rather
than the model manager, and keeping them there pushed that module's
maintainability index below the A grade the project enforces.
"""

import os
import stat
import unittest
from unittest.mock import patch

from modules import models


def _raise_missing():
    """Stand in for lstat() on a path that does not exist."""
    raise FileNotFoundError("no such directory")


def _fake_dir_stat(mode=stat.S_IFDIR | 0o755, uid=None):
    """Build an os.stat_result standing in for a real fstat() of a bound directory."""
    if uid is None:
        geteuid = getattr(os, "geteuid", None)
        uid = geteuid() if geteuid is not None else 0
    return os.stat_result((mode, 0, 0, 1, uid, 0, 0, 0, 0, 0))


class TestSeparatorCheckpointPurge(unittest.TestCase):
    def test_purge_cached_separator_checkpoint(self):
        fake_files = [
            "test_model.ckpt",
            "test_model.yaml",
            "test_model.json",
            "test_model_backup.ckpt",
            "test_model.notes.txt",
            "other.ckpt",
        ]
        with (
            patch("modules.runtime.model_cache.open_dir_handle", side_effect=lambda d: 9 if d == "/fake/dir" else None),
            patch("os.fstat", return_value=_fake_dir_stat()),
            patch("os.listdir", return_value=fake_files),
            patch("os.close"),
            patch("os.unlink") as mock_unlink,
        ):
            models._purge_cached_separator_checkpoint("test_model.ckpt", "/fake/dir")
            self.assertEqual(mock_unlink.call_count, 3)
            removed = [call.args[0] for call in mock_unlink.call_args_list]
            bound_to_descriptor = [call.kwargs.get("dir_fd") for call in mock_unlink.call_args_list]
            self.assertEqual(bound_to_descriptor, [9, 9, 9], "unlink must be relative to the held descriptor")
            self.assertIn("test_model.ckpt", removed)
            self.assertIn("test_model.yaml", removed)
            self.assertIn("test_model.json", removed)
            self.assertNotIn("test_model_backup.ckpt", removed)
            self.assertNotIn("test_model.notes.txt", removed)

    def test_purge_abandons_when_binding_fails_on_a_platform_that_supports_it(self):
        """A failed bind where descriptors work can mean a link or a swap: purge nothing."""
        with (
            patch("modules.runtime.model_cache.DIR_FD_SUPPORTED", True),
            patch("modules.runtime.model_cache.open_dir_handle", return_value=None),
            patch("os.listdir", return_value=["test_model.ckpt"]),
            patch("os.unlink") as mock_unlink,
        ):
            models._purge_cached_separator_checkpoint("test_model.ckpt", "/fake/dir")
            mock_unlink.assert_not_called()

    def test_purge_falls_back_to_pathnames_where_descriptors_are_unsupported(self):
        """Windows has no dir_fd support at all; recovery must still purge there."""
        with (
            patch("modules.runtime.model_cache.DIR_FD_SUPPORTED", False),
            patch("modules.runtime.model_cache.open_dir_handle", return_value=None),
            patch(
                "os.lstat",
                side_effect=lambda d: _fake_dir_stat() if d == "/fake/dir" else _fake_dir_stat(mode=stat.S_IFREG | 0o644),
            ),
            patch("os.listdir", return_value=["test_model.ckpt", "keep.txt"]),
            patch("os.unlink") as mock_unlink,
        ):
            models._purge_cached_separator_checkpoint("test_model.ckpt", "/fake/dir")
            mock_unlink.assert_called_once_with(os.path.join("/fake/dir", "test_model.ckpt"))

    def test_pathname_fallback_survives_an_unreadable_directory(self):
        """listdir failing must not leave the entry list unbound."""
        with (
            patch("modules.runtime.model_cache.DIR_FD_SUPPORTED", False),
            patch("modules.runtime.model_cache.open_dir_handle", return_value=None),
            patch("os.lstat", side_effect=lambda d: _fake_dir_stat() if d == "/fake/dir" else _raise_missing()),
            patch("os.listdir", side_effect=PermissionError("denied")),
            patch("os.unlink") as mock_unlink,
        ):
            models._purge_cached_separator_checkpoint("test_model.ckpt", "/fake/dir")
            mock_unlink.assert_not_called()

    def test_pathname_fallback_never_unlinks_through_a_link(self):
        """A link planted where a checkpoint belongs is skipped, not unlinked."""

        def fake_lstat(path):
            if path == "/fake/dir":
                return _fake_dir_stat()
            if path == os.path.join("/fake/dir", "test_model.ckpt"):
                return _fake_dir_stat(mode=stat.S_IFLNK | 0o777)
            return _raise_missing()

        with (
            patch("modules.runtime.model_cache.DIR_FD_SUPPORTED", False),
            patch("modules.runtime.model_cache.open_dir_handle", return_value=None),
            patch("os.lstat", side_effect=fake_lstat),
            patch("os.listdir", return_value=["test_model.ckpt"]),
            patch("os.unlink") as mock_unlink,
        ):
            models._purge_cached_separator_checkpoint("test_model.ckpt", "/fake/dir")
            mock_unlink.assert_not_called()

    def test_pathname_fallback_refuses_a_symlinked_directory(self):
        with (
            patch("modules.runtime.model_cache.DIR_FD_SUPPORTED", False),
            patch("modules.runtime.model_cache.open_dir_handle", return_value=None),
            patch("os.lstat", return_value=_fake_dir_stat(mode=stat.S_IFLNK | 0o777)),
            patch("os.listdir", return_value=["test_model.ckpt"]),
            patch("os.unlink") as mock_unlink,
        ):
            models._purge_cached_separator_checkpoint("test_model.ckpt", "/fake/dir")
            mock_unlink.assert_not_called()

    def test_purge_cached_separator_checkpoint_skips_non_directory_descriptor(self):
        with (
            patch("modules.runtime.model_cache.open_dir_handle", return_value=9),
            patch("os.fstat", return_value=_fake_dir_stat(mode=stat.S_IFREG | 0o644)),
            patch("os.listdir", return_value=["test_model.ckpt"]),
            patch("os.close"),
            patch("os.unlink") as mock_unlink,
        ):
            models._purge_cached_separator_checkpoint("test_model.ckpt", "/fake/dir")
            mock_unlink.assert_not_called()

    def test_purge_cached_separator_checkpoint_skips_foreign_owned_directory(self):
        if not hasattr(os, "geteuid"):
            self.skipTest("ownership checks require POSIX uids")
        with (
            patch("modules.runtime.model_cache.open_dir_handle", return_value=9),
            patch("os.fstat", return_value=_fake_dir_stat(uid=os.geteuid() + 1)),
            patch("os.listdir", return_value=["test_model.ckpt"]),
            patch("os.close"),
            patch("os.unlink") as mock_unlink,
        ):
            models._purge_cached_separator_checkpoint("test_model.ckpt", "/fake/dir")
            mock_unlink.assert_not_called()

    def test_purge_cached_separator_checkpoint_skips_unstatable_descriptor(self):
        with (
            patch("modules.runtime.model_cache.open_dir_handle", return_value=9),
            patch("os.fstat", side_effect=OSError("gone")),
            patch("os.listdir", return_value=["test_model.ckpt"]),
            patch("os.close"),
            patch("os.unlink") as mock_unlink,
        ):
            models._purge_cached_separator_checkpoint("test_model.ckpt", "/fake/dir")
            mock_unlink.assert_not_called()
