"""Tests for input path normalization."""

from unittest.mock import patch

from modules.media import file_utils


def test_resolve_input_path_removes_matching_double_quotes():
    assert file_utils.resolve_input_path('  "video file.mp4"  ') == "video file.mp4"


def test_resolve_input_path_preserves_mismatched_quotes():
    assert file_utils.resolve_input_path("  \"video file.mp4'  ") == "\"video file.mp4'"


def test_resolve_input_path_removes_matching_single_quotes_from_prompt():
    with patch("builtins.input", return_value="  'video file.mp4'  "):
        assert file_utils.resolve_input_path(None) == "video file.mp4"
