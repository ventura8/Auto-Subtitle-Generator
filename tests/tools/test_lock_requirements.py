"""The lock-to-requirements tool must audit the versions this platform actually installs."""

import importlib.util
import os
import sys
import tempfile
import unittest
from unittest.mock import patch


def _load_lock_requirements_module():
    module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lock_requirements.py")
    spec = importlib.util.spec_from_file_location("lock_requirements_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load tests/tools/lock_requirements.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lock_requirements = _load_lock_requirements_module()

TORCH_LOCK = {
    "package": [
        {"name": "torch", "version": "2.10.0", "groups": ["ml"], "markers": 'implementation_name != "cpython"'},
        {
            "name": "torch",
            "version": "2.14.0",
            "groups": ["ml"],
            "markers": 'platform_system == "Darwin" and implementation_name == "cpython"',
        },
        {
            "name": "torch",
            "version": "2.14.0+cu132",
            "groups": ["ml"],
            "markers": 'platform_system != "Darwin" and implementation_name == "cpython"',
        },
        {"name": "pyyaml", "version": "6.0.3", "groups": ["main"]},
        {"name": "pytest", "version": "9.1.1", "groups": ["dev"]},
    ]
}


class TestSelectRequirements(unittest.TestCase):
    def test_marker_excluded_torch_is_not_audited(self):
        # This is the bug: first-entry-wins picked the PyPy-only 2.10.0 and reported its CVEs.
        selected = lock_requirements.select_requirements(TORCH_LOCK)
        self.assertNotEqual(selected["torch"], "2.10.0")
        self.assertIn(selected["torch"], {"2.14.0", "2.14.0+cu132"})

    def test_only_the_entry_for_this_platform_survives(self):
        selected = lock_requirements.select_requirements(TORCH_LOCK)
        expected = "2.14.0" if sys.platform == "darwin" else "2.14.0+cu132"
        self.assertEqual(selected["torch"], expected)

    def test_dev_group_is_not_audited_but_main_is(self):
        selected = lock_requirements.select_requirements(TORCH_LOCK)
        self.assertIn("pyyaml", selected)
        self.assertNotIn("pytest", selected)

    def test_highest_applicable_version_wins_when_several_apply(self):
        lock = {"package": [{"name": "x", "version": "1.2.0", "groups": ["main"]}, {"name": "x", "version": "1.10.0", "groups": ["main"]}]}
        self.assertEqual(lock_requirements.select_requirements(lock)["x"], "1.10.0")

    def test_entries_missing_name_or_version_are_skipped(self):
        lock = {"package": [{"name": "x", "groups": ["main"]}, {"version": "1.0", "groups": ["main"]}]}
        self.assertEqual(lock_requirements.select_requirements(lock), {})


class TestMarkerApplies(unittest.TestCase):
    def test_empty_marker_applies(self):
        self.assertTrue(lock_requirements.marker_applies(None))
        self.assertTrue(lock_requirements.marker_applies(""))

    def test_marker_for_another_platform_does_not_apply(self):
        self.assertFalse(lock_requirements.marker_applies('platform_system == "Plan9"'))

    def test_unparseable_marker_is_kept_rather_than_silently_dropped(self):
        self.assertTrue(lock_requirements.marker_applies("this is not a marker"))

    def test_poetry_empty_marker_never_applies(self):
        # Poetry writes "<empty>" for an entry no environment installs; it must
        # not win the highest-version pick over the version actually installed.
        self.assertFalse(lock_requirements.marker_applies("<empty>"))
        self.assertFalse(lock_requirements.marker_applies({"ml": "<empty>"}))
        lock = {
            "package": [
                {"name": "nvidia-cusparse", "version": "12.7.10.1", "groups": ["ml"]},
                {"name": "nvidia-cusparse", "version": "12.8.2.51", "groups": ["ml"], "markers": "<empty>"},
            ]
        }
        self.assertEqual(lock_requirements.select_requirements(lock), {"nvidia-cusparse": "12.7.10.1"})

    def test_per_group_marker_table_uses_audited_groups_only(self):
        windows_only = 'platform_system == "Windows"'
        never = 'platform_system == "Plan9"'
        # Applies when any audited group's marker applies.
        self.assertEqual(lock_requirements.marker_applies({"ml": windows_only, "dev": never}), sys.platform == "win32")
        # A table with only non-audited groups does not exclude the entry.
        self.assertTrue(lock_requirements.marker_applies({"dev": never}))
        # A table whose audited marker never applies excludes it.
        self.assertFalse(lock_requirements.marker_applies({"main": never}))


class TestWriteRequirements(unittest.TestCase):
    def test_writes_sorted_pins_for_this_platform(self):
        with tempfile.TemporaryDirectory() as folder:
            lock_path = os.path.join(folder, "poetry.lock")
            with open(lock_path, "w", encoding="utf-8") as handle:
                handle.write(
                    '[[package]]\nname = "zlib-ng"\nversion = "1.0"\ngroups = ["main"]\n\n'
                    '[[package]]\nname = "alpha"\nversion = "2.0"\ngroups = ["ml"]\n\n'
                    '[[package]]\nname = "torch"\nversion = "2.10.0"\ngroups = ["ml"]\nmarkers = \'implementation_name != "cpython"\'\n'
                )
            out = os.path.join(folder, "req.txt")
            count = lock_requirements.write_requirements(out, lock_path)
            with open(out, encoding="utf-8") as handle:
                lines = handle.read().splitlines()
        self.assertEqual(count, 2)
        self.assertEqual(lines, ["alpha==2.0", "zlib-ng==1.0"])

    def test_main_prints_summary_and_requires_output_path(self):
        with tempfile.TemporaryDirectory() as folder:
            lock_path = os.path.join(folder, "poetry.lock")
            with open(lock_path, "w", encoding="utf-8") as handle:
                handle.write('[[package]]\nname = "a"\nversion = "1"\ngroups = ["main"]\n')
            out = os.path.join(folder, "req.txt")
            with patch("builtins.print") as mock_print:
                self.assertEqual(lock_requirements.main(["prog", out, lock_path]), 0)
            self.assertIn("1 packages", mock_print.call_args[0][0])
        with patch("builtins.print"):
            self.assertEqual(lock_requirements.main(["prog"]), 2)


if __name__ == "__main__":
    unittest.main()
