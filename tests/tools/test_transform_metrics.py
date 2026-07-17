import importlib.util
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET


def _load_transform_metrics_module():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    module_path = os.path.join(repo_root, "tests", "tools", "transform_metrics.py")
    spec = importlib.util.spec_from_file_location("transform_metrics_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load tests/transform_metrics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTransformCoverage(unittest.TestCase):
    def setUp(self):
        self.coverage_tool = _load_transform_metrics_module()

    def test_generate_badge_uses_scaled_percentage_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            badge_path = os.path.join(temp_dir, "coverage.svg")
            self.coverage_tool.generate_badge(92, badge_path)

            self.assertTrue(os.path.exists(badge_path))
            with open(badge_path, "r", encoding="utf-8") as handle:
                content = handle.read()
            self.assertIn("92%", content)
            self.assertIn("#97ca00", content)

    def test_generate_badge_fractional_input_follows_scaled_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            badge_path = os.path.join(temp_dir, "coverage.svg")
            self.coverage_tool.generate_badge(91.6, badge_path)

            with open(badge_path, "r", encoding="utf-8") as handle:
                content = handle.read()
            # Contract: generate_badge expects a 0-100 coverage value.
            self.assertIn("92%", content)
            self.assertIn("#97ca00", content)

    def test_calculate_total_coverage_percent_aggregates_lines_and_branches(self):
        root = ET.Element(
            "coverage",
            {
                "lines-valid": "200",
                "lines-covered": "150",
                "branches-valid": "100",
                "branches-covered": "60",
            },
        )

        total = self.coverage_tool._calculate_total_coverage_percent(root)
        self.assertAlmostEqual(total, 70.0)

    def test_calculate_total_coverage_percent_falls_back_to_line_rate(self):
        root = ET.Element(
            "coverage",
            {
                "lines-valid": "0",
                "lines-covered": "0",
                "branches-valid": "0",
                "branches-covered": "0",
                "line-rate": "0.875",
            },
        )

        total = self.coverage_tool._calculate_total_coverage_percent(root)
        self.assertAlmostEqual(total, 87.5)

    def test_markdown_summary_uses_unavailable_complexity_sentinel(self):
        root = ET.Element("coverage", {"line-rate": "0.9"})
        packages = ET.SubElement(root, "packages")
        ET.SubElement(packages, "package", {"name": "modules/config.py", "line-rate": "1.0"})

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "coverage_summary.md")
            self.coverage_tool._generate_markdown_summary(root, output_path)

            with open(output_path, "r", encoding="utf-8") as handle:
                content = handle.read()

        self.assertIn("complexity--1-lightgrey", content)

    def test_markdown_summary_falls_back_to_package_complexity_attribute(self):
        root = ET.Element("coverage", {"line-rate": "0.9"})
        packages = ET.SubElement(root, "packages")
        ET.SubElement(
            packages,
            "package",
            {"name": "modules/translation.py", "line-rate": "0.8", "complexity": "7.0"},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "coverage_summary.md")
            self.coverage_tool._generate_markdown_summary(root, output_path)

            with open(output_path, "r", encoding="utf-8") as handle:
                content = handle.read()

        self.assertNotIn("N/A", content)
        self.assertIn("complexity-7-yellowgreen", content)

    def test_resolve_class_complexity_falls_back_when_xml_zero(self):
        cls = ET.Element(
            "class",
            {
                "filename": "modules/example.py",
                "complexity": "0",
            },
        )

        with unittest.mock.patch.object(self.coverage_tool, "_calculate_file_complexity", return_value=11) as mock_calc:
            resolved = self.coverage_tool._resolve_class_complexity(cls)

        self.assertEqual(resolved, 11)
        mock_calc.assert_called_once_with("modules/example.py")


if __name__ == "__main__":
    unittest.main()
