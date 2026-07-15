import datetime
import importlib
import os
import sys
import xml.etree.ElementTree as ET

COMPLEXITY_UNAVAILABLE = -1

# =============================================================================
# BADGE GENERATION
# =============================================================================


def _calculate_total_coverage_percent(root):
    """Return total coverage using line+branch metrics when available."""
    try:
        lines_valid = int(root.get("lines-valid", 0))
        lines_covered = int(root.get("lines-covered", 0))
        branches_valid = int(root.get("branches-valid", 0))
        branches_covered = int(root.get("branches-covered", 0))

        total_valid = lines_valid + branches_valid
        total_covered = lines_covered + branches_covered

        if total_valid > 0:
            return (total_covered / total_valid) * 100

        line_rate = float(root.get("line-rate", 0))
        return line_rate * 100
    except (TypeError, ValueError):
        return 0.0


def generate_badge(coverage, output_path="assets/coverage.svg"):
    """Generates a coverage badge SVG."""
    try:
        coverage = float(coverage)
    except (TypeError, ValueError):
        coverage = 0.0

    color = "#e05d44"  # red
    if coverage >= 95:
        color = "#4c1"  # brightgreen
    elif coverage >= 90:
        color = "#97ca00"  # green
    elif coverage >= 75:
        color = "#dfb317"  # yellow
    elif coverage >= 50:
        color = "#fe7d37"  # orange

    coverage_str = f"{int(round(coverage))}%"
    label_text = "Coverage"
    value_text = coverage_str

    # Estimate widths
    # 6px approx per char + padding
    label_width = 61
    value_width = int(len(value_text) * 8.5) + 10

    total_width = label_width + value_width

    # Center positions
    label_x = label_width / 2.0 * 10
    value_x = (label_width + value_width / 2.0) * 10

    svg = (
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="20" role="img" """
        f"""aria-label="{label_text}: {value_text}">
    <title>{label_text}: {value_text}</title>
    <linearGradient id="s" x2="0" y2="100%">
        <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
        <stop offset="1" stop-opacity=".1"/>
    </linearGradient>
    <clipPath id="r">
        <rect width="{total_width}" height="20" rx="3" fill="#fff"/>
    </clipPath>
    <g clip-path="url(#r)">
        <rect width="{label_width}" height="20" fill="#555"/>
        <rect x="{label_width}" width="{value_width}" height="20" fill="{color}"/>
        <rect width="{total_width}" height="20" fill="url(#s)"/>
    </g>
    <g fill="#fff" text-anchor="middle"
       font-family="Verdana,Geneva,DejaVu Sans,sans-serif"
       text-rendering="geometricPrecision" font-size="110">
        <text aria-hidden="true" x="{int(label_x)}" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)" """
        f"""textLength="{label_width * 10 - 100}">{label_text}</text>
        <text x="{int(label_x)}" y="140" transform="scale(.1)" fill="#fff" textLength="{label_width * 10 - 100}">{label_text}</text>
        <text aria-hidden="true" x="{int(value_x)}" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)" """
        f"""textLength="{value_width * 10 - 100}">{value_text}</text>
        <text x="{int(value_x)}" y="140" transform="scale(.1)" fill="#fff" textLength="{value_width * 10 - 100}">"""
        f"""{value_text}</text>
    </g>
</svg>"""
    )

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"Generated badge: {output_path} ({coverage_str})")


# =============================================================================
# COVERAGE TRANSFORMATION & SUMMARY
# =============================================================================


def _get_complexity_color(complexity):
    """Returns a color hex code based on cyclomatic complexity."""
    if complexity in (COMPLEXITY_UNAVAILABLE, 0):
        return "lightgrey"
    if complexity <= 5:
        return "brightgreen"
    if complexity <= 10:
        return "yellowgreen"
    if complexity <= 20:
        return "yellow"
    if complexity <= 30:
        return "orange"
    return "red"


def _parse_complexity_value(raw_value):
    """Parses a complexity value into an integer or returns unavailable sentinel."""
    if raw_value is None:
        return COMPLEXITY_UNAVAILABLE
    try:
        parsed_value = int(float(raw_value))
        if parsed_value <= 0:
            return COMPLEXITY_UNAVAILABLE
        return parsed_value
    except (TypeError, ValueError):
        return COMPLEXITY_UNAVAILABLE


def _resolve_package_complexity(pkg, cls):
    """Resolves package complexity from source file analysis or XML attributes."""
    complexity = COMPLEXITY_UNAVAILABLE

    if cls is not None:
        complexity = _parse_complexity_value(cls.get("complexity"))
        if complexity == COMPLEXITY_UNAVAILABLE:
            class_file = cls.get("filename")
            if class_file:
                complexity = _calculate_file_complexity(class_file)

    if complexity == COMPLEXITY_UNAVAILABLE:
        complexity = _parse_complexity_value(pkg.get("complexity"))

    if complexity == COMPLEXITY_UNAVAILABLE:
        return COMPLEXITY_UNAVAILABLE
    return complexity


def _resolve_class_complexity(cls):
    """Resolves class complexity from XML first, then file analysis."""
    complexity = _parse_complexity_value(cls.get("complexity"))
    if complexity != COMPLEXITY_UNAVAILABLE:
        return complexity

    class_file = cls.get("filename")
    if class_file:
        complexity = _calculate_file_complexity(class_file)

    if complexity == COMPLEXITY_UNAVAILABLE:
        return COMPLEXITY_UNAVAILABLE
    return complexity


def _calculate_file_complexity(file_path):
    """Calculates the average cyclomatic complexity for a file."""
    try:
        radon_complexity = importlib.import_module("radon.complexity")
        cc_visit = getattr(radon_complexity, "cc_visit")

        if not os.path.exists(file_path):
            return COMPLEXITY_UNAVAILABLE
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
        blocks = cc_visit(code)
        if not blocks:
            return 1
        average_complexity = sum(b.complexity for b in blocks) / len(blocks)
        return int(average_complexity + 0.5)
    except Exception:
        return COMPLEXITY_UNAVAILABLE


def _generate_markdown_summary(root, output_path="coverage_summary.md"):
    """Generates a markdown summary from the XML root."""
    total_coverage = _calculate_total_coverage_percent(root)
    summary = "# Coverage and Complexity Report\n\n"
    summary += f"**Total Project Coverage: {total_coverage:.2f}%**\n\n"
    summary += f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    summary += "## File Breakdown\n\n"
    summary += "| File | Coverage | Complexity |\n"
    summary += "| :--- | :---: | :---: |\n"

    for pkg in root.findall(".//package"):
        pkg_name = pkg.get("name")
        l_rate = float(pkg.get("line-rate", 0)) * 100

        cls = pkg.find(".//class")
        complexity = _resolve_package_complexity(pkg, cls)

        comp_color = _get_complexity_color(complexity)
        complexity_label = str(complexity)
        comp_badge = f"![{complexity_label}](https://img.shields.io/badge/complexity-{complexity_label}-{comp_color})"
        summary += f"| {pkg_name} | {int(l_rate)}% | {comp_badge} |\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Generated summary: {output_path}")


def transform_coverage(xml_file, badge_output_path="assets/coverage.svg"):
    """Transforms cobertura.xml by splitting classes into packages and generating reports."""
    if not os.path.exists(xml_file):
        print(f"Error: {xml_file} not found")
        sys.exit(1)

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        total_coverage = _calculate_total_coverage_percent(root)
        generate_badge(total_coverage, badge_output_path)
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        sys.exit(1)

    packages_el = root.find("packages")
    if packages_el is None:
        _generate_markdown_summary(root)
        return

    # Collect all classes from all existing packages
    all_classes = []
    for pkg in packages_el.findall("package"):
        classes_el = pkg.find("classes")
        if classes_el is not None:
            all_classes.extend(classes_el.findall("class"))

    # Clear existing packages
    packages_el.clear()

    # Create new package per class
    for cls in all_classes:
        filename = cls.get("filename")
        pkg_name = filename
        class_complexity = _resolve_class_complexity(cls)
        cls.set("complexity", str(class_complexity))

        new_pkg = ET.SubElement(packages_el, "package")
        new_pkg.set("name", pkg_name)

        for attr in ["line-rate", "branch-rate"]:
            new_pkg.set(attr, cls.get(attr) or "0.0")
        new_pkg.set("complexity", str(class_complexity))

        new_classes = ET.SubElement(new_pkg, "classes")
        new_classes.append(cls)

    tree.write(xml_file, encoding="UTF-8", xml_declaration=True)
    print(f"Successfully transformed {xml_file}: Split {len(all_classes)} classes into separate packages.")
    _generate_markdown_summary(root)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transform_coverage.py <cobertura.xml> [badge_output_path]")
        sys.exit(1)

    badge_output = sys.argv[2] if len(sys.argv) > 2 else "assets/coverage.svg"
    transform_coverage(sys.argv[1], badge_output)
