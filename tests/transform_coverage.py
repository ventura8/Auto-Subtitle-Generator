import xml.etree.ElementTree as ET
import sys
import os
import datetime

# =============================================================================
# BADGE GENERATION
# =============================================================================


def generate_badge(line_rate, output_path="assets/badge.svg"):
    """Generates a coverage badge SVG."""
    try:
        coverage = float(line_rate) * 100
    except ValueError:
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

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="20" role="img" """ \
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
    <g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" text-rendering="geometricPrecision" font-size="110">
        <text aria-hidden="true" x="{int(label_x)}" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)" """ \
        f"""textLength="{label_width*10 - 100}">{label_text}</text>
        <text x="{int(label_x)}" y="140" transform="scale(.1)" fill="#fff" textLength="{label_width*10 - 100}">{label_text}</text>
        <text aria-hidden="true" x="{int(value_x)}" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)" """ \
        f"""textLength="{value_width*10 - 100}">{label_text}</text>
        <text x="{int(value_x)}" y="140" transform="scale(.1)" fill="#fff" textLength="{value_width*10 - 100}">""" \
        f"""{value_text}</text>
    </g>
</svg>"""

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"Generated badge: {output_path} ({coverage_str})")

# =============================================================================
# COVERAGE TRANSFORMATION & SUMMARY
# =============================================================================


def _get_complexity_color(complexity):
    """Returns a color hex code based on cyclomatic complexity."""
    if complexity <= 5:
        return "brightgreen"
    if complexity <= 10:
        return "yellowgreen"
    if complexity <= 20:
        return "yellow"
    if complexity <= 30:
        return "orange"
    return "red"


def _calculate_file_complexity(file_path):
    """Calculates the average cyclomatic complexity for a file."""
    try:
        from radon.complexity import cc_visit  # type: ignore
        if not os.path.exists(file_path):
            return 0
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        blocks = cc_visit(code)
        if not blocks:
            return 1
        return sum(b.complexity for b in blocks) // len(blocks)
    except Exception:
        return 0


def _generate_markdown_summary(root, output_path="coverage_summary.md"):
    """Generates a markdown summary from the XML root."""
    total_coverage = int(float(root.get('line-rate', 0)) * 100)
    summary = "# Coverage and Complexity Report\n\n"
    summary += f"**Total Project Coverage: {total_coverage}%**\n\n"
    summary += f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    summary += "## File Breakdown\n\n"
    summary += "| File | Coverage | Complexity |\n"
    summary += "| :--- | :---: | :---: |\n"

    for pkg in root.findall('.//package'):
        pkg_name = pkg.get('name')
        l_rate = float(pkg.get('line-rate', 0)) * 100

        # Try to find a class file to calculate complexity
        complexity = 0
        cls = pkg.find('.//class')
        if cls is not None:
            complexity = _calculate_file_complexity(cls.get('filename'))

        comp_color = _get_complexity_color(complexity)
        comp_badge = (
            f"![{complexity}](https://img.shields.io/badge/"
            f"complexity-{complexity}-{comp_color})"
        )
        summary += f"| {pkg_name} | {int(l_rate)}% | {comp_badge} |\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Generated summary: {output_path}")


def transform_coverage(xml_file):
    """Transforms cobertura.xml by splitting classes into packages and generating reports."""
    if not os.path.exists(xml_file):
        print(f"Error: {xml_file} not found")
        sys.exit(1)

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        line_rate = root.get("line-rate", "0")
        generate_badge(line_rate)
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        sys.exit(1)

    packages_el = root.find('packages')
    if packages_el is None:
        _generate_markdown_summary(root)
        return

    # Collect all classes from all existing packages
    all_classes = []
    for pkg in packages_el.findall('package'):
        classes_el = pkg.find('classes')
        if classes_el is not None:
            all_classes.extend(classes_el.findall('class'))

    # Clear existing packages
    packages_el.clear()

    # Create new package per class
    for cls in all_classes:
        filename = cls.get('filename')
        pkg_name = filename

        new_pkg = ET.SubElement(packages_el, 'package')
        new_pkg.set('name', pkg_name)

        for attr in ['line-rate', 'branch-rate', 'complexity']:
            new_pkg.set(attr, cls.get(attr) or '0.0')

        new_classes = ET.SubElement(new_pkg, 'classes')
        new_classes.append(cls)

    tree.write(xml_file, encoding='UTF-8', xml_declaration=True)
    print(f"Successfully transformed {xml_file}: Split {len(all_classes)} classes into separate packages.")
    _generate_markdown_summary(root)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transform_coverage.py <cobertura.xml>")
        sys.exit(1)

    transform_coverage(sys.argv[1])
