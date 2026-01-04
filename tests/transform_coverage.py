import xml.etree.ElementTree as ET
import sys
import os
import datetime

def generate_badge(line_rate, output_path="assets/badge.svg"):
    try:
        coverage = float(line_rate) * 100
    except ValueError:
        coverage = 0.0

    color = "#e05d44" # red
    if coverage >= 95:
        color = "#4c1" # brightgreen
    elif coverage >= 90:
         color = "#97ca00" # green
    elif coverage >= 75:
        color = "#dfb317" # yellow
    elif coverage >= 50:
        color = "#fe7d37" # orange

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

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="20" role="img" aria-label="{label_text}: {value_text}">
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
        <text aria-hidden="true" x="{int(label_x)}" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)" textLength="{label_width*10 - 100}">{label_text}</text>
        <text x="{int(label_x)}" y="140" transform="scale(.1)" fill="#fff" textLength="{label_width*10 - 100}">{label_text}</text>
        <text aria-hidden="true" x="{int(value_x)}" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)" textLength="{value_width*10 - 100}">{value_text}</text>
        <text x="{int(value_x)}" y="140" transform="scale(.1)" fill="#fff" textLength="{value_width*10 - 100}">{value_text}</text>
    </g>
</svg>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"Generated badge: {output_path} ({coverage_str})")

def get_complexity(file_path):
    """Calculates average cyclomatic complexity using radon."""
    try:
        from radon.complexity import cc_visit
        from radon.cli.harvest import CCHarvester

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
        blocks = cc_visit(code)
        if not blocks:
            return 0
        total_cc = sum(b.complexity for b in blocks)
        return round(total_cc / len(blocks), 2)
    except Exception as e:
        print(f"[Debug] Complexity calculation Error for {file_path}: {e}")
        return 0


def generate_markdown_summary(root, output_path="coverage_summary.md"):
    """Generates a GitHub-flavored Markdown summary table from Cobertura XML."""
    line_rate = float(root.get("line-rate", 0)) * 100
    branch_rate = float(root.get("branch-rate", 0)) * 100
    
    summary = []
    summary.append("## 📊 Code Coverage Report")
    summary.append(f"**Total Coverage:** `{line_rate:.2f}%`")
    summary.append(f"**Branch Coverage:** `{branch_rate:.2f}%`")
    summary.append(f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append("")
    summary.append("| File | Coverage | Branches | Complexity |")
    summary.append("| :--- | :---: | :---: | :---: |")
    
    packages = root.find('packages')
    if packages is not None:
        for pkg in packages.findall('package'):
            pkg_name = pkg.get('name', 'unknown')
            l_rate = float(pkg.get('line-rate', 0)) * 100
            b_rate = float(pkg.get('branch-rate', 0)) * 100
            
            # Try to get complexity from XML or calculate it
            comp = pkg.get('complexity', '0')
            if comp == '0':
                # Attempt to find the file and calculate
                classes = pkg.find('classes')
                if classes is not None:
                    cls = classes.find('class')
                    if cls is not None:
                        filename = cls.get('filename')
                        if filename:
                            # Try to find the file: absolute, relative, or basename
                            candidates = [filename, os.path.basename(filename)]
                            target_file = None
                            for cand in candidates:
                                if os.path.exists(cand):
                                    target_file = cand
                                    break
                            
                            if target_file:
                                c_val = get_complexity(target_file)
                                print(f"[Debug] Found file {target_file}, complexity: {c_val}")
                                comp = str(c_val)
                            else:
                                print(f"[Debug] Could not find file for complexity: {filename} (Tried candidates: {candidates})")
                            
                            if pkg_name == ".":
                                pkg_name = os.path.basename(filename) if filename else pkg_name

            summary.append(f"| `{pkg_name}` | **{int(round(l_rate))}%** | {b_rate:.1f}% | {comp} |")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary))
    print(f"Generated Markdown summary: {output_path}")

def transform_coverage(xml_file):
    if not os.path.exists(xml_file):
        print(f"Error: {xml_file} not found")
        sys.exit(1)

    # Ensure output directory exists if outputting to file
    # (Though generate_badge takes output_path, we rely on default or passed one.
    # Here we hardcoded default in generate_badge signature)
    # Let's ensure 'assets' exists just in case if using default.
    if not os.path.exists("assets"):
        try:
            os.makedirs("assets")
        except: pass

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Generate badge from root line-rate
        root_line_rate = root.get("line-rate", "0")
        generate_badge(root_line_rate)
        # Summary will be generated after transformation below

    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        sys.exit(1)

    packages_el = root.find('packages')
    if packages_el is None:
        print("No <packages> element found")
        # Ensure we don't exit if we just want the badge, but the user requested transformation too.
        # Minimal xml might not have packages if empty execution.
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
        # Use basename or relative path as package name
        pkg_name = filename 
        
        new_pkg = ET.SubElement(packages_el, 'package')
        new_pkg.set('name', pkg_name)

        # Copy rate attributes from class to package
        for attr in ['line-rate', 'branch-rate', 'complexity']:
            if val := cls.get(attr):
                new_pkg.set(attr, val)
            else:
                new_pkg.set(attr, '0.0')

        # Create classes container
        new_classes = ET.SubElement(new_pkg, 'classes')
        new_classes.append(cls)

    tree.write(xml_file, encoding='UTF-8', xml_declaration=True)
    print(f"Successfully transformed {xml_file}: Split {len(all_classes)} classes into separate packages.")
    
    # Generate summary after transformation so all packages are split
    generate_markdown_summary(root)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transform_coverage.py <cobertura.xml>")
        sys.exit(1)

    transform_coverage(sys.argv[1])
