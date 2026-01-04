import os
import sys
import pytest

# Add current directory and tests directory to path to import transform_coverage
sys.path.append(os.path.dirname(__file__))

def pytest_sessionfinish(session, exitstatus):
    """
    Called after whole test run finished, right before returning the exit status to the system.
    """
    coverage_xml = "coverage.xml"

    # Check if the coverage file exists in the root directory
    root_dir = session.config.rootdir
    coverage_path = os.path.join(root_dir, coverage_xml)

    if os.path.exists(coverage_path):
        try:
            from transform_coverage import transform_coverage
            print(f"\n[Auto-Badge] Updating coverage badge from {coverage_xml}...")
            transform_coverage(coverage_path)
            print("[Auto-Badge] Badge successfully updated.")
        except ImportError:
            # Fallback if import fails for some reason
            import subprocess
            transform_script = os.path.join(os.path.dirname(__file__), "transform_coverage.py")
            if os.path.exists(transform_script):
                print(f"\n[Auto-Badge] Running transformation script: {transform_script}")
                subprocess.run([sys.executable, transform_script, coverage_path], check=False)
        except Exception as e:
            print(f"\n[Auto-Badge] Failed to update badge: {e}")
