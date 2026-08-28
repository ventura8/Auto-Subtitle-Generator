"""Hardware detection and system information utilities."""

import platform
import subprocess
import sys
from typing import Any

winreg: Any | None

try:
    import winreg as _winreg
except ImportError:
    winreg = None
else:
    winreg = _winreg


def _get_linux_cpu_name():
    """Extract CPU model name from Linux /proc/cpuinfo."""
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def _get_macos_cpu_name():
    """Extract CPU brand string on macOS."""
    try:
        output = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], timeout=5, stderr=subprocess.DEVNULL)
        return output.decode("utf-8").strip()
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _get_windows_cpu_name():
    """Extract CPU name from Windows registry."""
    if sys.platform == "win32" and winreg is not None:
        try:
            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)
            processor_name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
            return processor_name.strip()
        except OSError:
            pass
    return None


def _get_unix_cpu_name():
    """Extract CPU name on Linux or macOS."""
    if sys.platform.startswith("linux"):
        return _get_linux_cpu_name()
    if sys.platform == "darwin":
        return _get_macos_cpu_name()
    return None


def get_cpu_name():
    """Returns the processor name across Windows, Linux, and macOS."""
    return _get_windows_cpu_name() or _get_unix_cpu_name() or platform.processor() or "Unknown CPU"
