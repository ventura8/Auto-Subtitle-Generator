"""Hardware detection and system information utilities."""

import platform
import sys
from typing import Any

winreg: Any | None

try:
    import winreg as _winreg
except ImportError:
    winreg = None
else:
    winreg = _winreg


def get_cpu_name():
    """Returns the processor name."""
    if sys.platform == "win32" and winreg is not None:
        try:
            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)
            processor_name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
            return processor_name.strip()
        except OSError:
            pass
    return platform.processor() or "Unknown CPU"
