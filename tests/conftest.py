import contextlib
import os
import shutil
import sys
from unittest.mock import MagicMock

import pytest

torch_mock = MagicMock()
torch_mock.__version__ = "2.0.1"
torch_mock.cuda = MagicMock()
torch_mock.cuda.is_available.return_value = True
torch_mock.cuda.device_count.return_value = 1
m_props = MagicMock()
m_props.total_memory = 24 * 1024**3
m_props.name = "Test GPU"
torch_mock.cuda.get_device_properties.return_value = m_props
sys.modules["torch"] = torch_mock
# Also mock submodules often used directly
sys.modules["torch.cuda"] = torch_mock.cuda

m_transformers = MagicMock()
m_transformers.__version__ = "4.30.0"
sys.modules["transformers"] = m_transformers

sys.modules["faster_whisper"] = MagicMock()
sys.modules["audio_separator"] = MagicMock()
sys.modules["audio_separator.separator"] = MagicMock()

# Cross-platform mocks for Linux CI
if sys.platform != "win32":
    sys.modules["winreg"] = MagicMock()
    import ctypes

    if getattr(ctypes, "windll", None) is None:
        setattr(ctypes, "windll", MagicMock())

# Standard pytest hook to fix sys.path
_p = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _p not in sys.path:
    sys.path.insert(0, _p)


@pytest.fixture(autouse=True)
def _no_work_dir_bleed_into_repo():
    """Fail any test that leaves a work directory or scratch entry in the repository root.

    Orchestration tests use cwd-relative fake video names; they must patch
    ``modules.workdir.ensure_work_dir`` instead of creating real directories.
    """
    initial_entries = set(_leaked_temp_entries())
    yield
    leaked = [entry for entry in _leaked_temp_entries() if entry not in initial_entries]
    _remove_leaked_entries(leaked)
    assert not leaked, f"test leaked temp entries into the repo root: {leaked}"


def _is_pipeline_temp_entry(entry):
    """Return True for a per-video work directory or a safe_io scratch entry."""
    return entry.endswith(".asg-temp") or entry.startswith(".asg-tmp-")


def _leaked_temp_entries():
    """Return repository-root entries that look like pipeline temp artifacts."""
    return sorted(entry for entry in os.listdir(_p) if _is_pipeline_temp_entry(entry))


def _remove_leaked_entries(entries):
    """Clean up leaked entries so one failing test does not cascade into the next.

    Entries may be directories (``*.asg-temp``, ``.asg-tmp-*``) or plain files;
    ``shutil.rmtree`` silently does nothing for a file under
    ``ignore_errors``, which would leave it to fail every later test too.
    """
    for entry in entries:
        path = os.path.join(_p, entry)
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path, ignore_errors=True)
            continue
        with contextlib.suppress(OSError):
            os.unlink(path)


def pytest_collection_modifyitems(items):
    """Mark real-dependency tests so ordinary validation can exclude them."""
    for item in items:
        if "tests/e2e/" in item.path.as_posix():
            item.add_marker("e2e")
