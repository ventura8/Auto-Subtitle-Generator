import os
import sys
from unittest.mock import MagicMock

torch_mock = MagicMock()
torch_mock.__version__ = "2.0.1"
torch_mock.cuda = MagicMock()
torch_mock.cuda.is_available.return_value = True
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
