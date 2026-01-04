import sys
import os
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
sys.modules["librosa"] = MagicMock()
sys.modules["soundfile"] = MagicMock()
sys.modules["fastdtw"] = MagicMock()

# Globally block real subprocesses
m_sub = MagicMock()
m_sub.Popen.return_value.wait.return_value = 0
m_sub.Popen.return_value.returncode = 0
m_sub.Popen.return_value.stderr.readline.return_value = ""
m_sub.Popen.return_value.poll.return_value = 0
m_sub.Popen.return_value.communicate.return_value = (b"", b"")
sys.modules["subprocess"] = m_sub

# Patch multiprocessing and psutil for numeric hardware defaults
m_mp = MagicMock()
m_mp.cpu_count.return_value = 8
sys.modules["multiprocessing"] = m_mp

m_ps = MagicMock()
m_ps.virtual_memory.return_value.total = 16 * 1024**3
sys.modules["psutil"] = m_ps

# Cross-platform mocks for Linux CI
if sys.platform != "win32":
    sys.modules["winreg"] = MagicMock()
    import ctypes
    if not hasattr(ctypes, "windll"):
        ctypes.windll = MagicMock()

# Standard pytest hook to fix sys.path
_p = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _p not in sys.path:
    sys.path.insert(0, _p)
