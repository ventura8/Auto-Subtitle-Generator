"""Tests for shared translator helpers."""

import sys
from unittest.mock import MagicMock, patch

from modules.translators import common


def test_import_transformers_module_preserves_torchaudio_imports_after_error():
    """A torchaudio failure must not poison later imports through sys.modules."""
    existing_torchaudio = sys.modules.get("torchaudio")
    sys.modules.pop("torchaudio", None)
    try:

        def mock_import(name, *args, **kwargs):
            if name == "torchaudio":
                raise RuntimeError("CUDA mismatch")
            return object()

        with patch("modules.translators.common.importlib.import_module", side_effect=mock_import):
            common.import_transformers_module()
        assert "torchaudio" not in sys.modules
    finally:
        if existing_torchaudio is not None:
            sys.modules["torchaudio"] = existing_torchaudio


def test_import_transformers_module_when_torchaudio_already_in_modules():
    """Transformers import must preserve existing torchaudio module when present."""
    mock_mod = sys.modules.get("sys")
    missing = object()
    existing_torchaudio = sys.modules.get("torchaudio", missing)
    try:
        if mock_mod is not None:
            sys.modules["torchaudio"] = mock_mod
        with patch("modules.translators.common.importlib.import_module", return_value=object()):
            common.import_transformers_module()
        assert sys.modules.get("torchaudio") is mock_mod
    finally:
        sys.modules.pop("torchaudio", None)
        if existing_torchaudio is not missing:
            sys.modules["torchaudio"] = existing_torchaudio


def test_resolve_device_map_uses_cuda_only_when_available():
    with patch("modules.translators.common.torch", None):
        assert common.resolve_device_map() is None


def test_resolve_device_map_uses_mps_when_cuda_is_unavailable():
    torch_module = MagicMock()
    torch_module.cuda.is_available.return_value = False
    torch_module.backends.mps.is_available.return_value = True

    with patch("modules.translators.common.torch", torch_module):
        assert common.resolve_device_map() == "mps"


def test_add_device_load_kwargs_includes_cuda_precision():
    torch_module = MagicMock()
    torch_module.float16 = object()

    kwargs = common.add_device_load_kwargs({}, "cuda:0", torch_module)

    assert kwargs["device_map"] == "cuda:0"
    assert kwargs["dtype"] is torch_module.float16


def test_add_device_load_kwargs_ignores_missing_device_map():
    assert common.add_device_load_kwargs({}, None, None) == {}


def test_add_device_load_kwargs_omits_cuda_dtype_for_mps():
    torch_module = MagicMock()
    torch_module.float16 = object()

    kwargs = common.add_device_load_kwargs({}, "mps", torch_module)

    assert kwargs == {"device_map": "mps"}


def test_is_corrupt_model_error_matches_known_tokens():
    assert common.is_corrupt_model_error(RuntimeError("file is not a valid safetensors archive"))
    assert common.is_corrupt_model_error(ValueError("invalid safetensors header"))
    assert common.is_corrupt_model_error(OSError("piece size is not valid"))
    assert not common.is_corrupt_model_error(RuntimeError("CUDA out of memory"))


def test_purge_hf_model_cache_removes_directory():
    with patch("os.path.isdir", return_value=True), patch("shutil.rmtree") as mock_rmtree:
        common.purge_hf_model_cache("facebook/nllb-200-1.3B")
        mock_rmtree.assert_called_once()


def test_load_with_cache_recovery_success():
    loader = MagicMock(return_value="model_instance")
    result = common.load_with_cache_recovery(loader, "model_id", {"key": "val"})
    assert result == "model_instance"
    loader.assert_called_once_with("model_id", key="val")


def test_load_with_cache_recovery_corrupt_cache_retry():
    loader = MagicMock(side_effect=[RuntimeError("invalid safetensors header"), "recovered_model"])
    logger = MagicMock()
    with patch("modules.translators.common.purge_hf_model_cache") as mock_purge:
        result = common.load_with_cache_recovery(loader, "model_id", {"param": 1}, logger=logger, model_label="TestModel")
    assert result == "recovered_model"
    mock_purge.assert_called_once_with("model_id")
    logger.warning.assert_called_once()
    assert loader.call_count == 2


def test_load_with_cache_recovery_oserror_fallback():
    loader = MagicMock(side_effect=[OSError("Network down"), "offline_model"])
    result = common.load_with_cache_recovery(loader, "model_id", {"param": 1})
    assert result == "offline_model"
    assert loader.call_args_list[0].kwargs == {"param": 1}
    assert loader.call_args_list[1].kwargs == {"param": 1, "local_files_only": True}


def test_load_with_cache_recovery_reraises_unknown_error():
    loader = MagicMock(side_effect=TypeError("Unknown error"))
    import pytest

    with pytest.raises(TypeError):
        common.load_with_cache_recovery(loader, "model_id")
