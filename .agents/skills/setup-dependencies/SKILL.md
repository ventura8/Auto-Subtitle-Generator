______________________________________________________________________

## name: setup-dependencies description: Bootstrap, install, or repair local development setup: Python 3.12.x (>=3.12,\<3.13), Poetry virtualenv, PyTorch CUDA 13.2 / cuDNN, and local FFmpeg on Windows.

# Setup Dependencies Skill

Use this skill when onboarding, bootstrapping, or repairing local environment dependencies, FFmpeg tools, and hardware acceleration on Windows.

## Goal

Bring the local machine to a fully functional state capable of running high-performance GPU subtitle extraction.

## Dependency Invariants

1. **Python Runtime**: Python 3.12 (specifically `>=3.12,<3.13`).
1. **Package Manager**: Poetry (`pyproject.toml` is the single source of truth for dependencies).
1. **PyTorch with CUDA 13.2**: Explicit wheel source from `https://download.pytorch.org/whl/cu132`.
1. **NVIDIA CUDA/cuDNN**: Windows-specific wheels `nvidia-cudnn-cu13` and `nvidia-cublas`.
1. **FFmpeg**: Standalone official static build installed into local environment or system PATH.

## Installation & Repair Workflow

### 1. Execute Setup Script

Run the idempotent PowerShell bootstrap script:

```powershell
.\install_dependencies.ps1
```

### 2. Verify Python Virtual Environment & Poetry

```powershell
# Verify Poetry environment info
poetry env info

# Ensure dependencies are locked and synced
poetry install --all-extras --sync
```

### 3. Verify CUDA Acceleration

```powershell
poetry run python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 4. Verify FFmpeg Availability

```powershell
poetry run python -c "import subprocess; from modules.media.ffmpeg_utils import get_ffmpeg_paths; ff, fp = get_ffmpeg_paths(); subprocess.run([ff, '-version'], check=True, stdout=subprocess.DEVNULL); subprocess.run([fp, '-version'], check=True, stdout=subprocess.DEVNULL); print('FFmpeg binaries verified:', ff, fp)"
```

### 5. Validate CLI Entrypoint

```powershell
poetry run python auto_subtitle.py --help
```
