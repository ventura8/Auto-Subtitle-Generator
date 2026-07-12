# Sets up the environment for auto_subtitle.py
# Optimization: RTX 5090 / CUDA 12.8 Stable
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$InformationPreference = "Continue"

Set-Location -Path $PSScriptRoot

Write-Information "=== Setting up Auto-Subtitle Generator Environment (RTX 5090 Ready) ==="

function Invoke-CheckedCommand {
    param(
        [string]$Executable,
        [string[]]$Arguments
    )

    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Executable $($Arguments -join ' ')"
    }
}

# 1. Check for Python
try {
    $pyVersion = python --version 2>&1
    Write-Information "Found Python: $pyVersion"
}
catch {
    Write-Warning "Python not found in PATH."
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Information "Attempting to install Python 3.12..."
        try {
            winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
            if ($LASTEXITCODE -ne 0) {
                throw "Winget failed to install Python 3.12."
            }
            Write-Information "`n[!] Python installed. Please restart script."
            exit
        }
        catch { Write-Error "Winget failed to install Python." }
    }
    else { Write-Error "Python not found. Please install Python 3.12 manually." }
}

# 2. Create Virtual Environment
Write-Information "`nStep 2: Setting up Python Virtual Environment..."
if (-not (Test-Path "$PSScriptRoot\.venv\Scripts\python.exe")) {
    Write-Information "Creating virtual environment..."
    $resolvedPyVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to resolve Python interpreter version."
    }

    [version]$minVersion = "3.12.0"
    [version]$maxVersionExclusive = "3.13.0"
    [version]$currentVersion = $resolvedPyVersion
    if ($currentVersion -lt $minVersion) {
        throw "Python 3.12+ is required to create .venv. Found $currentVersion"
    }
    if ($currentVersion -ge $maxVersionExclusive) {
        throw "Python version must be >= 3.12.0 and < 3.13.0 to create .venv. Found $currentVersion"
    }

    Invoke-CheckedCommand "python" @("-m", "venv", ".venv")
    Write-Information "Created virtual environment."
}
else {
    Write-Information "Virtual environment already exists."
}

$VenvPy = "$PSScriptRoot\.venv\Scripts\python.exe"

if (-not (Test-Path $VenvPy)) {
    throw "Virtual environment interpreter not found at $VenvPy"
}

$venvResolvedVersion = & $VenvPy -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to resolve .venv Python interpreter version at $VenvPy"
}

[version]$minVersion = "3.12.0"
[version]$maxVersionExclusive = "3.13.0"
[version]$venvVersion = $venvResolvedVersion
if ($venvVersion -lt $minVersion) {
    throw ".venv Python version is too old. Required >= 3.12.0 and < 3.13.0, found $venvVersion at $VenvPy"
}
if ($venvVersion -ge $maxVersionExclusive) {
    throw ".venv Python version is incompatible. Required >= 3.12.0 and < 3.13.0, found $venvVersion at $VenvPy"
}

# 3. Check for FFmpeg (Local Install)
Write-Information "`nStep 3: Setting up Local FFmpeg (Full Build)..."
$ffmpegDir = "$PSScriptRoot\.venv\ffmpeg"
$ffmpegBin = "$ffmpegDir\bin\ffmpeg.exe"

if (-not (Test-Path $ffmpegBin)) {
    try {
        # Download FFmpeg (ZIP)
        $ffmpegUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
        $ffmpegZip = "$PSScriptRoot\ffmpeg.zip"
        
        Write-Information "Downloading FFmpeg (Master Latest Win64 GPL ZIP)..."
        Invoke-WebRequest -Uri $ffmpegUrl -OutFile $ffmpegZip -UserAgent "NativeHost"
        
        Write-Information "Extracting FFmpeg..."
        # Extract to venv root temporarily; it creates a subfolder like 'ffmpeg-master-latest-win64-gpl'
        Expand-Archive -Path $ffmpegZip -DestinationPath "$PSScriptRoot\.venv" -Force
        
        # Rename the extracted folder to 'ffmpeg'
        $extractedDir = Get-ChildItem -Path "$PSScriptRoot\.venv" -Directory -Filter "ffmpeg-*" | Select-Object -First 1
        if ($extractedDir) {
            # If 'ffmpeg' folder already exists (e.g. from failed run), remove it first
            if (Test-Path $ffmpegDir) { Remove-Item $ffmpegDir -Recurse -Force }
            Rename-Item -Path $extractedDir.FullName -NewName "ffmpeg"
        }
        
        Write-Information "FFmpeg installed locally in venv."
    }
    catch {
        Write-Error "Failed to download or install FFmpeg: $_"
        exit
    }
    finally {
        # Cleanup ZIP
        if (Test-Path $ffmpegZip) { Remove-Item $ffmpegZip -Force }
    }
}
else {
    Write-Information "Local FFmpeg already exists."
}

# 4. Install Dependencies
Write-Information "`nStep 4: Installing Dependencies via Poetry..."

try {
    $setupPhase = "Upgrading pip"
    Invoke-CheckedCommand $VenvPy @("-m", "pip", "install", "--upgrade", "pip")

    $setupPhase = "Installing Poetry"
    Write-Information "Installing Poetry in the virtual environment..."
    Invoke-CheckedCommand $VenvPy @("-m", "pip", "install", "poetry")

    $setupPhase = "Configuring Poetry"
    Write-Information "Configuring Poetry and installing runtime dependencies..."
    Invoke-CheckedCommand $VenvPy @("-m", "poetry", "config", "--local", "virtualenvs.in-project", "true")
    Invoke-CheckedCommand $VenvPy @("-m", "poetry", "config", "--local", "virtualenvs.create", "false")

    $lockPath = Join-Path $PSScriptRoot "poetry.lock"
    if (-not (Test-Path $lockPath)) {
        $setupPhase = "Generating poetry.lock"
        Write-Information "poetry.lock not found. Resolving dependencies once to generate lockfile..."
        Invoke-CheckedCommand $VenvPy @("-m", "poetry", "lock", "--no-interaction")
    }
    else {
        Write-Information "Using existing poetry.lock (skip dependency resolve)."
    }

    # Use install instead of sync here to avoid uninstalling Poetry from the same environment mid-command.
    $setupPhase = "Installing runtime dependencies"
    Invoke-CheckedCommand $VenvPy @("-m", "poetry", "install", "--no-root", "--only", "main", "--no-interaction")

    $setupPhase = "CUDA 13 runtime validation"
    Write-Information "Validating CUDA 13 BLAS runtime for Faster-Whisper..."
    $gpuValidationCode = @'
import ctypes
import glob
import os
import sys

base = os.path.join(sys.prefix, "Lib", "site-packages")
candidate_dirs = [
    os.path.join(base, "torch", "lib"),
    os.path.join(base, "nvidia", "cu13", "bin"),
    os.path.join(base, "nvidia", "cublas", "bin"),
    os.path.join(base, "nvidia", "cudnn", "bin"),
]

for path in candidate_dirs:
    if not os.path.isdir(path):
        continue
    os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(path)
        except OSError:
            pass

ctypes.CDLL("cublas64_13.dll")
from faster_whisper import WhisperModel
model = WhisperModel("large-v3", device="cuda", compute_type="float16", num_workers=1)
print("FW_CUDA13_OK")
del model
'@
    $gpuValidationScript = Join-Path $env:TEMP ("validate_fw_cuda13_" + [guid]::NewGuid().ToString("N") + ".py")
    try {
        Set-Content -Path $gpuValidationScript -Value $gpuValidationCode -Encoding UTF8
        Invoke-CheckedCommand $VenvPy @($gpuValidationScript)
    }
    finally {
        if (Test-Path $gpuValidationScript) {
            Remove-Item $gpuValidationScript -Force
        }
    }

    Write-Information "Dependencies installed successfully."
}
catch {
    Write-Error "Failed during setup phase '$setupPhase'. Error details: $_"
    if ($setupPhase -eq "CUDA 13 runtime validation") {
        Write-Error "CUDA 13 validation failed. Ensure the CUDA 13 torch + cublas stack is installed in .venv."
    }
}

# 5. Create Start Batch File
Write-Information "`nStep 5: Updating Launcher..."
$batContent = @"
@echo off
set PATH=%~dp0.venv\ffmpeg\bin;%PATH%
call .venv\Scripts\activate
python auto_subtitle.py %*
pause
"@
Set-Content "start.bat" $batContent

Write-Information "`n=== Installation Complete! ==="
Write-Information "Run 'start.bat' to use the tool."
# Read-Host
