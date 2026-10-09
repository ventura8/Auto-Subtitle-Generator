# Sets up the environment for auto_subtitle.py
# Optimization: RTX 5090 / CUDA 13.2 Stable (PyTorch cu132 wheels)
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

    $stdoutPath = [System.IO.Path]::GetTempFileName()
    $stderrPath = [System.IO.Path]::GetTempFileName()

    try {
        $quotedArguments = @(
            $Arguments | ForEach-Object {
                if ($_ -match "\s") {
                    '"' + ($_ -replace '"', '\"') + '"'
                }
                else {
                    $_
                }
            }
        )

        $process = Start-Process `
            -FilePath $Executable `
            -ArgumentList $quotedArguments `
            -NoNewWindow `
            -Wait `
            -PassThru `
            -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError $stderrPath

        foreach ($outputLine in Get-Content -Path $stdoutPath -ErrorAction SilentlyContinue) {
            Write-Information $outputLine
        }

        foreach ($outputLine in Get-Content -Path $stderrPath -ErrorAction SilentlyContinue) {
            Write-Information $outputLine
        }

        if ($process.ExitCode -ne 0) {
            throw "Command failed with exit code $($process.ExitCode): $Executable $($Arguments -join ' ')"
        }
    }
    finally {
        if (Test-Path $stdoutPath) {
            Remove-Item $stdoutPath -Force
        }
        if (Test-Path $stderrPath) {
            Remove-Item $stderrPath -Force
        }
    }
}

function Install-PowerShellLintDependency {
    $requiredVersion = "1.22.0"

    $installedModule = Get-Module -ListAvailable -Name PSScriptAnalyzer |
        Sort-Object Version -Descending |
        Select-Object -First 1

    if ($installedModule -and $installedModule.Version -eq [version]$requiredVersion) {
        Write-Information "PSScriptAnalyzer is already installed."
        return
    }

    Write-Information "Installing PSScriptAnalyzer for PowerShell lint gates..."
    $psGallery = Get-PSRepository -Name PSGallery -ErrorAction SilentlyContinue
    if (-not $psGallery) {
        try {
            Register-PSRepository -Default -ErrorAction Stop
            $psGallery = Get-PSRepository -Name PSGallery -ErrorAction SilentlyContinue
        }
        catch {
            throw "PSGallery is not registered. Register it or enable PowerShellGet defaults before installing PSScriptAnalyzer."
        }
    }

    if (-not $psGallery) {
        throw "PSGallery is unavailable after registration attempt. Cannot install PSScriptAnalyzer."
    }

    if ($psGallery.Name -ne "PSGallery" -or $psGallery.SourceLocation -notlike "https://www.powershellgallery.com/api/v2*") {
        throw "PSGallery repository configuration is unexpected. Refusing to install PSScriptAnalyzer from an unverified source."
    }

    Install-Module PSScriptAnalyzer -Scope CurrentUser -RequiredVersion $requiredVersion -Force -Repository PSGallery
}

function Install-LocalFfmpeg {
    param([string]$VenvRoot, [string]$FfmpegDir)

    # Versioned gyan.dev release build from its GitHub mirror. Versioned tags are kept,
    # unlike rolling autobuilds, which upstream prunes after about two weeks.
    $ffmpegUrl = "https://github.com/GyanD/codexffmpeg/releases/download/9.0.2/ffmpeg-9.0.2-full_build.zip"
    $expectedHash = "759D0A9831C436A0EB331AD36F236C06CB04AAA0005DA46571F0E9D3D9206F6B"
    $ffmpegZip = "$PSScriptRoot\ffmpeg.zip"

    Write-Information "No system FFmpeg found. Installing a local copy into the virtual environment."
    try {
        Write-Information "Downloading FFmpeg (pinned gyan.dev 9.0.2 full build)..."
        Invoke-WebRequest -Uri $ffmpegUrl -OutFile $ffmpegZip -UserAgent "NativeHost"

        Write-Information "Verifying FFmpeg archive SHA256 integrity..."
        $computedHash = (Get-FileHash -Path $ffmpegZip -Algorithm SHA256).Hash
        if ($computedHash -ne $expectedHash) {
            throw "FFmpeg archive integrity verification failed! Expected: $expectedHash, Found: $computedHash"
        }
        Write-Information "FFmpeg archive integrity verified successfully."

        Write-Information "Extracting FFmpeg..."
        # The archive holds one folder, 'ffmpeg-9.0.2-full_build', which becomes .venv\ffmpeg.
        Expand-Archive -Path $ffmpegZip -DestinationPath $VenvRoot -Force
        $extractedDir = Get-ChildItem -Path $VenvRoot -Directory -Filter "ffmpeg-*" | Select-Object -First 1
        if (-not $extractedDir) {
            throw "FFmpeg archive did not contain the expected ffmpeg-* folder."
        }
        # A folder left by a failed earlier run is replaced.
        if (Test-Path $FfmpegDir) { Remove-Item $FfmpegDir -Recurse -Force }
        Rename-Item -Path $extractedDir.FullName -NewName "ffmpeg"
        Write-Information "FFmpeg installed locally in venv."
    }
    catch {
        Write-Error "Failed to download or install FFmpeg: $_"
        exit 1
    }
    finally {
        if (Test-Path $ffmpegZip) { Remove-Item $ffmpegZip -Force }
    }
}

function Install-Cuda12Compatibility {
    param([string]$PythonExecutable)

    Write-Information "Provisioning CUDA 12 compatibility runtime for Faster-Whisper..."

    $compatPackages = @(
        "nvidia-cuda-runtime-cu12",
        "nvidia-cublas-cu12",
        "nvidia-cudnn-cu12"
    )

    try {
        Invoke-CheckedCommand $PythonExecutable (@("-m", "pip", "install", "--upgrade") + $compatPackages)
    }
    catch {
        Write-Warning "CUDA 12 compatibility packages were not fully installed via pip: $_"
        Write-Warning "Attempting local CUDA compatibility shim fallback..."
    }

    $compatScript = @'
import os
import shutil
import sys

base = os.path.join(sys.prefix, "Lib", "site-packages")
candidates = [
    os.path.join(base, "torch", "lib"),
    os.path.join(base, "nvidia", "cu12", "bin"),
    os.path.join(base, "nvidia", "cublas", "bin"),
    os.path.join(base, "nvidia", "cudnn", "bin"),
    os.path.join(base, "nvidia", "cuda_runtime", "bin"),
    os.path.join(base, "nvidia", "cu13", "bin"),
]

for path in candidates:
    if not os.path.isdir(path):
        continue
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(path)
        except OSError:
            pass

def has_cuda12_blas() -> bool:
    for path in candidates:
        if os.path.isfile(os.path.join(path, "cublas64_12.dll")):
            return True
    return False

if has_cuda12_blas():
    print("CUDA12_RUNTIME_OK")
    raise SystemExit(0)

source = None
for path in candidates:
    candidate = os.path.join(path, "cublas64_13.dll")
    if os.path.isfile(candidate):
        source = candidate
        break

if source is None:
    raise RuntimeError("Neither cublas64_12.dll nor cublas64_13.dll was found in known runtime directories")

shim_dir = os.path.join(base, "nvidia", "cuda12_compat", "bin")
os.makedirs(shim_dir, exist_ok=True)
target = os.path.join(shim_dir, "cublas64_12.dll")

if not os.path.isfile(target):
    shutil.copy2(source, target)

print("CUDA12_RUNTIME_SHIMMED")
'@

    $compatScriptPath = Join-Path $env:TEMP ("ensure_cuda12_compat_" + [guid]::NewGuid().ToString("N") + ".py")
    try {
        Set-Content -Path $compatScriptPath -Value $compatScript -Encoding UTF8
        Invoke-CheckedCommand $PythonExecutable @($compatScriptPath)
    }
    finally {
        if (Test-Path $compatScriptPath) {
            Remove-Item $compatScriptPath -Force
        }
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

# 3. FFmpeg: an installed FFmpeg always wins (AGENTS.md rule 6, same order as
# modules/media/ffmpeg_utils.get_ffmpeg_paths); the venv copy is only a fallback.
Write-Information "`nStep 3: Checking FFmpeg..."
$ffmpegDir = "$PSScriptRoot\.venv\ffmpeg"
$ffmpegBin = "$ffmpegDir\bin\ffmpeg.exe"
$systemFfmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
$systemFfprobe = Get-Command ffprobe -ErrorAction SilentlyContinue

if ($systemFfmpeg -and $systemFfprobe) {
    Write-Information "Found system FFmpeg: $($systemFfmpeg.Source)"
}
elseif (Test-Path $ffmpegBin) {
    Write-Information "Local FFmpeg already exists."
}
else {
    Install-LocalFfmpeg -VenvRoot "$PSScriptRoot\.venv" -FfmpegDir $ffmpegDir
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
    Invoke-CheckedCommand $VenvPy @("-m", "poetry", "install", "--no-root", "--with", "ml", "--without", "dev", "--no-interaction")

    $setupPhase = "CUDA 12 compatibility provisioning"
    Install-Cuda12Compatibility -PythonExecutable $VenvPy

    $setupPhase = "Faster-Whisper GPU runtime validation"
    Write-Information "Validating Faster-Whisper GPU runtime..."
    $gpuValidationCode = @'
import os
import sys

base = os.path.join(sys.prefix, "Lib", "site-packages")
candidate_dirs = [
    os.path.join(base, "torch", "lib"),
    os.path.join(base, "nvidia", "cu12", "bin"),
    os.path.join(base, "nvidia", "cu13", "bin"),
    os.path.join(base, "nvidia", "cublas", "bin"),
    os.path.join(base, "nvidia", "cudnn", "bin"),
    os.path.join(base, "nvidia", "cuda_runtime", "bin"),
    os.path.join(base, "nvidia", "cuda12_compat", "bin"),
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

from faster_whisper import WhisperModel
model = WhisperModel("large-v3", device="cuda", compute_type="float16", num_workers=1)
print("FW_GPU_RUNTIME_OK")
del model
'@
    $gpuValidationScript = Join-Path $env:TEMP ("validate_fw_gpu_runtime_" + [guid]::NewGuid().ToString("N") + ".py")
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
    if ($setupPhase -eq "Faster-Whisper GPU runtime validation") {
        Write-Error "Faster-Whisper GPU validation failed. Ensure required CUDA runtime DLLs (for example cublas64_12.dll) are present in .venv and rerun install_dependencies.ps1."
    }
    if ($setupPhase -eq "CUDA 12 compatibility provisioning") {
        Write-Error "CUDA 12 compatibility provisioning failed. The installer attempted both dependency installation and local shim creation but could not provide cublas64_12.dll."
    }
    exit 1
}

# 5. Install PowerShell lint dependencies
Write-Information "`nStep 5: Installing PowerShell lint dependencies..."
Install-PowerShellLintDependency

# 6. Create Start Batch File
Write-Information "`nStep 6: Updating Launcher..."
$batContent = @"
@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo ==================================================================
    echo Auto-Subtitle-Generator: Virtual environment not found.
    echo Starting automated environment and dependency installation...
    echo ==================================================================
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install_dependencies.ps1
    if errorlevel 1 (
        echo ERROR: Environment setup failed. Dependency installation did not complete.
        pause
        exit /b 1
    )
)
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Environment setup failed. Virtual environment not found.
    pause
    exit /b 1
)
set PATH=%~dp0.venv\ffmpeg\bin;%PATH%
call .venv\Scripts\activate.bat
python auto_subtitle.py %*
if errorlevel 1 (
    echo.
    pause
)
"@
Set-Content "start.bat" $batContent

Write-Information "`n=== Installation Complete! ==="
Write-Information "Run 'start.bat' to use the tool."
# Read-Host
