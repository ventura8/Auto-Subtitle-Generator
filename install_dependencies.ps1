# Sets up the environment for auto_subtitle.py
# Optimization: RTX 5090 / CUDA 12.8 Stable
$ErrorActionPreference = "Stop"

Set-Location -Path $PSScriptRoot

Write-Host "=== Setting up Auto-Subtitle Generator Environment (RTX 5090 Ready) ===" -ForegroundColor Cyan

function Invoke-CheckedCommand {
    param(
        [string]$Executable,
        [string[]]$Arguments
    )

    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $Executable $($Arguments -join ' ')"
    }
}

# 1. Check for Python
try {
    $pyVersion = python --version 2>&1
    Write-Host "Found Python: $pyVersion" -ForegroundColor Green
}
catch {
    Write-Warning "Python not found in PATH."
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "Attempting to install Python 3.12..." -ForegroundColor Cyan
        try {
            winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
            Write-Host "`n[!] Python installed. Please restart script." -ForegroundColor Yellow
            exit
        }
        catch { Write-Error "Winget failed to install Python." }
    }
    else { Write-Error "Python not found. Please install Python 3.12 manually." }
}

# 2. Create Virtual Environment
Write-Host "`nStep 2: Setting up Python Virtual Environment..." -ForegroundColor Yellow
if (-not (Test-Path "$PSScriptRoot\.venv\Scripts\python.exe")) {
    Write-Host "Creating virtual environment..."
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
    Write-Host "Created virtual environment." -ForegroundColor Green
}
else {
    Write-Host "Virtual environment already exists." -ForegroundColor Green
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
Write-Host "`nStep 3: Setting up Local FFmpeg (Full Build)..." -ForegroundColor Yellow
$ffmpegDir = "$PSScriptRoot\.venv\ffmpeg"
$ffmpegBin = "$ffmpegDir\bin\ffmpeg.exe"

if (-not (Test-Path $ffmpegBin)) {
    try {
        # Download FFmpeg (ZIP)
        $ffmpegUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
        $ffmpegZip = "$PSScriptRoot\ffmpeg.zip"
        
        Write-Host "Downloading FFmpeg (Master Latest Win64 GPL ZIP)..." -ForegroundColor Cyan
        Invoke-WebRequest -Uri $ffmpegUrl -OutFile $ffmpegZip -UserAgent "NativeHost"
        
        Write-Host "Extracting FFmpeg..." -ForegroundColor Cyan
        # Extract to venv root temporarily; it creates a subfolder like 'ffmpeg-master-latest-win64-gpl'
        Expand-Archive -Path $ffmpegZip -DestinationPath "$PSScriptRoot\.venv" -Force
        
        # Rename the extracted folder to 'ffmpeg'
        $extractedDir = Get-ChildItem -Path "$PSScriptRoot\.venv" -Directory -Filter "ffmpeg-*" | Select-Object -First 1
        if ($extractedDir) {
            # If 'ffmpeg' folder already exists (e.g. from failed run), remove it first
            if (Test-Path $ffmpegDir) { Remove-Item $ffmpegDir -Recurse -Force }
            Rename-Item -Path $extractedDir.FullName -NewName "ffmpeg"
        }
        
        Write-Host "FFmpeg installed locally in venv." -ForegroundColor Green
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
    Write-Host "Local FFmpeg already exists." -ForegroundColor Green
}

# 4. Install Dependencies
Write-Host "`nStep 4: Installing Dependencies via Poetry..." -ForegroundColor Yellow

try {
    Invoke-CheckedCommand $VenvPy @("-m", "pip", "install", "--upgrade", "pip")

    Write-Host "Installing Poetry in the virtual environment..." -ForegroundColor Cyan
    Invoke-CheckedCommand $VenvPy @("-m", "pip", "install", "poetry")

    Write-Host "Configuring Poetry and installing runtime dependencies..." -ForegroundColor Cyan
    Invoke-CheckedCommand $VenvPy @("-m", "poetry", "config", "--local", "virtualenvs.in-project", "true")
    Invoke-CheckedCommand $VenvPy @("-m", "poetry", "config", "--local", "virtualenvs.create", "false")
    Invoke-CheckedCommand $VenvPy @("-m", "poetry", "-v", "install", "--only", "main")

    Write-Host "Dependencies installed successfully." -ForegroundColor Green
}
catch {
    Write-Error "Failed to install dependencies. Error details: $_"
}

# 5. Create Start Batch File
Write-Host "`nStep 5: Updating Launcher..." -ForegroundColor Yellow
$batContent = @"
@echo off
set PATH=%~dp0.venv\ffmpeg\bin;%PATH%
call .venv\Scripts\activate
python auto_subtitle.py %*
pause
"@
Set-Content "start.bat" $batContent

Write-Host "`n=== Installation Complete! ===" -ForegroundColor Green
Write-Host "Run 'start.bat' to use the tool."
# Read-Host
