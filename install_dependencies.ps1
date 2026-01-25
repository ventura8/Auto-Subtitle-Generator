# Sets up the environment for auto_subtitle.py
# Optimization: RTX 5090 / CUDA 12.8 Nightly
$ErrorActionPreference = "Stop"

Set-Location -Path $PSScriptRoot

Write-Host "=== Setting up Auto-Subtitle Generator Environment (RTX 5090 Ready) ===" -ForegroundColor Cyan

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
if (-not (Test-Path "$PSScriptRoot\venv\Scripts\python.exe")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
    Write-Host "Created virtual environment." -ForegroundColor Green
}
else {
    Write-Host "Virtual environment already exists." -ForegroundColor Green
}

$VenvPy = "$PSScriptRoot\venv\Scripts\python.exe"
$VenvPip = "$PSScriptRoot\venv\Scripts\pip.exe"

# 3. Check for FFmpeg (Local Install)
Write-Host "`nStep 3: Setting up Local FFmpeg (Full Build)..." -ForegroundColor Yellow
$ffmpegDir = "$PSScriptRoot\venv\ffmpeg"
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
        Expand-Archive -Path $ffmpegZip -DestinationPath "$PSScriptRoot\venv" -Force
        
        # Rename the extracted folder to 'ffmpeg'
        $extractedDir = Get-ChildItem -Path "$PSScriptRoot\venv" -Directory -Filter "ffmpeg-*" | Select-Object -First 1
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
Write-Host "`nStep 4: Installing Dependencies..." -ForegroundColor Yellow

try {
    & $VenvPy -m pip install --upgrade pip

    # Uninstall existing PyTorch to ensure clean GPU install
    Write-Host "Ensuring clean PyTorch installation..." -ForegroundColor Cyan
    try {
        $msg = & $VenvPip uninstall -y torch torchvision torchaudio 2>&1
    }
    catch {
        Write-Host "Clean uninstall skipped (packages likely not installed)." -ForegroundColor DarkGray
    }

    # 1. Install PyTorch ecosystem FIRST (Force CUDA 12.8 Nightly for RTX 5090 Blackwell)
    Write-Host "Installing PyTorch Nightly with CUDA 12.8 Support..." -ForegroundColor Cyan
    & $VenvPip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

    # 2. Install Audio-Separator with GPU extras explicitly
    Write-Host "Installing Audio-Separator (GPU)..." -ForegroundColor Cyan
    & $VenvPip install "audio-separator[gpu]"

    # 3. Install other requirements (excluding torch to prevent downgrade)
    Write-Host "Installing remaining dependencies..." -ForegroundColor Cyan
    
    # Read requirements and filter out torch/audio-separator lines
    $reqContent = Get-Content "$PSScriptRoot\requirements.txt"
    $filteredReq = $reqContent | Where-Object { 
        $_ -notmatch "^torch" -and 
        $_ -notmatch "audio-separator" -and 
        $_ -notmatch "--extra-index-url"
    }
    $filteredReq | Set-Content "$PSScriptRoot\requirements_filtered.temp.txt"
    
    & $VenvPip install -r "$PSScriptRoot\requirements_filtered.temp.txt"
    Remove-Item "$PSScriptRoot\requirements_filtered.temp.txt" -Force

    Write-Host "Dependencies installed successfully." -ForegroundColor Green
}
catch {
    Write-Error "Failed to install dependencies. Error details: $_"
}

# 5. Create Start Batch File
Write-Host "`nStep 5: Updating Launcher..." -ForegroundColor Yellow
$batContent = @"
@echo off
set PATH=%~dp0venv\ffmpeg\bin;%PATH%
call venv\Scripts\activate
python auto_subtitle.py %*
pause
"@
Set-Content "start.bat" $batContent

Write-Host "`n=== Installation Complete! ===" -ForegroundColor Green
Write-Host "Run 'start.bat' to use the tool."
# Read-Host
