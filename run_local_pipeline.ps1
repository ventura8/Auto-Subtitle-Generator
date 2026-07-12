param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$InformationPreference = "Continue"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot
$VenvPy = "$repoRoot\.venv\Scripts\python.exe"

if (-not (Test-Path $VenvPy)) {
    throw "Virtual environment interpreter not found at $VenvPy. Run install_dependencies.ps1 first."
}

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Action
    )

    Write-Information "==> $Name"
    & $Action
}

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

function Test-CommandAvailable {
    param([string]$CommandName)

    return [bool](Get-Command $CommandName -ErrorAction SilentlyContinue)
}

function Install-WingetPackage {
    param(
        [string]$PackageId,
        [string]$DisplayName
    )

    if (-not (Test-CommandAvailable "winget")) {
        throw "winget is required to auto-install $DisplayName. Install it manually and retry."
    }

    Write-Information "Installing $DisplayName via winget..."
    $wingetOutput = (& winget install -e --id $PackageId --accept-package-agreements --accept-source-agreements --silent 2>&1) | Out-String

    if ($LASTEXITCODE -eq 0) {
        return
    }

    # winget may return a non-zero code when package is already present/up-to-date.
    if ($wingetOutput -match "No available upgrade found" -or $wingetOutput -match "No newer package versions are available" -or $wingetOutput -match "Found an existing package already installed") {
        Write-Information "$DisplayName is already installed and up to date."
        return
    }

    throw "winget failed to install $DisplayName ($PackageId). Output: $wingetOutput"
}

function Install-McpCliTooling {
    if (Test-CommandAvailable "mcp") {
        Write-Information "MCP CLI is already available."
        return
    }

    if (-not (Test-CommandAvailable "npm")) {
        Write-Warning "npm not found. Skipping MCP CLI auto-install. Install Node.js LTS to enable MCP CLI setup."
        return
    }

    Write-Information "Installing MCP CLI via npm..."
    & npm install -g @modelcontextprotocol/cli
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Failed to install MCP CLI via npm. You can install it manually later."
        return
    }

    if (Test-CommandAvailable "mcp") {
        Write-Information "MCP CLI installed successfully."
    }
    else {
        Write-Warning "MCP CLI package installed but 'mcp' command was not detected in PATH for this session."
    }
}

function Invoke-PoetryCommand {
    param([string[]]$Arguments)

    Invoke-CheckedCommand $VenvPy (@("-m", "poetry") + $Arguments)
}

function Initialize-Poetry {
    try {
        Invoke-PoetryCommand @("--version")
    }
    catch {
        Write-Information "Poetry was not found. Installing Poetry..."
        Invoke-CheckedCommand $VenvPy @("-m", "pip", "install", "poetry")
        Invoke-PoetryCommand @("--version")
    }
}

function Test-PoetryLockFresh {
    $null = & $VenvPy -m poetry check
    return ($LASTEXITCODE -eq 0)
}

function Invoke-PoetryLockRefresh {
    if (-not (Test-Path "$repoRoot\poetry.lock")) {
        Write-Information "poetry.lock not found. Generating lockfile..."
        Invoke-PoetryCommand @("lock", "--no-interaction")
        return
    }

    if (-not (Test-PoetryLockFresh)) {
        Write-Information "poetry.lock is out of date. Regenerating lockfile..."
        Invoke-PoetryCommand @("lock", "--no-interaction")
        Invoke-PoetryCommand @("check")
    }
}

$pipelineFailed = $false
$failureMessage = $null

try {
    Invoke-Step "Install Poetry" {
        Initialize-Poetry
    }

    Invoke-Step "Install developer PR review tooling (GitHub CLI + MCP CLI)" {
        if (-not (Test-CommandAvailable "gh")) {
            Install-WingetPackage "GitHub.cli" "GitHub CLI"
        }
        else {
            Write-Information "GitHub CLI is already available."
        }

        if (Test-CommandAvailable "gh") {
            $ghVersion = (& gh --version | Select-Object -First 1)
            if ($ghVersion) {
                Write-Information "Detected: $ghVersion"
            }
        }

        Install-McpCliTooling
    }

    Invoke-Step "Ensure Poetry lockfile is fresh" {
        Invoke-PoetryLockRefresh
    }

    Invoke-Step "Install test dependencies (main + dev, no ml)" {
        Invoke-PoetryCommand @("install", "-v", "--only", "main,dev", "--no-root")
    }

    Invoke-Step "Run PowerShell lint" {
        & "$repoRoot\.github\scripts\Invoke-PowerShellLint.ps1" -ScriptPaths @(
            "$repoRoot\install_dependencies.ps1",
            "$repoRoot\run_local_pipeline.ps1"
        )
    }

    Invoke-Step "Run Markdown auto-delinter (mdformat)" {
        Invoke-PoetryCommand @("run", "mdformat", "README.md", "AGENTS.md", "docs", ".github")
    }

    Invoke-Step "Run Markdown linter (pymarkdown)" {
        Invoke-PoetryCommand @("run", "pymarkdown", "scan", "README.md", "AGENTS.md", "docs", ".github")
    }

    Invoke-Step "Run Ruff" {
        Invoke-PoetryCommand @("run", "ruff", "check", "modules", "auto_subtitle.py", "tests")
    }

    Invoke-Step "Run Ruff format check" {
        Invoke-PoetryCommand @("run", "ruff", "format", "--check", ".")
    }

    Invoke-Step "Run Flake8" {
        Invoke-PoetryCommand @("run", "flake8", "modules", "auto_subtitle.py", "tests")
    }

    Invoke-Step "Run Pylint" {
        Invoke-PoetryCommand @("run", "pylint", "modules", "auto_subtitle.py")
    }

    Invoke-Step "Run tests with coverage" {
        Invoke-PoetryCommand @(
            "run",
            "pytest",
            "-o",
            "addopts=",
            "--cov=auto_subtitle",
            "--cov=modules",
            "--cov-branch",
            "--cov-report=xml",
            "--cov-report=json",
            "--cov-report=term",
            "--cov-fail-under=90",
            "tests/"
        )
    }

    Invoke-Step "Enforce per-file coverage >= 90%" {
        $coverageFiles = @(
            "auto_subtitle.py",
            "modules/config.py",
            "modules/isolated_translator.py",
            "modules/models.py",
            "modules/transcription.py",
            "modules/translation.py",
            "modules/utils.py"
        )

        foreach ($coverageFile in $coverageFiles) {
            Write-Information "   -> Checking $coverageFile"
            Invoke-PoetryCommand @(
                "run",
                "coverage",
                "report",
                "--include=$coverageFile",
                "--fail-under=90",
                "-m"
            )
        }
    }

}
catch {
    $pipelineFailed = $true
    $failureMessage = $_.Exception.Message
    Write-Error $failureMessage -ErrorAction Continue
}

try {
    Invoke-Step "Generate coverage badge and summary" {
        if ($pipelineFailed) {
            Write-Information "Skipping badge generation because a previous pipeline step failed."
            return
        }

        if (-not (Test-Path "coverage.xml")) {
            throw "coverage.xml was not generated by the test step."
        }

        Invoke-PoetryCommand @("run", "python", "tests/transform_metrics.py", "coverage.xml", "assets/coverage.svg")
    }
}
catch {
    $pipelineFailed = $true
    $failureMessage = $_.Exception.Message
    Write-Error $failureMessage -ErrorAction Continue
}

if ($pipelineFailed) {
    exit 1
}

Write-Information "Local pipeline completed successfully."