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

    $stdoutPath = [System.IO.Path]::GetTempFileName()
    $stderrPath = [System.IO.Path]::GetTempFileName()

    try {
        $psi = [System.Diagnostics.ProcessStartInfo]::new()
        $psi.FileName = $Executable
        $quotedArguments = @(
            $Arguments | ForEach-Object {
                if ($_ -eq "") {
                    '""'
                }
                elseif ($_ -match '[\s"]') {
                    '"' + ($_ -replace '"', '\\"') + '"'
                }
                else {
                    $_
                }
            }
        )
        $psi.Arguments = $quotedArguments -join ' '
        $psi.UseShellExecute = $false
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.CreateNoWindow = $true

        $process = [System.Diagnostics.Process]::new()
        $process.StartInfo = $psi
        $null = $process.Start()

        $stdoutText = $process.StandardOutput.ReadToEnd()
        $stderrText = $process.StandardError.ReadToEnd()
        $process.WaitForExit()
        $exitCode = $process.ExitCode

        Set-Content -Path $stdoutPath -Value $stdoutText -Encoding UTF8
        Set-Content -Path $stderrPath -Value $stderrText -Encoding UTF8

        foreach ($outputLine in Get-Content -Path $stdoutPath -ErrorAction SilentlyContinue) {
            Write-Information $outputLine
        }

        foreach ($outputLine in Get-Content -Path $stderrPath -ErrorAction SilentlyContinue) {
            Write-Information $outputLine
        }

        if ($exitCode -ne 0) {
            throw "Command failed: $Executable $($Arguments -join ' ')"
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

function Add-NodePathsToSessionPath {
    $nodeCandidatePaths = @(
        "$env:ProgramFiles\nodejs",
        "$env:LOCALAPPDATA\Programs\nodejs"
    )

    foreach ($path in $nodeCandidatePaths) {
        if (-not (Test-Path $path)) {
            continue
        }

        $currentPathEntries = @($env:PATH -split ';' | Where-Object { $_ })
        if ($currentPathEntries -contains $path) {
            continue
        }

        $env:PATH = "$path;$env:PATH"
    }
}

function Add-PathEntryIfMissing {
    param([string]$PathEntry)

    if (-not $PathEntry) {
        return
    }

    if (-not (Test-Path $PathEntry)) {
        return
    }

    $currentPathEntries = @($env:PATH -split ';' | Where-Object { $_ })
    if ($currentPathEntries -contains $PathEntry) {
        return
    }

    $env:PATH = "$PathEntry;$env:PATH"
}

function Add-NpmGlobalBinToSessionPath {
    if (-not (Test-CommandAvailable "npm")) {
        return
    }

    $npmPrefix = ((& npm prefix -g 2>$null) | Out-String).Trim()
    if (-not $npmPrefix) {
        return
    }

    Add-PathEntryIfMissing $npmPrefix
}

function Test-NpmGlobalPackageInstalled {
    param([string]$PackageName)

    if (-not (Test-CommandAvailable "npm")) {
        return $false
    }

    $null = & npm list -g $PackageName --depth=0 2>$null
    return ($LASTEXITCODE -eq 0)
}

function Install-NodeToolingIfMissing {
    if (Test-CommandAvailable "npm") {
        Add-NpmGlobalBinToSessionPath
        return $true
    }

    Write-Information "npm not found. Attempting automatic Node.js LTS installation via winget..."

    try {
        Install-WingetPackage "OpenJS.NodeJS.LTS" "Node.js LTS"
    }
    catch {
        Write-Warning "Automatic Node.js LTS install failed: $_"
        return $false
    }

    Add-NodePathsToSessionPath
    Add-NpmGlobalBinToSessionPath
    return (Test-CommandAvailable "npm")
}

function Test-McpCommandAvailable {
    return ((Test-CommandAvailable "mcp") -or (Test-CommandAvailable "mcp-inspector"))
}

function Test-McpPackageInstalledButCommandMissing {
    if (-not (Test-NpmGlobalPackageInstalled "@modelcontextprotocol/inspector")) {
        return $false
    }

    Add-NpmGlobalBinToSessionPath
    return (-not (Test-McpCommandAvailable))
}

function Install-McpInspectorPackage {
    Write-Information "Installing MCP inspector CLI via npm..."
    & npm install -g @modelcontextprotocol/inspector
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Failed to install MCP inspector via npm. You can install it manually later."
        return $false
    }

    return $true
}

function Install-McpCliTooling {
    if (-not (Install-NodeToolingIfMissing)) {
        Write-Warning "npm is still unavailable after automatic setup attempt. Install Node.js LTS manually, then re-run this pipeline."
        return
    }

    Add-NpmGlobalBinToSessionPath

    if (Test-McpCommandAvailable) {
        Write-Information "MCP tooling is already available."
        return
    }

    if (Test-McpPackageInstalledButCommandMissing) {
        Write-Warning "MCP inspector is installed globally, but no command was detected in PATH for this session. Open a new terminal and re-run the pipeline."
        return
    }

    if (-not (Install-McpInspectorPackage)) {
        return
    }

    if (Test-McpCommandAvailable) {
        Write-Information "MCP tooling installed successfully."
    }
    else {
        Write-Warning "MCP package installed but no MCP command was detected in PATH for this session."
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

function Test-DevDependenciesAvailable {
    try {
        Invoke-PoetryCommand @("install", "--only", "main,dev", "--no-root")
        return $true
    }
    catch {
        Write-Warning "Failed to synchronize main/dev dependencies: $_"
        return $false
    }
}

function Invoke-PoetryLockRegenerate {
    Write-Information "Regenerating poetry.lock from pyproject.toml..."
    Invoke-PoetryCommand @("lock", "--regenerate", "--no-interaction")
    Invoke-PoetryCommand @("check")
}

function Test-PoetryLockIsValid {
    Invoke-PoetryCommand @("check")
}

$pipelineFailed = $false
$failureMessage = $null

try {
    Invoke-Step "Install Poetry" {
        Initialize-Poetry
    }

    Invoke-Step "Validate Poetry lockfile" {
        Test-PoetryLockIsValid
    }

    Invoke-Step "Report optional developer PR review tooling (GitHub CLI + MCP CLI)" {
        if (Test-CommandAvailable "gh") {
            $ghVersion = (& gh --version | Select-Object -First 1)
            if ($ghVersion) {
                Write-Information "Detected optional GitHub CLI: $ghVersion"
            }
        }
        else {
            Write-Information "Optional GitHub CLI not found (quality gate continues)."
        }

        if (Test-McpCommandAvailable) {
            Write-Information "Detected optional MCP tooling in PATH."
        }
        else {
            Write-Information "Optional MCP tooling not found (quality gate continues)."
        }
    }

    Invoke-Step "Ensure test dependencies (main + dev, no ml)" {
        if (Test-DevDependenciesAvailable) {
            Write-Information "Main/dev dependencies synchronized successfully."
        }
        else {
            throw "Unable to synchronize main/dev dependencies."
        }
    }

    Invoke-Step "Enforce zero-suppression policy" {
        Invoke-PoetryCommand @("run", "python", "tests/tools/check_no_suppressions.py")
    }

    Invoke-Step "Run PowerShell lint" {
        & "$repoRoot\.github\scripts\Invoke-PowerShellLint.ps1" -ScriptPaths @(
            "$repoRoot\.github\scripts\Invoke-PowerShellLint.ps1",
            "$repoRoot\install_dependencies.ps1",
            "$repoRoot\run_local_pipeline.ps1"
        ) -MaxCyclomaticComplexity 9 -MaxNestingDepth 4
    }

    Invoke-Step "Run Markdown auto-delinter (mdformat)" {
        Invoke-PoetryCommand @("run", "mdformat", "README.md", "AGENTS.md", "docs", ".github")
    }

    Invoke-Step "Run Markdown linter (pymarkdown)" {
        Invoke-PoetryCommand @("run", "pymarkdown", "scan", "README.md", "AGENTS.md", "docs", ".github")
    }

    Invoke-Step "Run isort import order check" {
        Invoke-PoetryCommand @("run", "isort", "--check-only", "--filter-files", "auto_subtitle.py", "modules", "tests")
    }

    Invoke-Step "Run Black format check" {
        Invoke-PoetryCommand @("run", "black", "--check", "auto_subtitle.py", "modules", "tests")
    }

    Invoke-Step "Run Taplo format check" {
        Invoke-PoetryCommand @("run", "taplo", "format", "--check", "pyproject.toml", "poetry.toml")
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

    Invoke-Step "Run Pylint on tests (errors-only)" {
        $env:PYTHONPATH = "."
        Invoke-PoetryCommand @("run", "pylint", "tests", "--errors-only")
    }

    Invoke-Step "Run mypy" {
        Invoke-PoetryCommand @("run", "mypy", "auto_subtitle.py", "modules", "tests")
    }

    Invoke-Step "Run pyright" {
        Invoke-PoetryCommand @("run", "pyright", "auto_subtitle.py", "modules", "tests")
    }

    Invoke-Step "Run Bandit security scan (high severity/high confidence)" {
        Invoke-PoetryCommand @("run", "bandit", "-c", "pyproject.toml", "-q", "-r", "auto_subtitle.py", "modules", "-lll", "-iii")
    }

    Invoke-Step "Run dependency vulnerability scan (pip-audit)" {
        $auditRequirementsPath = [System.IO.Path]::GetTempFileName()
        $lockParserPath = [System.IO.Path]::GetTempFileName()
        $lockParser = @"
import sys
import tomllib
from pathlib import Path

lock_data = tomllib.loads(Path("poetry.lock").read_text(encoding="utf-8"))
packages = lock_data.get("package", [])
selected_by_name = {}
for package in packages:
    groups = set(package.get("groups", []))
    if not groups.intersection({"main", "ml"}):
        continue
    name = package.get("name")
    version = package.get("version")
    if not name or not version:
        continue
    # Keep one pinned version per package name for pip-audit requirements input.
    if name not in selected_by_name:
        selected_by_name[name] = version

lines = [f"{name}=={version}" for name, version in sorted(selected_by_name.items())]
Path(sys.argv[1]).write_text("\n".join(lines) + "\n", encoding="utf-8")
"@
        try {
            Set-Content -Path $lockParserPath -Value $lockParser -Encoding UTF8
            Invoke-CheckedCommand $VenvPy @($lockParserPath, $auditRequirementsPath)
            Invoke-PoetryCommand @("run", "pip-audit", "--requirement", $auditRequirementsPath, "--no-deps", "--disable-pip")
        }
        finally {
            if (Test-Path $lockParserPath) {
                Remove-Item $lockParserPath -Force
            }
            if (Test-Path $auditRequirementsPath) {
                Remove-Item $auditRequirementsPath -Force
            }
        }
    }

    Invoke-Step "Run Radon complexity (A-grade enforced)" {
        $radonArgs = @(
            "-m",
            "poetry",
            "run",
            "radon",
            "cc",
            "auto_subtitle.py",
            "modules",
            "tests/modules",
            "tests/orchestration",
            "-s",
            "-a"
        )

        $radonOutput = (& $VenvPy @radonArgs 2>&1) | Out-String
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed: $VenvPy $($radonArgs -join ' ')"
        }

        Set-Content -Path "$repoRoot\radon_report.txt" -Value $radonOutput -Encoding UTF8
        Write-Information $radonOutput

        $nonAGrades = @(
            ($radonOutput -split "`r?`n") |
                Where-Object { $_ -match "\s-\s[B-F]\s\(" }
        )
        if ($nonAGrades.Count -gt 0) {
            Write-Error "Radon found non-A complexity grades:" -ErrorAction Continue
            foreach ($line in $nonAGrades) {
                Write-Error "  $line" -ErrorAction Continue
            }
            throw "Radon complexity gate failed. All functions/methods must be grade A."
        }
    }

    Invoke-Step "Run Radon maintainability index (A-grade enforced)" {
        $radonMiArgs = @(
            "-m",
            "poetry",
            "run",
            "radon",
            "mi",
            "auto_subtitle.py",
            "modules",
            "tests/modules",
            "tests/orchestration",
            "-s"
        )

        $radonMiOutput = (& $VenvPy @radonMiArgs 2>&1) | Out-String
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed: $VenvPy $($radonMiArgs -join ' ')"
        }

        Set-Content -Path "$repoRoot\radon_mi_report.txt" -Value $radonMiOutput -Encoding UTF8
        Write-Information $radonMiOutput

        $nonAGrades = @(
            ($radonMiOutput -split "`r?`n") |
                ForEach-Object {
                    $line = $_
                    $match = [regex]::Match($line, "\s-\s([A-F])\s\(")
                    if ($match.Success) {
                        $grade = $match.Groups[1].Value
                        if ($grade -ne "A") {
                            $line
                        }
                    }
                }
        )

        if ($nonAGrades.Count -gt 0) {
            Write-Error "Radon MI gate failed. Non-A grades were found:" -ErrorAction Continue
            foreach ($line in $nonAGrades) {
                Write-Error "  $line" -ErrorAction Continue
            }
            throw "Radon MI gate failed. All files must have A-grade maintainability."
        }
    }

    Invoke-Step "Run Radon Halstead metrics (hal)" {
        $radonHalArgs = @(
            "-m",
            "poetry",
            "run",
            "radon",
            "hal",
            "auto_subtitle.py",
            "modules",
            "tests/modules",
            "tests/orchestration"
        )

        $radonHalOutput = (& $VenvPy @radonHalArgs 2>&1) | Out-String
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed: $VenvPy $($radonHalArgs -join ' ')"
        }

        Set-Content -Path "$repoRoot\radon_hal_report.txt" -Value $radonHalOutput -Encoding UTF8
        Write-Information $radonHalOutput
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
            "modules/configuration/config.py",
            "modules/pipeline/isolated_translator.py",
            "modules/models.py",
            "modules/pipeline/transcription.py",
            "modules/pipeline/translation.py",
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

        Invoke-PoetryCommand @("run", "genbadge", "coverage", "-i", "coverage.xml", "-o", "assets/coverage.svg")
        Invoke-PoetryCommand @("run", "python", "tests/tools/transform_metrics.py", "coverage.xml")
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