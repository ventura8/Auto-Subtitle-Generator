______________________________________________________________________

## description: Execute local pipeline checks, linting, security scans, unit tests, code coverage gates (>=90%), and metric reports on Windows/PowerShell.

# Run Local Pipeline Workflow

Use this workflow to validate changes locally against all quality gates before committing.

## Steps

### 1. Run Canonical Pipeline

Execute the full PowerShell validation script:

```powershell
.\run_local_pipeline.ps1
```

### 2. Verify Gate Results

Verify each section completed cleanly:

- Zero suppressions check
- Markdown formatting and scan
- Ruff, Flake8, Pylint
- Mypy and Pyright static typing
- Bandit and Pip-audit security scan
- Radon Cyclomatic Complexity (< 10) & Maintainability Index (A-rank)
- Pytest suite with $\\ge 90%$ line and branch coverage
- Per-file coverage $\\ge 90%$ for all core modules
- Badge and metric artifact generation
