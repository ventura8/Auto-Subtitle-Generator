---
description: Apply lint and test fixes for a file in a single pass.
---

# AI File Fix Workflow

This workflow is designed to resolve coding issues in a specific file by prioritizing linting before testing, ensuring high quality and cross-platform compatibility.

## Steps

### 1. Fix Linting Issues
Always use `autopep8` to resolve simple PEP 8 violations automatically before running checks.
- **Priority**: Finish all lint fixes before moving to tests.
- **Goal**: Zero linting errors.

```powershell
# Turbo fix lints
autopep8 --in-place --recursive .

# Verify remaining issues
flake8 <path_to_file> --count --max-line-length=127 --statistics
```

### 2. Run Tests & Coverage
Once linting is clean, execute relevant tests with coverage.
- **Requirement**: Use `./run_tests_with_coverage.ps1` to ensure coverage is captured.
- **Single Pass**: If tests fail, analyze the failures and apply fixes immediately.

### 3. Cross-Platform Compatibility (Mocks)
When fixing tests or mocks, ensure they are compatible with both **Windows** and **Linux**.
- **Important**: If mocking platform-specific components (like `os.add_dll_directory` or `ctypes.windll`), always use `mock.patch(..., create=True)`.

### 4. Coverage Validation
After tests pass, check the coverage report.
- **Threshold**: Total coverage MUST be at least **90%**.
- **Badge**: Always generate/update the coverage badge after a successful test run.

```powershell
# Run tests and generate report
./run_tests_with_coverage.ps1
```

> [!IMPORTANT]
> Always verify that the coverage percentage in the summary is ≥ 90%. If it drops, add missing tests for the file being touched.
