______________________________________________________________________

## description: Apply lint and test fixes for a file in a single pass.

# AI File Fix Workflow

This workflow is designed to resolve coding issues in a specific file by prioritizing linting before testing, ensuring high quality and cross-platform compatibility.

## Steps

### 1. Fix Linting Issues

Use the repository quality stack directly. Do not rely on one-shot auto-fixers that bypass project rules.

- **Priority**: Finish all lint fixes before moving to tests.
- **Goal**: Zero linting errors.

```powershell
poetry run ruff format <path_to_file>
poetry run ruff check <path_to_file>
poetry run flake8 <path_to_file>
poetry run pylint <path_to_file>
```

### 2. Run Tests & Coverage

Once linting is clean, execute relevant tests with coverage.

- **Requirement**: Use `./run_local_pipeline.ps1` for full repository validation.
- **Single Pass**: If tests fail, analyze the failures and apply fixes immediately.

### 3. Cross-Platform Compatibility (Mocks)

When fixing tests or mocks, ensure they are compatible with both **Windows** and **Linux**.

- **Important**: If mocking platform-specific components (like `os.add_dll_directory` or `ctypes.windll`), always use `mock.patch(..., create=True)`.

### 4. Coverage Validation

After tests pass, check the coverage report.

- **Threshold**: Total coverage MUST be at least **90%**.
- **Badge**: Always generate/update the coverage badge after a successful test run.

```powershell
# Full local gate
./run_local_pipeline.ps1
```

### 5. Suppressions and Security

- **No suppressions**: never introduce `# noqa`, `# type: ignore`, or warning-ignore filters.
- **Security**: ensure `bandit -c pyproject.toml -q -r auto_subtitle.py modules -lll -iii` and `pip-audit` pass.

> [!IMPORTANT]
> Always verify that the coverage percentage in the summary is ≥ 90%. If it drops, add missing tests for the file being touched.
