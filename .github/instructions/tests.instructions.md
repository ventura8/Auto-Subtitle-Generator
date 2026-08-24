______________________________________________________________________

## applyTo: "tests/\*\*/\*.py" description: Use when creating or updating tests for unit, coverage, and pipeline behavior in this project.

# Test Instructions

## Coverage and Scope

- Cover behavioral changes with focused tests in `tests/`.
- Maintain overall and per-file test coverage $\\ge 90%$.
- Keep tests deterministic, fast, and isolated from live GPU/external network dependencies.
- Prefer mocking external boundaries (FFmpeg, CUDA device queries, HuggingFace downloads).
- Never mock internal owned modules.

## Platform Compatibility

- Keep tests compatible with Windows and Linux.
- When mocking platform-specific attributes (`os.add_dll_directory`, `ctypes.windll`), always use `mock.patch(..., create=True)`.

## Assertions & Quality

- Assert user-visible behavior and outputs, not fragile internal implementation details.
- Add negative-path tests for failures in FFmpeg, process crashes, and invalid inputs.
- Never add `# noqa` or `# type: ignore` to test code.

## Validation Commands

```powershell
poetry run pytest tests/
.\run_local_pipeline.ps1
```
