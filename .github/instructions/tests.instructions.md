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
- Use the `unittest` style with `self.assert*`: every bare `assert` adds to the Radon cyclomatic complexity, which must stay grade A (CC 1-5).

## Test Layout

- Give every test file a unique basename. There is no `tests/__init__.py`, so a second `test_common.py` anywhere fails collection (hence `test_asr_common.py`, `test_canary_wrapper.py`).
- Load heavy libraries only through `importlib` / `load_optional_torch()`; `tests/conftest.py` mocks `torch`, `transformers` and `faster_whisper` in-process.
- Real-library checks run in a fresh interpreter (`tests/e2e/test_real_nvidia_asr.py`), because the conftest mocks apply in-process. `ASG_E2E_NVIDIA_ASR=1` adds the real Canary/Parakeet checkpoints.
- Tests that change ASR state add `self.addCleanup(asr_settings.reset)` (and clear any `--asr` override they set).

## Validation Commands

```powershell
poetry run pytest tests/
.\run_local_pipeline.ps1
```
