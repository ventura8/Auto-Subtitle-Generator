______________________________________________________________________

## name: test-runner description: Execute unit tests, orchestration tests, and enforce >=90% total and per-file code coverage.

# Test Runner Skill

Use this skill to execute unit tests, orchestration tests, and measure line/branch coverage across all modules.

## Dependency & Mocking Philosophy

- **Never mock owned code**: Do not mock internal classes, helper functions, or modules from `auto_subtitle.py` or `modules/` in unit tests unless testing explicit isolation boundaries.
- **Mock external boundaries only**:
  - FFmpeg execution / subprocess calls (`subprocess.Popen`, `subprocess.run`).
  - PyTorch GPU device calls / CUDA VRAM queries (`torch.cuda.is_available()`, `torch.cuda.get_device_properties()`).
  - External ML model instantiation (`faster_whisper.WhisperModel`, `transformers.MarianMTModel`, `transformers.CanaryForConditionalGeneration`, `transformers.ParakeetForTDT`, `audio_separator`).
- **Heavy libraries**: `tests/conftest.py` mocks `torch`, `transformers` and `faster_whisper` in-process; product code loads them only through `importlib` / `load_optional_torch()`. Real-library checks run in a fresh interpreter (`tests/e2e/test_real_nvidia_asr.py`; `ASG_E2E_NVIDIA_ASR=1` adds the real checkpoints).
  - Audio extraction / heavy filesystem media files.
- **Cross-Platform Mocks**: When mocking Windows-specific APIs or DLL directory additions (`os.add_dll_directory`, `ctypes.windll`), always use `mock.patch(..., create=True)` so tests run reliably on both Windows and Linux CI.

## Commands

### 1. Targeted Unit Tests

```powershell
# Run a specific test file
poetry run pytest tests/modules/test_models.py -v

# Run targeted orchestration tests
poetry run pytest tests/orchestration/test_orchestrator.py -v
```

### 2. Full Suite with Coverage Gate

```powershell
poetry run pytest `
  -o addopts= `
  --cov=auto_subtitle `
  --cov=modules `
  --cov-branch `
  --cov-report=xml `
  --cov-report=json `
  --cov-report=term `
  --cov-fail-under=90 `
  tests/
```

### 3. Per-File Coverage Verification (>= 90%)

Every key module must independently achieve $\\ge 90%$ coverage:

```powershell
$coverageFiles = @(
    "auto_subtitle.py",
    "modules/configuration/config.py",
    "modules/configuration/version.py",
    "modules/pipeline/isolated_translator.py",
    "modules/models.py",
    "modules/pipeline/transcription.py",
    "modules/pipeline/translation.py",
    "modules/utils.py",
    "modules/workdir.py",
    "modules/asr/audio_source.py",
    "modules/asr/canary.py",
    "modules/asr/common.py",
    "modules/asr/cues.py",
    "modules/asr/language_id.py",
    "modules/asr/languages.py",
    "modules/asr/parakeet.py",
    "modules/asr/pipeline.py",
    "modules/asr/routing.py",
    "modules/asr/tdt_guard.py",
    "modules/configuration/asr_settings.py"
)

foreach ($file in $coverageFiles) {
    poetry run coverage report --include=$file --fail-under=90 -m
}
```

### 4. Badge & Summary Generation

```powershell
poetry run genbadge coverage -i coverage.xml -o assets/coverage.svg
poetry run python tests/tools/transform_metrics.py coverage.xml
```

## Adding New Tests

When adding or modifying product code:

1. Mirror module structure under `tests/modules/` or `tests/orchestration/`.
1. Name test classes after behavior under test (e.g. `TestTranscriptionPipeline`, `TestVramOptimizerAllocation`).
1. Include positive path, negative path (corrupt media, missing model, invalid language), and edge condition coverage.
1. Ensure no test suppresses warnings or introduces test-specific `# type: ignore` / `# noqa`.
1. Give every test file a unique basename: there is no `tests/__init__.py`, so a duplicate name (e.g. a second `test_common.py`) fails collection.
1. Use `unittest` `self.assert*` methods; bare `assert` statements count toward Radon CC, which must stay grade A (CC 1-5).
1. Tests that touch ASR state add `self.addCleanup(asr_settings.reset)`.
