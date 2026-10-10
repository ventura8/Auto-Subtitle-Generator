# Development & Standards

## 🧩 Modular Architecture

The project is refactored for maintainability and scalability. Core components
are extracted into the `modules/` package:

- **`auto_subtitle.py`**: Minimal orchestrator handling CLI arguments, loops,
  and high-level staging.
- **`modules/configuration/config.py`**: Centralized constants, language
  mappings, and YAML loading.
- **`modules/models.py`**: Hardware-aware AI model management (`ModelManager`,
  `SystemOptimizer`), including the NVIDIA ASR slot (`get_asr` /
  `offload_asr`).
- **`modules/asr/`** and **`modules/configuration/asr_settings.py`**: NVIDIA
  Canary/Parakeet engines (streaming audio, language vote, routing, decoding,
  filters, cues) and the `asr:` config section / `--asr` override.
- **`modules/runtime/model_cache.py`**: Centralized model corruption detection and
  cache-purging auto-recovery for all downloaded AI models and tokenizers.
- **`modules/utils.py`** plus focused subpackages in `modules/media/`,
  `modules/pipeline/`, `modules/runtime/`, and `modules/subtitles/`: reusable
  IO, FFmpeg, orchestration helpers, logging, and subtitle persistence.

## Development Workflow

### Installation

- **Windows**: Users run `install_dependencies.ps1` (PowerShell).
- **Linux / macOS**: Users run `install_dependencies.sh` (Bash).
- Both scripts install **PyTorch Stable** (with CUDA 13.2 support on Linux/Windows)
  and `faster-whisper`; compatibility runtimes remain isolated from the CUDA 13
  libraries.

### Execution

- **Drag & Drop**: Primary user interaction (handled via `sys.argv`).
- **Command Line**: `python auto_subtitle.py <path_to_video>`.

## Quality Control & Guidelines

1. **Error Handling**: Use the `log()` helper for consistent output.
1. **Testing**:
   - Run the local CI pipeline and update the badge: `./run_local_pipeline.ps1`
   - **Strict Requirement**: Maintain at least **90% test coverage** for the
     entire project and for every file in the per-file list (including each
     `modules/asr/*.py` and `modules/configuration/asr_settings.py`).
   - Real-library checks of the NVIDIA engines live in
     `tests/e2e/test_real_nvidia_asr.py` (subprocesses, because `conftest.py`
     mocks torch/transformers in-process); `ASG_E2E_NVIDIA_ASR=1` adds the
     real checkpoints. CI runs them in the `e2e_nvidia_asr` job.
   - ASR benchmark: `python -m tests.tools.asr_benchmark` (optional `bench`
     Poetry group for VoxPopuli); it writes only to `--out` and the Hugging
     Face cache.
   - Badge and reports are generated automatically on every test run.
1. **Linting & Code Quality**:
   - **Strict Complexity Limit**: All functions must be Radon **grade A
     (CC 1-5)**; the gate fails on any B-F line. Ruff's mccabe
     (`max-complexity = 9`) and flake8 `--max-complexity=9` are only
     backstops. Every file must also keep maintainability index grade A.
   - **Heavy Libraries**: `torch`, `transformers`, `faster_whisper`, `numpy`,
     `soundfile` and `huggingface_hub` are imported only through `importlib`
     or `load_optional_torch()`, never at module top level, so the gate runs
     without the `ml` group.
   - **Zero Suppressions**: Do **NOT** use `# noqa`, `# type: ignore`, warning
     ignore filters, or linter/type checker ignore knobs. If code fails checks,
     fix the root cause.
   - **Suppression Scanner**: `tests/tools/check_no_suppressions.py` must pass
     in local and CI gates.
   - **Formatting**: Use `ruff format` for repository formatting consistency.
   - **Markdown Quality**: Run `mdformat` as the automatic de-linter and
     `pymarkdown scan` as the Markdown linter.
   - **Security**: Run `bandit -lll -iii` and `pip-audit`
     inside the quality gate.
   - **CI Pipeline**: `run_local_pipeline.ps1` is the local quality gate.
     `.github/workflows/ci.yml` runs equivalent Markdown, Ruff, Flake8, Pylint,
     security scans, and pytest coverage checks directly in GitHub Actions
     rather than invoking the PowerShell script.
   - **CI Security Defaults**: Workflow permissions default to read-only
     repository contents and checkout steps disable persisted credentials.
     Poetry installs wheels only (`POETRY_INSTALLER_ONLY_BINARY=":all:"`), so
     no dependency's setup script runs on a runner, with one exception: `diffq`,
     which publishes no wheel, builds from its pinned sdist
     (`POETRY_INSTALLER_NO_BINARY`) in the Linux ML job, and its build code
     does run there. Add a package to that exception only when it has no wheel
     for the runner.
   - **Docker Build Context**: `docker/Dockerfile.ubuntu` copies an explicit
     list of files and directories, never `COPY . /app`, so local secrets,
     caches and media can't leak into the image. Extend the list when the
     installer or the E2E suite needs a new top-level path.
   - **AI Workspace**: Agents should follow `.github/skills/fix-file/SKILL.md`
     when applying targeted file fixes.
1. **Documentation Synchronization**:
   - **Mandatory**: Every time you perform work on the project, you must update
     all relevant `.md` files (`AGENTS.md`, `README.md`, `docs/`,
     `.github/instructions/`, and `.agents/skills/`).
   - Prevent documentation drift across releases, model lifecycle changes, and
     pipeline components.
1. **Run Summaries**:
   - Keep per-file summary output accurate in docs.
   - For multi-file processing, document both aggregate batch summary fields and
     per-file batch stats.
