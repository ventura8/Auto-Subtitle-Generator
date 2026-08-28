# Project Agent Rules & Development Guidelines

## Project Overview

`Auto-Subtitle-Generator` is a local-first, GPU-accelerated video/audio subtitle
generation and translation pipeline. It leverages `faster-whisper`
(CTranslate2) for speech recognition, Hugging Face `transformers`
(`NLLBTranslator` default, `TranslateGemmaTranslator` optional) in isolated
child processes for translation, and `audio-separator` for vocal isolation.

- **Primary Target OS**: Windows 11 / Windows 10 (PowerShell first,
  cross-platform compatible).
- **Python Version**: Python `3.12.x` (managed via Poetry).
- **Core Orchestrator**: `auto_subtitle.py`.
- **Modular Subpackages**: `modules/` (`configuration/`, `media/`, `pipeline/`,
  `runtime/`, `subtitles/`).

______________________________________________________________________

## Strict Quality & Policy Invariants

### 1. Zero Suppressions Allowed (Mandatory)

- **NEVER** introduce suppression directives: `# noqa`, `# type: ignore`,
  `# pylint: disable`, `# bandit: disable`, or warning-ignore filters.
- `tests/tools/check_no_suppressions.py` is enforced on every pipeline run
  across all code, tests, and configurations.
- Fix underlying type signatures, logic branches, and lint issues at their
  root cause.

### 2. Cyclomatic Complexity Limit (< 10)

- Every function and method across `auto_subtitle.py`, `modules/`, and
  `tests/` must maintain Cyclomatic Complexity of **A-rank (< 10)**.
- Monolithic functions must be decomposed into small, testable,
  single-responsibility helper functions.

### 3. Strict Code Coverage Thresholds (>= 90%)

- Overall pipeline code coverage must be $\\ge 90%$.
- Every key module must independently achieve $\\ge 90%$ line and branch
  coverage:
  - `auto_subtitle.py`
  - `modules/configuration/config.py`
  - `modules/pipeline/isolated_translator.py`
  - `modules/models.py`
  - `modules/pipeline/transcription.py`
  - `modules/pipeline/translation.py`
  - `modules/utils.py`

### 4. Mandatory Documentation Synchronization (Strict)

- **ALWAYS** update all relevant `.md` documentation files (`AGENTS.md`, `README.md`,
  `docs/`, `.github/instructions/`, and `.agents/skills/`) whenever code,
  architecture, model behaviors, flags, or workflows are modified.
- Never complete a task or change without synchronizing the corresponding markdown
  docs to prevent documentation drift.
- Ensure all updated markdown files pass `mdformat` auto-formatting and
  `pymarkdown scan` checks.

### 5. Canonical Local Quality Gate

- `.\run_local_pipeline.ps1` is the canonical gate. It executes:
  1. Suppression scanner (`check_no_suppressions.py`).
  1. Markdown auto-formatting & scan (`mdformat` + `pymarkdown`).
  1. Python linting (`ruff format --check`, `ruff check`, `flake8`, `pylint`).
  1. Type checking (`mypy`, `pyright`).
  1. Security scans (`bandit`, `pip-audit`).
  1. Radon maintainability metrics (`radon cc`, `radon mi`, `radon hal`).
  1. Test suite execution with coverage (`pytest --cov`).
  1. Per-file $\\ge 90%$ coverage verification.
  1. Badge and metric generation (`genbadge`, `transform_metrics.py`).

### 6. Prefer Installed Dependencies Over Built Ones (Mandatory)

- Always resolve an external binary or library from the **system/environment
  installation first**. A bundled, vendored, or locally built copy is only ever
  a **fallback** when nothing is installed.
- Rationale: installed packages receive OS security updates, match the host's
  architecture and codec/driver set, and avoid shipping a stale duplicate that
  silently diverges from what the installer verified.
- Discovery order for any external tool is therefore:
  1. `shutil.which(...)` / `PATH` lookup (installed).
  1. Bundled or venv-local copies (built).
  1. A bare command name as a last-resort fallback.
- Installer scripts and runtime discovery **must agree** on this order.
  `install_dependencies.sh` probes the system FFmpeg before the venv copy, so
  `modules/media/ffmpeg_utils.get_ffmpeg_paths()` must do the same.
- Do not add a build/vendor step for a dependency that can be installed via the
  platform package manager or an existing wheel.

______________________________________________________________________

## Architectural Contracts

1. **Orchestration vs Modules**:
   - `auto_subtitle.py` handles CLI parsing, batch loops, high-level staging,
     summary logging, and exit codes.
   - Reusable business logic lives under `modules/`.
1. **GPU Memory & Optimizer Profiles**:
   - **Never** use `device_map="auto"`. Use explicit CUDA device indices and
     compute types.
   - Memory profiles (`ULTRA`, `HIGH`, `MID`, `LOW`, `CPU`) dynamically adapt
     models, batch sizes, and precision to available VRAM.
1. **Subprocess Process Isolation**:
   - Translation runs in `modules/pipeline/isolated_translator.py` to prevent
     CUDA memory fragmentation and guarantee full VRAM reclamation.
1. **Atomic Output & Resumability**:
   - Subtitle outputs (`.srt`, `.vtt`, `.txt`) are written to temporary files
     and atomically renamed.
   - Existing outputs are safely skipped when valid subtitles already exist.
1. **Model Download Integrity & Auto-Recovery**:
   - Every downloaded AI model and tokenizer checkpoint (`audio-separator`,
     `faster-whisper`, `nllb`, `translategemma`) incorporates auto-detection of
     corrupted/truncated downloads (`is_corrupt_model_error`), automated cache
     purging (`modules/runtime/model_cache.py`), and transparent re-download
     recovery before inference.

______________________________________________________________________

## Available Agent Skills

The repository provides modular skills under `.agents/skills/` and `.github/skills/`:

### Primary Agent Skills (`.agents/skills/`)

- **`pipeline-runner`**: [`.agents/skills/pipeline-runner/SKILL.md`](.agents/skills/pipeline-runner/SKILL.md)
  (Execute full local validation pipeline and per-file coverage gates).
- **`code-linter`**: [`.agents/skills/code-linter/SKILL.md`](.agents/skills/code-linter/SKILL.md)
  (Run Ruff, Flake8, Pylint, Mypy, Pyright, Bandit, and Radon without
  suppressions).
- **`test-runner`**: [`.agents/skills/test-runner/SKILL.md`](.agents/skills/test-runner/SKILL.md)
  (Run unit tests, orchestration tests, and verify $\\ge 90%$ code coverage).
- **`fix-file`**: [`.agents/skills/fix-file/SKILL.md`](.agents/skills/fix-file/SKILL.md)
  (Focused single-file repair workflow with minimal safe diffs).
- **`model-optimizer`**: [`.agents/skills/model-optimizer/SKILL.md`](.agents/skills/model-optimizer/SKILL.md)
  (Manage VRAM tiers, Faster Whisper compute types, and isolated translation).
- **`setup-dependencies`**: [`.agents/skills/setup-dependencies/SKILL.md`](.agents/skills/setup-dependencies/SKILL.md)
  (Bootstrap Python 3.12+, Poetry, PyTorch CUDA 13.2, and local FFmpeg).
- **`pr-comment-resolution`**: [`.agents/skills/pr-comment-resolution/SKILL.md`](.agents/skills/pr-comment-resolution/SKILL.md)
  (Resolve PR review comments with gh CLI & MCP).
- **`review-with-coderabbit`**: [`.agents/skills/review-with-coderabbit/SKILL.md`](.agents/skills/review-with-coderabbit/SKILL.md)
  (Run local CodeRabbit CLI reviews or replay stored findings).
- **`release-prep`**: [`.agents/skills/release-prep/SKILL.md`](.agents/skills/release-prep/SKILL.md)
  (Derive release version from branch name, update docs, and sync metadata).
- **`docs-sync`**: [`.agents/skills/docs-sync/SKILL.md`](.agents/skills/docs-sync/SKILL.md)
  (Synchronize documentation, AGENTS.md, and skills after changes).
- **`architecture-review`**: [`.agents/skills/architecture-review/SKILL.md`](.agents/skills/architecture-review/SKILL.md)
  (Review and implement structural architecture and process isolation
  changes).

### GitHub Workflow Skills (`.github/skills/`)

- **`architecture-review`**: [`.github/skills/architecture-review/SKILL.md`](.github/skills/architecture-review/SKILL.md)
  (Review and implement structural architecture and process isolation changes).
- **`docs-sync`**: [`.github/skills/docs-sync/SKILL.md`](.github/skills/docs-sync/SKILL.md)
  (Synchronize documentation, AGENTS.md, and skills after changes).
- **`fix-file`**: [`.github/skills/fix-file/SKILL.md`](.github/skills/fix-file/SKILL.md)
  (Apply a focused, minimal-diff fix to a target file end-to-end).
- **`markdown-quality`**: [`.github/skills/markdown-quality/SKILL.md`](.github/skills/markdown-quality/SKILL.md)
  (Run mdformat auto-delinter and pymarkdown quality scan across all markdown
  documents).
- **`pr-comment-resolution`**: [`.github/skills/pr-comment-resolution/SKILL.md`](.github/skills/pr-comment-resolution/SKILL.md)
  (Resolve PR review comments with gh CLI & MCP).
- **`project-setup-maintenance`**: [`.github/skills/project-setup-maintenance/SKILL.md`](.github/skills/project-setup-maintenance/SKILL.md)
  (Maintain onboarding and setup scripts, dependencies, and environment configs).
- **`release-prep`**: [`.github/skills/release-prep/SKILL.md`](.github/skills/release-prep/SKILL.md)
  (Derive release version from branch name, update docs, and sync metadata).
- **`run-local-pipeline`**: [`.github/skills/run-local-pipeline/SKILL.md`](.github/skills/run-local-pipeline/SKILL.md)
  (Execute local pipeline checks, linting, security scans, unit tests, and code
  coverage gates).
- **`setup-dependencies`**: [`.github/skills/setup-dependencies/SKILL.md`](.github/skills/setup-dependencies/SKILL.md)
  (Bootstrap Python 3.12+, Poetry, PyTorch CUDA 13.2, and local FFmpeg).

______________________________________________________________________

## Instructions & Customizations

- [`.github/copilot-instructions.md`](.github/copilot-instructions.md)
- [`.github/instructions/python.instructions.md`](.github/instructions/python.instructions.md)
- [`.github/instructions/tests.instructions.md`](.github/instructions/tests.instructions.md)
- [`.github/instructions/powershell.instructions.md`](.github/instructions/powershell.instructions.md)
- [`.github/instructions/architecture.instructions.md`](.github/instructions/architecture.instructions.md)
- [`.github/instructions/setup.instructions.md`](.github/instructions/setup.instructions.md)
