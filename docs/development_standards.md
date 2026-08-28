# Development & Standards

## 🧩 Modular Architecture

The project is refactored for maintainability and scalability. Core components
are extracted into the `modules/` package:

- **`auto_subtitle.py`**: Minimal orchestrator handling CLI arguments, loops,
  and high-level staging.
- **`modules/configuration/config.py`**: Centralized constants, language
  mappings, and YAML loading.
- **`modules/models.py`**: Hardware-aware AI model management (`ModelManager`,
  `SystemOptimizer`).
- **`modules/runtime/model_cache.py`**: Centralized model corruption detection and
  cache-purging auto-recovery for all downloaded AI models and tokenizers.
- **`modules/utils.py`** plus focused subpackages in `modules/media/`,
  `modules/pipeline/`, `modules/runtime/`, and `modules/subtitles/`: reusable
  IO, FFmpeg, orchestration helpers, logging, and subtitle persistence.

## Development Workflow

### Installation

- Users run `install_dependencies.ps1` (PowerShell).
- It installs **PyTorch Stable** (with CUDA 13.2 support) and `faster-whisper`;
  compatibility runtimes remain isolated from the CUDA 13 libraries.

### Execution

- **Drag & Drop**: Primary user interaction (handled via `sys.argv`).
- **Command Line**: `python auto_subtitle.py <path_to_video>`.

## Quality Control & Guidelines

1. **Error Handling**: Use the `log()` helper for consistent output.
1. **Testing**:
   - Run the local CI pipeline and update the badge: `./run_local_pipeline.ps1`
   - **Strict Requirement**: Maintain at least **90% test coverage** for the
     entire project.
   - Badge and reports are generated automatically on every test run.
1. **Linting & Code Quality**:
   - **Strict Complexity Limit**: All functions must have a Cyclomatic
     Complexity of **< 10**.
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
