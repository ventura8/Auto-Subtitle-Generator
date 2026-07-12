# Development & Standards

## 🧩 Modular Architecture

The project is refactored for maintainability and scalability. Core components
are extracted into the `modules/` package:

- **`auto_subtitle.py`**: Minimal orchestrator handling CLI arguments, loops,
  and high-level staging.
- **`modules/config.py`**: Centralized constants, NLLB mappings, and YAML
  loading.
- **`modules/models.py`**: Hardware-aware AI model management (`ModelManager`,
  `SystemOptimizer`).
- **`modules/utils.py`**: Reusable utility functions for IO, FFmpeg, and
  logging.

## Development Workflow

### Installation

- Users run `install_dependencies.ps1` (PowerShell).
- It installs **PyTorch Stable** (for CUDA 12.8 support) and `faster-whisper`.

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
   - **Zero Suppressions**: Do **NOT** use `# noqa: C901`. If a function is too
     complex, refactor it into helper functions.
   - **Formatting**: Use `ruff format` for repository formatting consistency.
   - **Markdown Quality**: Run `mdformat` as the automatic de-linter and
     `pymarkdown scan` as the Markdown linter.
   - **CI Pipeline**: `run_local_pipeline.ps1` is the local quality gate.
     `.github/workflows/ci.yml` runs equivalent Markdown, Ruff, Flake8, Pylint,
     and pytest coverage checks directly in GitHub Actions rather than invoking
     the PowerShell script.
   - **CI Security Defaults**: Workflow permissions default to read-only
     repository contents and checkout steps disable persisted credentials.
   - **AI Workspace**: Agents should follow `.github/skills/fix-file/SKILL.md`
     when applying targeted file fixes.
1. **Documentation**:
   - Always update `docs/instructions.md`, `README.md`, and relevant `docs/`
     files if necessary when making changes.
1. **Run Summaries**:
   - Keep per-file summary output accurate in docs.
   - For multi-file processing, document both aggregate batch summary fields and
     per-file batch stats.
