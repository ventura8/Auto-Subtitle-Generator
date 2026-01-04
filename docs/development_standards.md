# Development & Standards

## 🧩 Modular Architecture

The project is refactored for maintainability and scalability. Core components are extracted into the `modules/` package:

- **`auto_subtitle.py`**: Minimal orchestrator handling CLI arguments, loops, and high-level staging.
- **`modules/config.py`**: Centralized constants, NLLB mappings, and YAML loading.
- **`modules/models.py`**: Hardware-aware AI model management (`ModelManager`, `SystemOptimizer`).
- **`modules/utils.py`**: Reusable utility functions for IO, FFmpeg, and logging.

## Development Workflow

### Installation
- Users run `install_dependencies.ps1` (PowerShell).
- It installs **PyTorch Nightly** (for CUDA 12.6+ support) and `faster-whisper`.

### Execution
- **Drag & Drop**: Primary user interaction (handled via `sys.argv`).
- **Command Line**: `python auto_subtitle.py <path_to_video>`.

## Quality Control & Guidelines

1. **Error Handling**: Use the `log()` helper for consistent output.
2. **Testing**: 
    -   Run tests and update badge: `pytest` (or use `./run_tests_with_coverage.ps1`)
    -   **Strict Requirement**: Maintain at least **90% test coverage** for the entire project.
    -   Badge and reports are generated automatically on every test run.
3. **Linting & Code Quality**: 
    -   **Strict Complexity Limit**: All functions must have a Cyclomatic Complexity of **< 10**.
    -   **Zero Suppressions**: Do **NOT** use `# noqa: C901`. If a function is too complex, refactor it into helper functions.
    -   **Formatting**: Always use `autopep8` to fix formatting issues automatically.
    -   **CI Pipeline**: The pipeline runs `flake8` with `--max-complexity=10` and fails on any error.
    -   **AI Workspace**: Agents should follow the [/fix-file](file:///c:/Users/ventu/Projects/Auto-Subtitle-Generator/.agent/workflows/fix-file.md) workflow.
4. **Documentation**: 
    -   Always update `Instructions.md`, `README.md`, and relevant `docs/` files if necessary when making changes.
