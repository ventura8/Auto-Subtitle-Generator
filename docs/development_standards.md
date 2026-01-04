# Development & Standards

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
3. **Linting**: 
    -   Ensure all code is PEP 8 compliant. 
    -   Run `flake8 . --count --max-line-length=127 --statistics` before committing.
    -   CI pipeline automatically runs Lint, Test, and Report jobs.
    -   **AI Workspace**: Agents should follow the [/fix-file](file:///c:/Users/ventu/Projects/Auto-Subtitle-Generator/.agent/workflows/fix-file.md) workflow for automated lint and test fixes.
4. **Documentation**: 
    -   Always update `Instructions.md`, `README.md`, and relevant `docs/` files if necessary when making changes.
