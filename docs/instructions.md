# 🤖 AI Developer Guidelines (Strict Enforcement)

This document serves as the **SINGLE SOURCE OF TRUTH** for AI agents working on
this project. Adherence to these rules is mandatory.

## 1. 🏗️ Architecture & Structure

- **Modular Design**: All logic MUST reside in `modules/`. `auto_subtitle.py` is
  strictly an orchestrator (< 500 lines).
- **Isolated Execution**: Heavy AI translation tasks MUST run in a separate
  process (`isolated_translator.py`) to allow full VRAM cleanup.
- **No Shared State**: Modules should be stateless where possible. Use
  `ModelManager` for persistent state.

## 2. 🛡️ Reliability & Stability

- **Strict VRAM Enforcement**:
  - ❌ NEVER use `device_map="auto"` on GPU.
  - ✅ ALWAYS use `device_map="cuda:0"` (or specific device).
  - **Reason**: "Auto" offloads to Shared System RAM, causing performance
    degradation and unexpected behavior.
- **Robust Shutdown**:
  - ✅ ALWAYS implement a `SetConsoleCtrlHandler` (via `ctypes`) for Windows to
    intercept "X" button usage.
  - ✅ Ensure `utils.handle_shutdown` kills all subprocesses (including zombies).
- **Atomic Persistence**:
  - ✅ Write to `.tmp` files first, then `os.rename()` to final filename.
  - ✅ Check for existing *valid* output before processing (Resume capability).
- **Corrupt Download Auto-Recovery**:
  - ✅ Auto-detect corrupted/truncated model checkpoints or tokenizers via
    `is_corrupt_model_error`.
  - ✅ Automatically purge corrupted cache snapshots (`model_cache.py`) and
    re-download cleanly.

## 3. 🧹 Code Quality (Zero Tolerance)

- **Complexity Limit**: Cyclomatic Complexity MUST be **< 10**.
  - ❌ DO NOT use suppression patterns (`# noqa`, `# type: ignore`, warning
    filter ignores, or ignore-based config knobs). Refactor or type/fix code
    instead.
- **Mandatory Documentation Synchronization**:
  - ✅ **EVERY** time you do work on the project, you MUST update all relevant
    `.md` files (`AGENTS.md`, `README.md`, `docs/`, `.github/instructions/`,
    `.agents/skills/`).
- **Linting**:
  - ✅ Enforce suppression policy with
    `python tests/tools/check_no_suppressions.py`.
  - ✅ Run `ruff format --check .` for formatting verification.
  - ✅ Verify with `ruff check .` (must enforce cyclomatic complexity **< 10**
    via C90 with max complexity 9),
    `flake8 modules auto_subtitle.py --max-complexity=9`,
    `pylint modules`, `pylint tests --errors-only`, and
    the repository test suite.
  - ✅ Run security checks with `bandit -q -r auto_subtitle.py modules -lll -iii`
    and `pip-audit`.
  - ✅ Use `run_local_pipeline.ps1` as the canonical local quality gate.

## 4. ⚡ High-Performance Standards

- **Memory Management**:
  - ✅ Use `gc.collect()` and `torch.cuda.empty_cache()` inside any heavy loops
    (e.g., translation batches).
  - ✅ **Explicit Offloading**: Always unload previous models (e.g.,
    `model_mgr.offload_whisper()`) *before* starting a new heavy task.
  - ✅ **Tier-based Caps**: Batch sizes for NLLB must follow dynamic tier limits
    (32/16/8/4) to prevent VRAM overflow.
  - ✅ **Shared Memory Guard**: Forced device mapping MUST be used to prevent
    "System RAM Spillover".

## 5. 🎨 UI & UX Standards

- **Visual Identity**: Maintain the "High-Tech" aesthetic.
- **Startup Banner**: `utils.print_banner()` must display:
  - Real-time Hardware Stats (CPU/GPU/VRAM).
  - Auto-Tuned Profile (ULTRA/HIGH/MID).
  - Initialization status bar.

______________________________________________________________________

## 📚 Detailed Documentation Index

- [Project Overview & Directory Structure](project_overview.md)
- [Key Logic & Pipeline](pipeline_logic.md)
- [Hardware Optimization](hardware_optimization.md) (Detailed VRAM/Scaling
  logic)
- [Development & Standards](development_standards.md) (Detailed
  Linting/Testing rules)
- [Configuration](configuration.md)
- [Release Notes](releases/v1.2.0.md)
- [Release Prep Skill](../.github/skills/release-prep/SKILL.md)
