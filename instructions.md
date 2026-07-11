# 🤖 AI Developer Guidelines (Strict Enforcement)

This document serves as the **SINGLE SOURCE OF TRUTH** for AI agents working on this project. Adherence to these rules is mandatory.

## 1. 🏗️ Architecture & Structure
- **Modular Design**: All logic MUST reside in `modules/`. `auto_subtitle.py` is strictly an orchestrator (< 500 lines).
- **Isolated Execution**: Heavy AI translation tasks MUST run in a separate process (`isolated_translator.py`) to allow full VRAM cleanup.
- **No Shared State**: Modules should be stateless where possible. Use `ModelManager` for persistent state.

## 2. 🛡️ Reliability & Stability
- **Strict VRAM Enforcement**: 
  - ❌ NEVER use `device_map="auto"` on GPU.
  - ✅ ALWAYS use `device_map="cuda:0"` (or specific device).
  - **Reason**: "Auto" offloads to Shared System RAM, causing performance degradation and unexpected behavior.
- **Robust Shutdown**:
  - ✅ ALWAYS implement a `SetConsoleCtrlHandler` (via `ctypes`) for Windows to intercept "X" button usage.
  - ✅ Ensure `utils.handle_shutdown` kills all subprocesses (including zombies).
- **Atomic Persistence**:
  - ✅ Write to `.tmp` files first, then `os.rename()` to final filename.
  - ✅ Check for existing *valid* output before processing (Resume capability).

## 3. 🧹 Code Quality (Zero Tolerance)
- **Complexity Limit**: Cyclomatic Complexity MUST be **< 10**.
  - ❌ DO NOT use `# noqa: C901`. Refactor the function instead.
- **Linting**:
  - ✅ Run `ruff format --check .` for formatting verification.
  - ✅ Verify with `ruff check .` (must enforce cyclomatic complexity **< 10** via C90), `flake8 modules auto_subtitle.py --max-complexity=10`, and `pylint modules`.
  - ✅ Use `run_local_pipeline.ps1` as the canonical local quality gate.

## 4. ⚡ High-Performance Standards
- **Memory Management**:
  - ✅ Use `gc.collect()` and `torch.cuda.empty_cache()` inside any heavy loops (e.g., translation batches).
  - ✅ **Explicit Offloading**: Always unload previous models (e.g., `model_mgr.offload_whisper()`) *before* starting a new heavy task.
  - ✅ **Tier-based Caps**: Batch sizes for NLLB must follow dynamic tier limits (32/16/8/4) to prevent VRAM overflow.
  - ✅ **Shared Memory Guard**: Forced device mapping MUST be used to prevent "System RAM Spillover".

## 5. 🎨 UI & UX Standards
- **Visual Identity**: Maintain the "High-Tech" aesthetic.
- **Startup Banner**: `utils.print_banner()` must display:
  - Real-time Hardware Stats (CPU/GPU/VRAM).
  - Auto-Tuned Profile (ULTRA/HIGH/MID).
  - Initialization status bar.

---

## 📚 Detailed Documentation Index

- [Project Overview & Directory Structure](docs/project_overview.md)
- [Key Logic & Pipeline](docs/pipeline_logic.md)
- [Hardware Optimization](docs/hardware_optimization.md) (Detailed VRAM/Scaling logic)
- [Development & Standards](docs/development_standards.md) (Detailed Linting/Testing rules)
- [Configuration](docs/configuration.md)
