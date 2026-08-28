______________________________________________________________________

## name: model-optimizer description: Manage VRAM tier profiles, Faster Whisper compute types, isolated translation sub-processes, and atomic subtitle IO.

# Model Optimizer Skill

Use this skill when modifying model lifecycle, hardware auto-tuning, GPU memory management, or isolated worker architecture.

## Core Architectural Invariants

1. **Explicit Device Mapping Only**:

   - **Never** introduce `device_map="auto"` or `device_map="balanced"` for transformers or Faster Whisper.
   - Always map explicitly: `device="cuda"`, `device_index=0`, `compute_type=...` or fallback to `device="cpu"`.

1. **VRAM Tier Hierarchy (`modules/models.py`)**:

   - **`ULTRA` (>= 24GB VRAM)**: Tuned for `nllb_batch=8`, `translategemma_batch=24`, `translategemma_max_new_tokens=192`.
   - **`HIGH` (>= 10GB VRAM)**: Tuned for `nllb_batch=8`, `translategemma_batch=8`, `translategemma_max_new_tokens=192`.
   - **`MID` (< 10GB CUDA VRAM)**: Tuned for `nllb_batch=6`, `translategemma_batch=4`, `translategemma_max_new_tokens=160`.
   - **`CPU_ONLY` (CUDA unavailable / CPU fallback)**: Tuned for `nllb_batch=2`, `translategemma_batch=1`, `translategemma_max_new_tokens=144`.

1. **Subprocess Process Isolation (`isolated_translator.py`)**:

   - To prevent CUDA VRAM fragmentation and Out-Of-Memory (OOM) leaks between Whisper transcription and translation backends, translation runs in a separate child process.
   - The orchestrator completely offloads and frees Whisper VRAM before spinning up translation workers.
   - Translation sub-process must handle clean SIGINT / SIGTERM / Ctrl+C shutdown without orphaned processes.

1. **Atomic Subtitle IO & Resumability**:

   - All subtitle output (`.srt`, `.vtt`, `.txt`) writes to a temporary file first before atomically renaming into the destination path.
   - If an output file already exists, the pipeline safely skips processing to save time and compute.

## Verification Commands

```powershell
# Run model and optimizer tests
poetry run pytest tests/modules/test_models.py -v
poetry run pytest tests/modules/pipeline/translation/test_translation.py -v
poetry run pytest tests/modules/pipeline/translation/test_isolated.py -v

# Verify full pipeline
.\run_local_pipeline.ps1
```
