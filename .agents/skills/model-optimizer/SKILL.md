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

1. **NVIDIA ASR Engines (`modules/asr/`, `modules/runtime/vram_tuning.py`)**:

   - Canary-1B-v2 and Parakeet-TDT-0.6B-v3 run in-process through `ModelManager.get_asr(engine)` (one slot) and are released with `offload_asr()`; Whisper is offloaded before they load, and `offload_asr()` runs before translation. More than 64 MB still allocated after release logs a WARNING.
   - Device: `cuda:0` in bf16 only with native support (`is_bf16_supported(including_emulation=False)`), else fp32; CPU fp32 under `--cpu`. A load OOM moves to the CPU with a WARNING. Never `device_map="auto"` or `attn_implementation="eager"`.
   - Measured bf16 weights on an RTX 3080 Laptop: Canary 1.83 GB, Parakeet 1.18 GB (`ASR_WEIGHT_GB` charges 1.9 / 1.2). Under `auto`, a model whose weights plus a 1 GB activation floor do not fit the free VRAM falls back to Whisper (`asr_gpu_shortfall`).
   - `apply_dynamic_asr_batch` sizes span batches from the VRAM free after load, capped per profile (`ASR_BATCH_CAPS`: ULTRA 32, HIGH 16, MID 8, LOW 4, CPU_ONLY 4); `performance.asr_batch` pins it. `ASR_PER_SECOND_GB` values are placeholders until calibrated (`docs/hardware_optimization.md`).
   - Checkpoints are pinned (`asr_settings.MODEL_REVISIONS`) and loaded via `load_with_cache_recovery` (corrupt safetensors are purged and re-fetched).

1. **Subprocess Process Isolation (`isolated_translator.py`)**:

   - To prevent CUDA VRAM fragmentation and Out-Of-Memory (OOM) leaks between Whisper transcription and translation backends, translation runs in a separate child process.
   - The orchestrator completely offloads and frees Whisper and NVIDIA ASR VRAM before spinning up translation workers.
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
poetry run pytest tests/modules/test_models_asr.py tests/modules/runtime/test_vram_tuning_asr.py tests/modules/asr -v

# Verify full pipeline
.\run_local_pipeline.ps1
```
