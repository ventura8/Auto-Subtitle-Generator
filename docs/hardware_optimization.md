# Hardware Optimization

## `SystemOptimizer` (Hardware Detection)

- **Location**: `modules/models.py`.
- **Functionality**: Scans CPU (Ryzen optimization) and GPU (RTX
  50-series/Blackwell focus).
- **Profiles**: Uses `STANDARD` as the default/fallback profile and assigns one
  hardware-detection tier: `ULTRA` (>= 24 GB), `HIGH` (>= 12 GB), `MID`
  (>= 6 GB), `LOW`, or `CPU_ONLY`. Tiers set caps; the VRAM-aware tuning
  below decides the model and the batch against the actual card.

**AI Guideline**: When modifying settings, ensure they align with these
VRAM-based tiers. Always consider that the user is likely running high-end
hardware (AMD Ryzen 9 9950X3D + NVIDIA RTX 5090). Optimizations should favor
throughput while maintaining quality.

## VRAM-aware tuning (`modules/runtime/vram_tuning.py`)

Two decisions are made against the card actually present, not just its tier.

### Which NLLB model fits

`models.nllb: "auto"` (the shipped default) picks the largest NLLB whose fp16
weights fit **60 %** of the VRAM the pipeline may use (`performance.max_vram_usage_gb`
caps that when set). A model that does not fit fails to load with CUDA
out-of-memory and the loader falls back to the CPU, which is now logged as a
warning. That is what an RTX 3080 Laptop (8 GB) was doing with the 3.3B model:
130 OOM retries while loading, then 850 % CPU, 160 MiB on the GPU and
0.4x realtime. An explicit model id is always honoured, with a warning when it
will not fit.

| VRAM | Profile | NLLB on `auto` | Whisper compute |
| --- | --- | --- | --- |
| >= 24 GB | ULTRA | `nllb-200-3.3B` | float16 |
| >= 12 GB | HIGH | `nllb-200-3.3B` | float16 |
| 11 GB | MID | `nllb-200-3.3B` | float16 |
| 5-10 GB | MID | `nllb-200-distilled-1.3B` | float16 (>= 6 GB) |
| < 5 GB | LOW | `nllb-200-distilled-600M` | int8_float16 |

### How large a batch the free memory allows

After the translation worker has loaded the model it reads the VRAM still
free (`torch.cuda.mem_get_info`) and sizes `nllb_batch` from the measured
per-item activation cost, scaled by `num_beams`, using 80 % of that free
memory and never exceeding the profile cap. It logs one line, for example:

```text
[VRAM] 31 GB card, 20.3 GB free after load (6.3 GB weights)
       -> batch 16 (profile cap 16)
```

An explicit `performance.nllb_batch` disables this. Because it reads *free*
memory, a GPU shared with another process gets a smaller batch automatically.

### Measured on the RTX 3080 Laptop (8 GB)

| Build | Model | Where it ran | Batch | Per language |
| --- | --- | --- | --- | --- |
| pinned `nllb-200-3.3B` | 3.3B | CPU after 130 OOM retries | 6 | > 7 min |
| `auto` | distilled 1.3B | GPU, 2.56 GB resident | 10 (1.3 GB free) | 6 s |

The tuned run shared the card with two other GPU jobs, which is why only
1.3 GB was free after loading and the batch was sized below the cap.

### Calibration (RTX 5090, production generate settings, 10 beams)

| Model | Weights | Per item | Lines/s @8 | @16 |
| --- | --- | --- | --- | --- |
| `nllb-200-distilled-600M` | 1.15 GB | 0.063 GB | 13.9 | 28.0 |
| `nllb-200-distilled-1.3B` | 2.59 GB | 0.093 GB | 8.7 | 16.0 |
| `nllb-200-1.3B` | 2.59 GB | 0.092 GB | 10.6 | 19.2 |
| `nllb-200-3.3B` | 6.27 GB | 0.154 GB | 7.3 | 12.4 |

The calibration lines were uniform in length. Real subtitle cues are not, and
a batch is padded to its longest cue and beam-searched until the longest one
finishes, so the worker batches cues in order of length and restores cue
order afterwards. The profile cap is 16: on a 36-language run of the 2:13
test interview with the GPU shared by another process, a batch pinned at 8
or 16 took about 3.5 minutes, while 32 took 23 minutes as the allocator
churned. Sizing from *free* memory exists for exactly that situation.

## `NLLBTranslator` & OOM Recovery

- **Functionality**: Handles batch translation.
- **Smart OOM Recovery**: Aggressively clears cache and adjusts batch sizes if
  VRAM or System RAM saturation is detected.

**AI Guideline**:

- **Strict VRAM Enforcement**: ALWAYS use `device_map="cuda"` (or specific
  device) instead of `"auto"`. "Auto" allows offloading to Shared System RAM,
  which causes massive performance degradation and potential crashes.
- **Memory Management**: Implement proactive `gc.collect()` and
  `torch.cuda.empty_cache()` inside any batch processing loops (e.g.,
  translation) to preventing fragmentation.

## `ModelManager` (Persistent Loading)

- **Location**: `modules/models.py`.
- **Functionality**: Implements lazy loading for heavy AI models (Whisper plus
  the configured translation backend: NLLB or TranslateGemma) and persists them
  across multiple video files in a batch.
- **Benefit**: Eliminates re-initialization overhead (saving ~10s per video) and
  reduces VRAM fragmentation.
