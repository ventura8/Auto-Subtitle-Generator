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

Two NLLB decisions are made against the card actually present, not just its
tier; the NVIDIA ASR engines follow the same scheme (see below).

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

## NVIDIA ASR engines (Canary / Parakeet)

`--asr canary|parakeet|auto` loads an NVIDIA model in-process through
`ModelManager.get_asr`. Whisper is offloaded first, and the ASR model is
offloaded before the translation worker starts, so the two never share the
card with NLLB.

### Device and precision

- `cuda:0` in **bf16** only when the card supports it natively
  (`torch.cuda.is_bf16_supported(including_emulation=False)`, Ampere and
  newer); otherwise fp32. fp16 is not used until a benchmark justifies it.
- The CPU in fp32 under `--cpu`, without a usable CUDA device, and on Apple
  MPS (not used).
- A CUDA OOM while loading moves the model to the CPU with a WARNING. Under
  `auto` the VRAM is checked first (`asr_gpu_shortfall`): if the weights plus
  a 1 GB activation floor do not fit the free VRAM, Whisper on the GPU is
  used instead of Canary on the CPU.

### Weights (bf16, measured on an RTX 3080 Laptop)

| Model | Weights | Charged in `vram_tuning.ASR_WEIGHT_GB` |
| --- | --- | --- |
| `nvidia/canary-1b-v2` | 1.83 GB | 1.9 GB |
| `nvidia/parakeet-tdt-0.6b-v3` | 1.18 GB | 1.2 GB |

An unknown model id is charged as the larger model (1.9 GB).

### Batch of speech spans

`apply_dynamic_asr_batch` sizes the batch from the VRAM free after the model
loads: 80 % of it divided by `ASR_PER_SECOND_GB × max_segment_seconds`,
capped per profile. `performance.asr_batch` pins it; a model on the CPU uses
the `CPU_ONLY` cap. A CUDA OOM during decoding bisects the batch.

| Profile | ASR batch cap |
| --- | --- |
| ULTRA | 32 |
| HIGH | 16 |
| MID | 8 |
| LOW | 4 |
| CPU_ONLY | 4 |

> [!WARNING] `ASR_PER_SECOND_GB` (0.010 GB/s Canary, 0.006 GB/s Parakeet)
> are **placeholders**, not measurements. Calibrate them on the 5090 with the
> method used for NLLB above, and validate on the 3080, before relying on the
> dynamic batch.

### Benchmark results

Measured 2026-10-10 with `python -m tests.tools.asr_benchmark` at commit
`6f6648c`. WER is corpus-level with **diacritics kept** (cedilla ş/ţ folded to
comma-below ș/ț first); the Open-ASR column strips diacritics as published
leaderboards do. Intervals are paired cluster-bootstrap CIs (clustered by
sentence) of Canary minus Whisper. Whisper is faster-whisper `large-v3` with
the production prompt and beam 5; Canary and Parakeet run in bf16.

**Romanian, RTX 5090 (Windows 11, full test splits):**

| Corpus | Whisper | Canary | Parakeet | Canary − Whisper (95 % CI) |
| --- | --- | --- | --- | --- |
| FLEURS-ro, 883 utts | 8.91 % | **6.90 %** | 12.60 % | −2.68 … −1.33 pp |
| VoxPopuli-ro, 1330 utts | 10.14 % | **6.45 %** | 9.15 % | −4.77 … −2.66 pp |
| Diacritic error (ă â î ș ț) | 7.22 % | **2.86 %** | 4.51 % | — |
| Long-form, 35 min, WER | 17.23 % | **7.36 %** | 11.89 % | — |
| Long-form, median cue onset | 1.24 s | **0.41 s** | 0.43 s | — |
| Reversed-speech probe | 841 chars/min | **590** | 451 | — |

Silence, white/pink/brown noise and a 440 Hz tone produce no cues on any
engine. No engine dropped an utterance in the long-form run. The language
vote picked `ro` on a 35 s English intro followed by Romanian, where
faster-whisper's own first-window detection says `en`. Canary with 4 beams
reaches 6.48 % on FLEURS-ro at the same speed on the 5090; the shipped
decoder stays greedy. The segment-cap sweep put Canary at 8.77 / 7.69 / 7.36
/ 7.24 % for 8 / 12 / 15 / 20 s, so 15 s stays the default (shorter cues for
subtitles at almost no cost).

**Speed and memory.**

| Host | Engine | FLEURS-ro WER | RTF | Peak VRAM |
| --- | --- | --- | --- | --- |
| RTX 3080 Laptop 8 GB, 200 utts | Whisper | 9.19 % | 0.083 | 4.1 GB |
| | Canary | **6.61 %** | **0.005** | 2.8 GB |
| | Parakeet | 12.31 % | 0.027 | 2.1 GB |
| Core Ultra 7 255H CPU, 8 threads, 50 utts | Whisper int8 | 7.30 % | 0.603 | — |
| | Canary fp32 | 7.89 % | **0.071** | — |
| | Parakeet fp32 | 13.71 % | 0.055 | — |

On the 8 GB card the full `--asr auto` pipeline kept Canary on `cuda:0`,
returned VRAM to 7.4 GB free after the ASR offload, and left NLLB 2.9 GB free
after load against 3.0 GB in a Whisper-only run. On the CPU the 50-utterance
sample cannot separate Canary from Whisper on accuracy (95 % CI −1.67 …
+2.99 pp), but Canary is about 8x faster, so `auto` needs no CPU exception.

**Other languages (RTX 5090, 200-utterance screen, FLEURS WER):** Canary beat
Whisper by at least 10 % relative in bg, et, hr, lt, lv, mt, sk and sl, and
VoxPopuli agreed wherever it has the language. Whisper stayed better or equal
in cs, da, de, el, en, es, fi, fr, hu, it, nl, pl, pt, ru, sv and uk; Canary
was much worse on Greek (26.6 % against 9.3 %). Parakeet was never the best
engine.

**Confirmation on the full FLEURS test splits (RTX 5090), 99 % CI:**

| Lang | Utts | Whisper | Canary | Relative | Canary − Whisper (99 % CI) |
| --- | --- | --- | --- | --- | --- |
| bg | 658 | 12.33 % | **9.32 %** | −24 % | −4.18 … −1.96 pp |
| et | 893 | 18.33 % | **12.90 %** | −30 % | −6.67 … −4.18 pp |
| hr | 914 | 10.76 % | **8.56 %** | −21 % | −3.02 … −1.38 pp |
| lt | 986 | 23.78 % | **13.61 %** | −43 % | −11.39 … −8.98 pp |
| lv | 851 | 18.57 % | **10.26 %** | −45 % | −9.40 … −7.28 pp |
| mt | 926 | 69.96 % | **18.82 %** | −73 % | −53.51 … −48.74 pp |
| sk | 792 | 9.14 % | **7.03 %** | −23 % | −2.96 … −1.23 pp |
| sl | 834 | 18.76 % | **12.16 %** | −35 % | −7.83 … −5.33 pp |

VoxPopuli agreed for every one it covers (et, hr, sk, sl also at 99 %; lt in
direction, −18 % on 41 utterances). These eight join Romanian in the shipped
`asr.routes` (`asr_settings.CANARY_DEFAULT_LANGUAGES`).

A separate synthetic-TTS evaluation of ONNX exports on an Intel Arc iGPU
(Whisper-Pro-ASR, 82 clips) found Whisper ahead. That run used fp32 ONNX
exports and synthetic speech, so it is not comparable with these numbers;
re-measure on real speech before narrowing the routes.

### Default-decision rule

The shipped default flips to `asr.engine: auto` with Romanian routed to Canary
only if **all five** hold (all five passed on 2026-10-10):

1. Full FLEURS-ro test split: Canary's diacritics-kept WER is at least 10 %
   relative better than Whisper's, and the 95 % paired cluster-bootstrap CI
   excludes 0.
1. VoxPopuli-ro: Canary is no worse than Whisper by more than 1 point
   absolute.
1. Non-speech probes: Canary's characters per minute after filters is at
   most Whisper's.
1. Synthetic long-form: Canary's WER is not worse, no utterances are
   dropped, and its median cue-onset error is at most Whisper's + 0.2 s.
1. On the RTX 3080 Laptop the Canary stage stays on the GPU, and NLLB's
   free-after-load VRAM drops by at most 0.3 GB against a Whisper-only run.

Other languages join `routes` only with a 99 % CI on the full split and
agreement on VoxPopuli where it exists. Parakeet joins `routes` only if it
qualifies the same way; otherwise it stays an explicit "fast" engine. If
Romanian fails, Whisper stays the default, the NVIDIA engines stay opt-in,
and the numbers are published here anyway.

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
- **Functionality**: Implements lazy loading for heavy AI models (Whisper, the
  optional NVIDIA ASR model, plus the configured translation backend: NLLB or
  TranslateGemma) and persists them across multiple video files in a batch.
  The single NVIDIA ASR slot is released (`offload_asr`) before translation.
- **Benefit**: Eliminates re-initialization overhead (saving ~10s per video) and
  reduces VRAM fragmentation.
