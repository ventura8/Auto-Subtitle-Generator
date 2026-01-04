# Hardware Optimization

## `SystemOptimizer` (Hardware Detection)
- **Location**: `auto_subtitle.py`.
- **Functionality**: Scans CPU (Ryzen optimization) and GPU (RTX 50-series/Blackwell focus).
- **Profiles**: Assigns one of five profiles: `ULTRA`, `HIGH`, `MID`, `LOW`, or `CPU`.

**AI Guideline**: When modifying settings, ensure they align with these VRAM-based tiers. Always consider that the user is likely running high-end hardware (AMD Ryzen 9 9950X3D + NVIDIA RTX 5090). Optimizations should favor throughput while maintaining quality.

## `NLLBTranslator` & OOM Recovery
- **Functionality**: Handles batch translation.
- **Smart OOM Recovery**: Aggressively clears cache and adjusts batch sizes if VRAM or System RAM saturation is detected.

**AI Guideline**: 
- **Strict VRAM Enforcement**: ALWAYS use `device_map="cuda"` (or specific device) instead of `"auto"`. "Auto" allows offloading to Shared System RAM, which causes massive performance degradation and potential crashes.
- **Memory Management**: Implement proactive `gc.collect()` and `torch.cuda.empty_cache()` inside any batch processing loops (e.g., translation) to preventing fragmentation.

## `ModelManager` (Persistent Loading)
- **Location**: `auto_subtitle.py`.
- **Functionality**: Implements lazy loading for heavy AI models (Whisper, NLLB) and persists them across multiple video files in a batch.
- **Benefit**: Eliminates re-initialization overhead (saving ~10s per video) and reduces VRAM fragmentation.
