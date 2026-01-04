# Key Logic & Pipeline

## Core Processing Pipeline (The 5-Step Flow)

1. **Audio Prep**: Extracts audio from video and converts it to **44.1kHz Stereo 32-bit Float WAV** (High Fidelity).
2. **Vocal Separation**: Uses `BS-Roformer` to isolate vocals, removing background music/noise for better transcription accuracy.
3. **AI Transcription**: Uses `faster-whisper` (Large-v3) with **Contextual Prompting** and anti-hallucination filtering.
4. **AI Translation**: Uses `facebook/nllb-200-3.3B` for batch translation. Process is **isolated in a subprocess** (`isolated_translator.py`) to maximize VRAM recovery between languages.
5. **Final Muxing**: Embeds the original video and all generated SRT tracks into a final `.mkv` container.

## Key Logic Components

### `_transcribe_video_audio` (orchestrated in `auto_subtitle.py`)
- **Role**: Simplified, single-function transcription logic.
- **Hallucination Protection**: Filters out known AI artifacts (e.g., "Thanks for watching") during silent/noisy periods.
- **Contextual Seeding**: Automatically uses the video filename as the initial prompt context.

### `SystemOptimizer` (defined in `modules/models.py`)
- **Role**: Auto-detects hardware and sets performance profiles (ULTRA, HIGH, etc.).

### `ModelManager` (defined in `modules/models.py`)
- **Role**: Lazy loader for heavy AI models, ensuring they reside in memory once.

### `utils.log` (shared utility)
- **Role**: Thread-safe logging to both console and `subtitle_gen.log`.
