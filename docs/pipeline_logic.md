# Key Logic & Pipeline

## Core Processing Pipeline (The 6-Step Flow)

1. **Audio Prep**: Extracts audio from video and converts it to **16kHz Mono
   32-bit Float WAV**.
1. **Vocal Separation**: Uses `BS-Roformer` to isolate vocals, removing
   background music/noise for better transcription accuracy.
1. **AI Transcription**: Uses `faster-whisper` (Large-v3) with **Contextual
   Prompting** and anti-hallucination filtering.
1. **AI Translation**: Uses the configured translation engine (`nllb` or
   `translategemma`) in an **isolated subprocess** (`isolated_translator.py`).
   For non-English source language, the pivot-to-English phase and
   target-language translations run inside the same batch worker so model
   weights are loaded once per file.
1. **Final Muxing**: Embeds the original video and all generated SRT tracks into
   a final `_multilang` file that keeps the input video extension.
1. **Processing Summaries**:
   - Per-file summary: Logs total processing speed, media duration, and elapsed
     time after each input file.
   - Multi-file batch summary: Logs aggregate counts, total media duration,
     elapsed time, overall speed, and a per-file stats list.

## Key Logic Components

### `_transcribe_video_audio` (orchestrated in `auto_subtitle.py`)

- **Role**: Simplified, single-function transcription logic.
- **Hallucination Protection**: Filters out known AI artifacts (e.g., "Thanks
  for watching") during silent/noisy periods.
- **Contextual Seeding**: Automatically uses the video filename as the initial
  prompt context.

### `SystemOptimizer` (defined in `modules/models.py`)

- **Role**: Auto-detects hardware and sets performance profiles (ULTRA, HIGH,
  etc.).

### `ModelManager` (defined in `modules/models.py`)

- **Role**: Lazy loader for heavy AI models, ensuring they reside in memory
  once.

### `run_batch_translation_worker` (defined in `modules/isolated_translator.py`)

- **Role**: Runs optional pivot translation and all target jobs in one worker
  lifecycle.
- **Benefit**: Prevents duplicate translator weight loading for non-English
  input.

### `utils.log` (shared utility)

- **Role**: Thread-safe logging to both console and `subtitle_gen.log`.
