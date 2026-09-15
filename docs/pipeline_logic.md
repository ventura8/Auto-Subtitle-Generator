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

### Resume candidate selection (orchestrated in `auto_subtitle.py`)

- **Role**: Reuses valid source SRT files to avoid repeating transcription.
- **Translation safety**: When the recorded source language is available,
  configured translation-target SRT files are excluded from source candidates.
  If the recorded source is unavailable, existing SRT discovery remains a
  fallback for resumability.

### Symlink-safe sidecar writes (defined in `modules/safe_io.py`)

- **Role**: Every file the pipeline writes next to the input video (source and
  translated SRTs, `*.common_input.json`, `*.manifest.json`,
  `*.pivot_pivoted.json`, `*.source_lang.txt`, `*_temp.wav`, and the final
  `_multilang` container) is produced through `atomic_text_writer` or the
  `reserve_temp_path` / `promote_temp_path` pair.
- **Threat model**: The input directory is untrusted. A planted symlink with a
  predictable sidecar name would otherwise be followed by `open(..., "w")` or
  FFmpeg `-y` and overwrite whatever it points at.
- **Mechanism**: Every write is a `ScratchReservation` — a private `0700`
  directory beside the destination, an exclusively created file inside it
  (`O_CREAT | O_EXCL | O_NOFOLLOW`, umask applied at creation), and the
  recorded device/inode identities of both. Text is written through the
  creation descriptor and never reopened by pathname; FFmpeg is pointed at
  the reserved file. Promotion re-verifies both identities and, on POSIX,
  renames through the held directory descriptor (`renameat`), so a renamed
  or replaced scratch directory cannot redirect it; a symlink destination or
  a replaced scratch entry raises `SymlinkRefusedError`. Discard is bound the
  same way and removes the directory with `rmdir` only — never recursively.
- **Naming**: Scratch entries use the opaque prefix
  `.asg-tmp-<stem[:24]>-<sha256[:8]>-`, so long basenames cannot hit
  `ENAMETOOLONG` and distinct owners cannot collide. `cleanup_temp_files`
  deliberately does not match these entries: they are removed by their
  reservation (also on `TimeoutExpired`, via `finally`), and the per-video
  scan must never walk a directory found in the untrusted input folder.

### `SystemOptimizer` (defined in `modules/models.py`)

- **Role**: Auto-detects hardware and sets performance profiles (ULTRA, HIGH,
  etc.).

### `ModelManager` (defined in `modules/models.py`)

- **Role**: Lazy loader for heavy AI models, ensuring they reside in memory
  once.

### `modules/runtime/model_cache.py` (Model Download Integrity & Auto-Recovery)

- **Role**: Detects corrupt or incomplete model checkpoints across BS-Roformer,
  Faster-Whisper, NLLB, and TranslateGemma.
- **Auto-Recovery**: Automatically purges stale/corrupted disk caches and
  triggers a clean re-download transparently before inference.

### `run_batch_translation_worker` (defined in `modules/isolated_translator.py`)

- **Role**: Runs optional pivot translation and all target jobs in one worker
  lifecycle.
- **Benefit**: Prevents duplicate translator weight loading for non-English
  input.

### `utils.log` (shared utility)

- **Role**: Thread-safe logging to both console and `subtitle_gen.log`.
