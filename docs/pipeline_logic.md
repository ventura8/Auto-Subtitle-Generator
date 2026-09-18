# Key Logic & Pipeline

## Core Processing Pipeline (The 6-Step Flow)

1. **Audio Prep**: Extracts audio from video and converts it to **16kHz Mono
   32-bit Float WAV**.
1. **Vocal Separation**: Uses `BS-Roformer` to isolate vocals, removing
   background music/noise for better transcription accuracy. Inputs longer
   than `separation_chunk_minutes` (default 30) are separated in resumable
   chunks and the stems joined as 16 kHz mono, so multi-hour recordings are
   bounded in RAM and never approach the 4 GB WAV limit.
1. **AI Transcription**: Uses `faster-whisper` (Large-v3) with **Contextual
   Prompting** and anti-hallucination filtering.
1. **AI Translation**: Uses the configured translation engine (`nllb` or
   `translategemma`) in an **isolated subprocess** (`isolated_translator.py`).
   With `models.nllb: auto` the NLLB size is chosen to fit the card, and the
   batch is sized from the VRAM free once the model is loaded; cues are
   batched by length to keep padding small (see
   `docs/hardware_optimization.md`).
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

### Per-video work directory (defined in `modules/workdir.py`)

- **Role**: The single home for every temporary artifact of one input video:
  `<folder>/<base_name>.asg-temp/`. It holds `*_temp.wav`, the isolated
  `*_(Vocals)_*.wav` stem, `*.common_input.json`, `*.manifest.json`,
  `*.pivot_pivoted.json`, `.temp_output.*.json` worker outputs (and their
  `.tmp` staging files), `*.source_lang.txt`, and every `.asg-tmp-*` scratch
  entry. Only the per-language SRT files and the `_multilang` container are
  written beside the video.
- **Lifecycle**: `ensure_work_dir` creates it (`0700`, refusing a planted
  symlink or junction at that name) on first use. `_finish_temp_hygiene` in
  `auto_subtitle.py` decides at the end of every video: a terminal state
  (muxed output written, output already present, or no speech) calls
  `purge_work_dir` and the legacy `cleanup_temp_files` sweep; any other exit
  (failure, exception, Ctrl+C, power loss) keeps the directory so the next run
  resumes. The translation worker likewise keeps its manifest, input, pivot and
  partial outputs when it fails and removes them only after success.
- **Bound to the input**: before any stage runs, `bind_work_dir_to_source`
  compares the `source.json` stamp (input size and mtime) with the current
  video. A missing or different stamp discards the whole directory, so a
  video replaced by another file of the same name (even one with the same
  duration) never resumes from the previous file's extracted audio, vocals
  or chunk stems. If the stale directory cannot be removed the video fails
  rather than reusing it. The same outcome (`BIND_CHANGED`) makes that run
  ignore the SRT files beside the video: transcription is not resumed from
  them, every target language is redone and the English pivot SRT is not
  reused. Deliverables are never deleted, only regenerated.
- **Resume checks**: a resumed vocal stem must be a regular file whose probed
  duration matches `*_temp.wav` (within 2 s), otherwise it is discarded; a
  resumed `*.pivot_pivoted.json` must be a list of cue objects with string
  text and the same cue timings as the current input, otherwise it is
  discarded and the pivot pass reruns.
- **Shallow purge**: files and links directly inside the work directory are
  unlinked (links are never followed), `.asg-tmp-*` scratch directories are
  emptied one level down, then `rmdir`. Unknown sub-directories are left in
  place and reported, and the purge never recurses.

### Symlink-safe sidecar writes (defined in `modules/safe_io.py`)

- **Role**: Every file the pipeline writes (source and translated SRTs,
  `*.common_input.json`, `*.manifest.json`, `*.pivot_pivoted.json`,
  `*.source_lang.txt`, `*_temp.wav`, and the final `_multilang` container) is
  produced through `atomic_text_writer` or the `reserve_temp_path` /
  `promote_temp_path` pair. Both accept `scratch_dir=` so the scratch entry is
  created inside the work directory even when the destination is beside the
  video; `create_private_dir` gives Audio-Separator a private output
  directory inside the work directory.
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
  `ENAMETOOLONG` and distinct owners cannot collide. They are removed by
  their reservation (also on `TimeoutExpired` and `KeyboardInterrupt`, via
  `finally`); any entry orphaned by a power loss is removed by
  `purge_work_dir` when the video finishes. The legacy `cleanup_temp_files`
  sweep of the video folder deliberately does not match these entries and
  never walks a directory found in the untrusted input folder.

### Chunked vocal separation (defined in `modules/pipeline/transcription.py`)

- **Why**: Audio-Separator calls `librosa.load` on the whole input at
  44.1 kHz stereo float32 and writes a stem of the same shape. A 4-hour file
  is a 5 GB array before the model starts, several times that at peak, and a
  5 GB stem past the RIFF limit. The model itself bounds VRAM by segmenting
  internally, so this is a host-RAM and file-format problem.
- **Dispatch**: `_separate` probes the extracted WAV; anything longer than
  `config.SEPARATION_CHUNK_MINUTES` takes `_run_chunked_separation`, else the
  single pass is unchanged. `0` disables chunking.
- **Windows**: `_chunk_windows` tiles the duration; a tail under 60 s folds
  into the previous window (4 h 00 m 05 s becomes eight windows, the last
  30 m 05 s).
- **Per chunk**: cut the window with 2 s of context on each side into the
  private scratch directory (`write_audio_window`), run the separator there,
  trim the context off and downmix to 16 kHz mono float
  (`write_audio_window(..., mono_16k=True)`) through a `safe_io` reservation
  onto `<base>_sepchunk_NNN.wav` in the work directory, then delete the raw
  chunk and stem. Peak RAM for a 30-minute chunk is around 3 GB.
- **Join**: the concat manifest lists every chunk stem as an absolute path,
  because FFmpeg resolves relative entries against the manifest's own
  directory rather than the working directory.
- **Resume**: `_is_finished_chunk` reuses a chunk stem that is a regular file
  of the expected duration (within 0.5 s); a truncated one is redone. A
  power loss mid-separation costs at most one chunk.
- **Join**: `concat_audio_files` writes a concat manifest into the scratch
  directory and stream-copies the stems into `<base>_temp_(Vocals)_chunked.wav`
  (RF64 when large), again through a reservation; the chunk stems are then
  removed. The joined name matches the resume scan in
  `_get_separated_vocal_path`, which validates it against `*_temp.wav`.
- **Other stages**: extraction adds `-rf64 auto` so a >17 h source WAV stays
  valid; `faster-whisper` decodes to 16 kHz mono (0.9 GB for 4 h) and windows
  internally; translation holds only text; muxing is a stream copy.

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
