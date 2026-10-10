# Configuration

## Run-time Settings

All run-time settings are managed in `config.yaml`.

### Sections

1. **Whisper AI**: Model size (`large-v3`, `medium`, etc.) and optional context
   prompts. `force_detected_language: true` votes the language over 8
   windows of 30 s spread across the file's speech and tells Whisper that
   language.
1. **ASR Engine** (`asr:`): which model transcribes the audio.
   - `engine`: `auto` (default), `whisper`, `canary` or `parakeet`. The
     `--asr` command-line option overrides it for one run.
   - `routes`: language (ISO 639-1) → engine, used by `auto`; replaces the
     built-in table, which sends bg, et, hr, lt, lv, mt, ro, sk and sl to
     Canary (the languages where it measurably beat Whisper). Unlisted
     languages stay on Whisper.
   - `max_segment_seconds`: longest speech span handed to Canary/Parakeet in
     one call, clamped to 5-30 (default 15).
   - Canary and Parakeet cover 25 European languages (bg cs da de el en es et
     fi fr hr hu it lt lv mt nl pl pt ro ru sk sl sv uk). Any other language,
     a model that fails to load or fit, or mostly empty output falls back to
     Whisper with a logged warning.
   - An invalid value logs a warning and keeps its default; it never stops
     the run.
1. **Hallucination Filters**: Thresholds for silence and repetition detection,
   plus a **list of known hallucination phrases** to filter from output.
1. **File Types**: List of video file extensions to process (e.g., `.mp4`,
   `.mkv`).
1. **Translation Engine**: Choose `nllb` (default) or `translategemma`.
1. **Models**: Custom IDs for TranslateGemma, NLLB, Canary
   (`models.canary`, default `nvidia/canary-1b-v2`), Parakeet
   (`models.parakeet`, default `nvidia/parakeet-tdt-0.6b-v3`), and the Audio
   Separator model. The default Canary and Parakeet checkpoints are pinned to
   tested Hugging Face revisions; any other id loads its latest revision.
1. **Access Token**: For gated models, use the supported secure source
   (`HF_TOKEN` environment variable) instead of committing `hf_token` to
   `config.yaml`. Verify your local config file is ignored by version control
   before storing any secret-like values. Keep configuration/log output
   token-safe by redacting token values (for example: `hf_token: "***"` and
   `HF_TOKEN=***`) in shared logs or screenshots.
1. **Performance**: Manual overrides for internal thread counts, beam sizes, and
   batch sizes. `asr_batch` pins the number of Canary/Parakeet speech spans
   per call; `null` sizes it from the VRAM free after the model loads.
1. **VAD**: Voice Activity Detection parameters (e.g., minimum silence
   duration).

If `config.yaml` is missing, the script falls back to sensible internal
defaults.
