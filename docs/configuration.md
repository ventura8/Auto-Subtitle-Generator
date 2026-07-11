# Configuration

## Run-time Settings
All run-time settings are managed in `config.yaml`.

### Sections:
1.  **Whisper AI**: Model size (`large-v3`, `medium`, etc.) and optional context prompts.
2.  **Hallucination Filters**: Thresholds for silence and repetition detection, plus a **list of known hallucination phrases** to filter from output.
3.  **File Types**: List of video file extensions to process (e.g., `.mp4`, `.mkv`).
4.  **Translation Engine**: Choose `nllb` (default) or `translategemma`.
5.  **Models**: Custom IDs for TranslateGemma, NLLB, and the Audio Separator model.
6.  **Access Token**: For gated models, use the supported secure source (`HF_TOKEN` environment variable) instead of committing `hf_token` to `config.yaml`.
	Verify your local config file is ignored by version control before storing any secret-like values.
	Keep configuration/log output token-safe by redacting token values (for example: `hf_token: "***"` and `HF_TOKEN=***`) in shared logs or screenshots.
7.  **Performance**: Manual overrides for internal thread counts, beam sizes, and batch sizes.
8.  **VAD**: Voice Activity Detection parameters (e.g., minimum silence duration).

If `config.yaml` is missing, the script falls back to sensible internal defaults.
