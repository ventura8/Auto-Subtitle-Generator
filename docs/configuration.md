# Configuration

## Run-time Settings
All run-time settings are managed in `config.yaml`.

### Sections:
1.  **Whisper AI**: Model size (`large-v3`, `medium`, etc.) and optional context prompts.
2.  **Hallucination Filters**: Thresholds for silence and repetition detection, plus a **list of known hallucination phrases** to filter from output.
3.  **File Types**: List of video file extensions to process (e.g., `.mp4`, `.mkv`).
4.  **Models**: Custom paths or IDs for NLLB (Translator) and Audio Separator (Vocals).
5.  **Performance**: Manual overrides for internal thread counts, beam sizes, and batch sizes.
6.  **VAD**: Voice Activity Detection parameters (e.g., minimum silence duration).

If `config.yaml` is missing, the script falls back to sensible internal defaults.
