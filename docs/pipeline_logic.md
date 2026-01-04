# Key Logic & Pipeline

## Core Processing Pipeline (The 5-Step Flow)

1. **Audio Prep**: Extracts audio from video and converts it to 16kHz Mono WAV.
2. **Vocal Separation**: Uses `BS-Roformer` to isolate vocals, removing background music/noise for better transcription accuracy.
3. **AI Transcription**: Uses `faster-whisper` (Large-v3) running on CUDA to generate the source language SRT.
4. **AI Translation**: Uses `facebook/nllb-200-3.3B` for batch translation into 70+ languages.
5. **Final Muxing**: Embeds the original video and all generated SRT tracks (e.g., `en`, `es`, `fr`) into a final `.mkv` container.

## Key Logic Components

### `process_video`
- **Role**: The main orchestration function.
- **Model Management**: Accepts a shared `ModelManager` instance to reuse loaded AI models.
- **Resiliency Logic**: Checks for existing intermediate files to skip completed steps.
- **Cleanup**: Aggressively deletes temporary audio files (pattern: `{video_name}_temp*`) after processing.

### Audio Fidelity
- Maintain **32-bit float processing** to prevent clipping during normalization and isolation steps.

### Resiliency Guidelines
- When adding new steps, ensure they check for existing output files to support resuming after a crash.
