______________________________________________________________________

## applyTo: "{auto_subtitle.py,modules/\*\*/\*.py}" description: Use when changing pipeline architecture, module boundaries, orchestration flow, or model lifecycle behavior.

# Architecture Instructions

## Separation of Responsibilities

- Keep `auto_subtitle.py` strictly as an orchestrator (CLI arguments, batch loops, high-level staging).
- Put reusable business logic under `modules/`.
- Avoid moving heavy operational logic into CLI entrypoints.

## Process and Memory Model

- Keep isolated heavy translation execution patterns intact (`isolated_translator.py`).
- Preserve model offload/cleanup behavior before loading subsequent heavy components.
- Never use `device_map="auto"`. Use explicit device mapping.
- Avoid introducing hidden mutable shared state across modules.
- NVIDIA ASR engines (Canary/Parakeet, `modules/asr/`) run in-process through `ModelManager.get_asr(engine)`; offload Whisper before loading them and call `offload_asr()` before translation.
- Keep engine selection in `modules/configuration/asr_settings.py` (`--asr` / `asr.engine`, `asr.routes` for `auto`) and routing in `modules/asr/routing.py`. Every fallback to Whisper logs a WARNING; never fall back silently.
- Canary needs a source language: use the forced language, else the spread Whisper vote over 8 windows (`modules/asr/language_id.py`), never faster-whisper's first-window detection.
- Never materialise a whole long input in RAM: the ASR path streams 16 kHz spans and runs VAD block by block; a long non-16 kHz input is transcoded once to `<base>_asr16k.wav` in the work directory.

## Reliability Contracts

- Preserve atomic write behavior for generated subtitles (`.tmp` write then atomic rename).
- Preserve resume/skip logic for already processed outputs.
- Reuse a pivot or target SRT only while its cue timings match the current source; regenerate a stale one, never delete it.
- Keep the decoded-speech guard: mostly empty NVIDIA output redoes the file with Whisper instead of a silent "no speech" purge.
- Keep subprocess shutdown paths robust on Windows.

## Performance Contracts

- Respect optimizer tier decisions (`ULTRA` / `HIGH` / `MID` / `LOW` / `CPU`).
- NVIDIA ASR uses bf16 only with native GPU support, else fp32; pinned Hugging Face revisions; batch sizes from `apply_dynamic_asr_batch` (`performance.asr_batch` pins it).
- Avoid hardcoding profile overrides that bypass config and detection.
- Keep FFmpeg invocation patterns compatible with existing utility wrappers.
