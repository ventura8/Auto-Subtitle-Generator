# Copilot Instructions

This repository is a local-first subtitle generation pipeline with strict
reliability and performance constraints.

## Core Rules

- Keep orchestration in auto_subtitle.py and implementation logic in modules/.
- Preserve stateless module behavior where possible; keep long-lived model state
  in ModelManager patterns.
- Never introduce device_map="auto" for GPU model loading. Use explicit CUDA
  device mapping.
- ASR: Whisper is the default engine; NVIDIA Canary/Parakeet (modules/asr/,
  --asr / asr.engine, auto routing per language) run in-process through
  ModelManager.get_asr and are released with offload_asr() before
  translation. Every fallback to Whisper logs a WARNING; never fall back
  silently.
- Prioritize safe shutdown behavior on Windows. Do not weaken subprocess cleanup
  paths.
- Preserve resume behavior and atomic writes when touching subtitle output
  paths.

## Development Standards

- Keep every function at Radon grade A (cyclomatic complexity 1-5) and every
  file at maintainability index grade A; refactor instead of suppressing
  complexity warnings.
- Import heavy libraries (torch, transformers, faster_whisper, numpy,
  soundfile) only through importlib or load_optional_torch().
- Use clear, testable functions and avoid broad side effects.
- Update or add tests in tests/ when behavior changes.
- Keep changes minimal and avoid unrelated refactors.

## Performance Expectations

- Respect VRAM-tier profile logic and avoid hardcoding values that bypass
  optimizer decisions.
- Prefer batching/memory cleanup patterns already used by the project.
- Keep FFmpeg interactions compatible with local venv FFmpeg binaries.

## Setup and Tooling

- Python target is 3.12+ and Poetry is the package manager for project
  dependencies.
- For full local validation use run_local_pipeline.ps1.
- Use install_dependencies.ps1 for environment bootstrap, including local FFmpeg
  setup.
