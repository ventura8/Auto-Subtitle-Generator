# Project Overview

This document provides technical context for the **Auto Subtitle Generator**
project.

## 🏗 Project Architecture

The application is a high-performance, 100% local AI pipeline that processes
video files to generate and embed multi-language subtitles. It is designed for
"Bleeding Edge" hardware (NVIDIA RTX 50-series + AMD Ryzen 9000 series) with
automatic hardware detection to maximize performance.

## 📂 Directory Structure

```text
.
├── auto_subtitle.py            # Main entry point and orchestrator
├── config.yaml                 # User configuration
├── pyproject.toml              # Poetry project/dependency configuration
├── run_local_pipeline.ps1      # Local quality gate (lint + tests + coverage)
├── modules/                    # Core logic and AI models
│   ├── __init__.py
│   ├── config.py               # Internal configuration & constants
│   ├── models.py               # AI model wrappers (Whisper, NLLB,
│   │                           # TranslateGemma) and Optimizer
│   ├── utils.py                # Logging, FFmpeg, and file utilities
│   └── isolated_translator.py  # Isolated worker process for
│                               # translation jobs and pivot flow
├── docs/                       # Technical documentation
│   └── releases/               # Versioned release notes
├── tests/                      # Pytest suite
└── assets/                     # Logos and media
```
