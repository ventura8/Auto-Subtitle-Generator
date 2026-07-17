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
│   ├── configuration/          # Runtime configuration loading/validation
│   ├── media/                  # FFmpeg and hardware-related helpers
│   ├── pipeline/               # Transcription/translation stages
│   ├── runtime/                # Logging and progress/runtime utilities
│   ├── subtitles/              # Subtitle IO and timestamp helpers
│   ├── models.py               # AI model wrappers + optimizer
│   └── utils.py                # Shared utility helpers
├── docs/                       # Technical documentation
│   └── releases/               # Versioned release notes
├── tests/                      # Pytest suite
├── .github/workflows/ci.yml    # CI mirror of lint/type/security/test gates
└── assets/                     # Logos and media
```

## ✅ Quality Gate Summary

- Canonical local gate: `run_local_pipeline.ps1`
- Enforced checks: suppression scanner, markdown checks, Ruff, Flake8, Pylint,
  Bandit, pip-audit, pytest coverage, per-file coverage threshold

## 🚢 Release Artifacts

- Versioned release notes live in `docs/releases/vX.Y.Z.md`.
- GitHub release body drafts live in
  `docs/releases/vX.Y.Z-github-release.md`.
