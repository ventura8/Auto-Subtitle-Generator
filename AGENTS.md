# Project Agent Rules & Development Guidelines

## Project Overview

`Auto-Subtitle-Generator` is a local-first, GPU-accelerated video/audio subtitle
generation and translation pipeline. It leverages `faster-whisper`
(CTranslate2) for speech recognition, Hugging Face `transformers`
(`NLLBTranslator` default, `TranslateGemmaTranslator` optional) in isolated
child processes for translation, and `audio-separator` for vocal isolation.

- **Primary Target OS**: Windows 11 / Windows 10 (PowerShell first,
  cross-platform compatible).
- **Python Version**: Python `3.12.x` (managed via Poetry).
- **Core Orchestrator**: `auto_subtitle.py`.
- **Modular Subpackages**: `modules/` (`configuration/`, `media/`, `pipeline/`,
  `runtime/`, `subtitles/`), plus `modules/safe_io.py` for symlink-safe
  sidecar/temp writes and `modules/workdir.py` for the per-video work
  directory that holds every temporary artifact.

______________________________________________________________________

## Strict Quality & Policy Invariants

### 1. Zero Suppressions Allowed (Mandatory)

- **NEVER** introduce suppression directives: `# noqa`, `# type: ignore`,
  `# pylint: disable`, `# bandit: disable`, or warning-ignore filters.
- `tests/tools/check_no_suppressions.py` is enforced on every pipeline run
  across all code, tests, and configurations.
- Fix underlying type signatures, logic branches, and lint issues at their
  root cause.

### 2. Cyclomatic Complexity Limit (< 10)

- Every function and method across `auto_subtitle.py`, `modules/`, and
  `tests/` must maintain Cyclomatic Complexity of **A-rank (< 10)**.
- Monolithic functions must be decomposed into small, testable,
  single-responsibility helper functions.

### 3. Strict Code Coverage Thresholds (>= 90%)

- Overall pipeline code coverage must be $\\ge 90%$.
- Every key module must independently achieve $\\ge 90%$ line and branch
  coverage:
  - `auto_subtitle.py`
  - `modules/configuration/config.py`
  - `modules/pipeline/isolated_translator.py`
  - `modules/models.py`
  - `modules/pipeline/transcription.py`
  - `modules/pipeline/translation.py`
  - `modules/utils.py`
  - `modules/workdir.py`

### 4. Mandatory Documentation Synchronization (Strict)

- **ALWAYS** update all relevant `.md` documentation files (`AGENTS.md`, `README.md`,
  `docs/`, `.github/instructions/`, and `.agents/skills/`) whenever code,
  architecture, model behaviors, flags, or workflows are modified.
- Never complete a task or change without synchronizing the corresponding markdown
  docs to prevent documentation drift.
- Ensure all updated markdown files pass `mdformat` auto-formatting and
  `pymarkdown scan` checks.

### 5. Canonical Local Quality Gate

- `.\run_local_pipeline.ps1` is the canonical gate. It executes:
  1. Suppression scanner (`check_no_suppressions.py`).
  1. Markdown auto-formatting & scan (`mdformat` + `pymarkdown`).
  1. Python linting (`ruff format --check`, `ruff check`, `flake8`, `pylint`).
  1. Type checking (`mypy`, `pyright`).
  1. Security scans (`bandit`, `pip-audit` over `tests/tools/lock_requirements.py`,
     which selects the lock entries whose environment markers apply here).
  1. Radon maintainability metrics (`radon cc`, `radon mi`, `radon hal`).
  1. Test suite execution with coverage (`pytest --cov`).
  1. Per-file $\\ge 90%$ coverage verification.
  1. Badge and metric generation (`genbadge`, `transform_metrics.py`).

### 6. Prefer Installed Dependencies Over Built Ones (Mandatory)

- Always resolve an external binary or library from the **system/environment
  installation first**. A bundled, vendored, or locally built copy is only ever
  a **fallback** when nothing is installed.
- Rationale: installed packages receive OS security updates, match the host's
  architecture and codec/driver set, and avoid shipping a stale duplicate that
  silently diverges from what the installer verified.
- Discovery order for any external tool is therefore:
  1. `shutil.which(...)` / `PATH` lookup (installed).
  1. Bundled or venv-local copies (built).
  1. A bare command name as a last-resort fallback.
- Installer scripts and runtime discovery **must agree** on this order.
  `install_dependencies.sh` probes the system FFmpeg before the venv copy, so
  `modules/media/ffmpeg_utils.get_ffmpeg_paths()` must do the same.
- Do not add a build/vendor step for a dependency that can be installed via the
  platform package manager or an existing wheel.

______________________________________________________________________

## Architectural Contracts

1. **Orchestration vs Modules**:
   - `auto_subtitle.py` handles CLI parsing, batch loops, high-level staging,
     summary logging, and exit codes.
   - Reusable business logic lives under `modules/`.
1. **GPU Memory & Optimizer Profiles**:
   - **Never** use `device_map="auto"`. Use explicit CUDA device indices and
     compute types.
   - Memory profiles (`ULTRA` >= 24 GB, `HIGH` >= 12, `MID` >= 6, `LOW`,
     `CPU_ONLY`) set batch caps. The decisions that must match the actual
     card live in `modules/runtime/vram_tuning.py`: `models.nllb: auto`
     selects the largest NLLB whose fp16 weights fit 60 % of usable VRAM,
     Faster-Whisper uses `int8_float16` below 6 GB, and the translation
     worker sizes `nllb_batch` from the VRAM free *after* the model loads
     (`apply_dynamic_translation_batch`) unless `performance.nllb_batch`
     pins it. `performance.max_vram_usage_gb` caps the VRAM planned against.
     Constants are calibrated measurements (`docs/hardware_optimization.md`);
     re-measure before changing them. A model that does not fit falls back
     to the CPU with a logged warning; never let that path be silent.
   - The worker batches cues by text length and restores cue order, so
     padding stays small; keep any new batching path doing the same.
   - `_apply_profile_tuning` must not overwrite a pinned `nllb_batch`
     (`nllb_batch_overridden`): the worker loads config before it detects
     hardware. `performance.max_vram_usage_gb` is kept as a float end to end.
   - Probe CUDA through `optional_imports.is_cuda_usable(torch)`, never
     `torch.cuda.is_available()` alone: torch 2.14 reports availability even
     with the GPU hidden. `--cpu` hides devices with `CUDA_VISIBLE_DEVICES=-1`
     via `bootstrap.force_cpu_only_env()` (an empty string is not honoured),
     and `nvidia_paths.is_cuda_explicitly_disabled()` must keep recognising
     that value so bundled cuBLAS is never injected into a CPU-only run.
1. **Subprocess Process Isolation**:
   - Translation runs in `modules/pipeline/isolated_translator.py` to prevent
     CUDA memory fragmentation and guarantee full VRAM reclamation.
1. **Per-Video Work Directory & Temp Hygiene** (`modules/workdir.py`):
   - Every temporary artifact for one input video lives in
     `<folder>/<base_name>.asg-temp/`: `*_temp.wav`, the isolated
     `*_(Vocals)_*.wav` stem, the per-chunk `*_sepchunk_NNN.wav` stems of a
     chunked separation, `*.common_input.json`, `*.manifest.json`,
     `*.pivot_pivoted.json`, `.temp_output.*.json` worker outputs and their
     `.tmp` staging files, `*.source_lang.txt`, and every `.asg-tmp-*`
     scratch entry (including the ones backing SRT and `_multilang` writes).
     Only deliverables (`<base>.<lang>.srt`, `<base>_multilang.<ext>`) are
     written beside the video. **Never** create a temp file anywhere else.
   - Terminal states (muxed output written, output already present, or no
     speech) purge the work directory with `workdir.purge_work_dir` and sweep
     legacy sidecars beside the video with `utils.cleanup_temp_files`. Any
     other exit (failure, exception, Ctrl+C, power loss) keeps the directory
     intact so the next run resumes from the last completed stage.
   - Resume state belongs to one exact input: `process_video` calls
     `workdir.bind_work_dir_to_source` first, which stamps the directory with
     the input's size and mtime (`source.json`) and discards the whole
     directory when the stamp is missing or differs. A video replaced by
     another file of the same name must never be transcribed from the old
     file's extracted audio. On `BIND_CHANGED` the pipeline also ignores the
     SRT files beside the video (source resume, target skip and the English
     pivot SRT) for that run; it never deletes deliverables.
   - The purge is shallow by design: unlink files and links directly inside
     the work directory, empty `.asg-tmp-*` scratch directories one level
     down, then `rmdir`. Unknown sub-directories are left and reported.
     A symlink or junction at the work-directory name is refused.
1. **Atomic, Symlink-Safe Output & Resumability**:
   - Every pipeline write (SRTs, JSON manifests, `*.source_lang.txt`,
     `*_temp.wav`, the `_multilang` container) goes through
     `modules/safe_io.py` as a `ScratchReservation`: private `0700`
     directory + `O_EXCL | O_NOFOLLOW` file with recorded identities; text is
     written through the creation descriptor; promotion/discard re-verify
     identities and bind to the held directory descriptor on POSIX; a symlink
     destination or replaced scratch entry raises `SymlinkRefusedError`.
     Scratch entries use the opaque `.asg-tmp-` prefix and are placed in the
     work directory (`scratch_dir=`) so an interrupted write never leaves
     anything beside the video. **Never** `rmtree` or otherwise recurse into
     anything found in the input directory, and never read or change the
     process umask.
   - **Never** write a pipeline output with a bare `open(path, "w")` or point
     FFmpeg `-y` at a predictable name in the input directory. Audio-Separator
     writes into a private scratch directory inside the work directory; only
     the finished vocal stem is moved out.
   - Existing outputs are safely skipped when valid subtitles already exist.
     A resumed vocal stem must probe to the same duration as `*_temp.wav`;
     a truncated stem is discarded. A resumed `*.pivot_pivoted.json` must
     match the current segment timings; a stale one is discarded.
1. **Long Inputs Are Chunked Where Memory Demands It**:
   - Audio-Separator loads the whole input into RAM at 44.1 kHz stereo
     float32 and writes a stem of the same shape, so inputs longer than
     `config.SEPARATION_CHUNK_MINUTES` (default 30; 0 disables) are
     separated window by window in `modules/pipeline/transcription.py`:
     each window is cut with 2 s of context on both sides, separated in the
     private scratch directory, trimmed back to the window and downmixed to
     16 kHz mono as `<base>_sepchunk_NNN.wav` in the work directory, then
     all stems are joined with the FFmpeg concat demuxer. A finished chunk
     stem of the right duration is reused on resume. A tail shorter than
     60 s folds into the previous window.
   - Every WAV FFmpeg writes carries `-rf64 auto`, so nothing breaks at the
     4 GB RIFF limit. Whisper (`faster-whisper`) decodes to 16 kHz mono and
     windows internally, the translation worker holds only segment text, and
     muxing is a stream copy, so those stages need no chunking. **Never**
     add a stage that materialises a whole multi-hour input in RAM without
     a chunked path.
1. **Model Download Integrity & Auto-Recovery**:
   - Every downloaded AI model and tokenizer checkpoint (`audio-separator`,
     `faster-whisper`, `nllb`, `translategemma`) incorporates auto-detection of
     corrupted/truncated downloads (`is_corrupt_model_error`), automated cache
     purging (`modules/runtime/model_cache.py`), and transparent re-download
     recovery before inference.

______________________________________________________________________

## Available Agent Skills

The repository provides modular skills under `.agents/skills/` and `.github/skills/`:

### Primary Agent Skills (`.agents/skills/`)

- **`pipeline-runner`**: [`.agents/skills/pipeline-runner/SKILL.md`](.agents/skills/pipeline-runner/SKILL.md)
  (Execute full local validation pipeline and per-file coverage gates).
- **`code-linter`**: [`.agents/skills/code-linter/SKILL.md`](.agents/skills/code-linter/SKILL.md)
  (Run Ruff, Flake8, Pylint, Mypy, Pyright, Bandit, and Radon without
  suppressions).
- **`test-runner`**: [`.agents/skills/test-runner/SKILL.md`](.agents/skills/test-runner/SKILL.md)
  (Run unit tests, orchestration tests, and verify $\\ge 90%$ code coverage).
- **`fix-file`**: [`.agents/skills/fix-file/SKILL.md`](.agents/skills/fix-file/SKILL.md)
  (Focused single-file repair workflow with minimal safe diffs).
- **`model-optimizer`**: [`.agents/skills/model-optimizer/SKILL.md`](.agents/skills/model-optimizer/SKILL.md)
  (Manage VRAM tiers, Faster Whisper compute types, and isolated translation).
- **`setup-dependencies`**: [`.agents/skills/setup-dependencies/SKILL.md`](.agents/skills/setup-dependencies/SKILL.md)
  (Bootstrap Python 3.12+, Poetry, PyTorch CUDA 13.2, and local FFmpeg).
- **`pr-comment-resolution`**: [`.agents/skills/pr-comment-resolution/SKILL.md`](.agents/skills/pr-comment-resolution/SKILL.md)
  (Resolve PR review comments with gh CLI & MCP).
- **`review-with-coderabbit`**: [`.agents/skills/review-with-coderabbit/SKILL.md`](.agents/skills/review-with-coderabbit/SKILL.md)
  (Run local CodeRabbit CLI reviews or replay stored findings).
- **`release-prep`**: [`.agents/skills/release-prep/SKILL.md`](.agents/skills/release-prep/SKILL.md)
  (Derive release version from branch name, update docs, and sync metadata).
- **`docs-sync`**: [`.agents/skills/docs-sync/SKILL.md`](.agents/skills/docs-sync/SKILL.md)
  (Synchronize documentation, AGENTS.md, and skills after changes).
- **`architecture-review`**: [`.agents/skills/architecture-review/SKILL.md`](.agents/skills/architecture-review/SKILL.md)
  (Review and implement structural architecture and process isolation
  changes).

### GitHub Workflow Skills (`.github/skills/`)

- **`architecture-review`**: [`.github/skills/architecture-review/SKILL.md`](.github/skills/architecture-review/SKILL.md)
  (Review and implement structural architecture and process isolation changes).
- **`docs-sync`**: [`.github/skills/docs-sync/SKILL.md`](.github/skills/docs-sync/SKILL.md)
  (Synchronize documentation, AGENTS.md, and skills after changes).
- **`fix-file`**: [`.github/skills/fix-file/SKILL.md`](.github/skills/fix-file/SKILL.md)
  (Apply a focused, minimal-diff fix to a target file end-to-end).
- **`markdown-quality`**: [`.github/skills/markdown-quality/SKILL.md`](.github/skills/markdown-quality/SKILL.md)
  (Run mdformat auto-delinter and pymarkdown quality scan across all markdown
  documents).
- **`pr-comment-resolution`**: [`.github/skills/pr-comment-resolution/SKILL.md`](.github/skills/pr-comment-resolution/SKILL.md)
  (Resolve PR review comments with gh CLI & MCP).
- **`project-setup-maintenance`**: [`.github/skills/project-setup-maintenance/SKILL.md`](.github/skills/project-setup-maintenance/SKILL.md)
  (Maintain onboarding and setup scripts, dependencies, and environment configs).
- **`release-prep`**: [`.github/skills/release-prep/SKILL.md`](.github/skills/release-prep/SKILL.md)
  (Derive release version from branch name, update docs, and sync metadata).
- **`run-local-pipeline`**: [`.github/skills/run-local-pipeline/SKILL.md`](.github/skills/run-local-pipeline/SKILL.md)
  (Execute local pipeline checks, linting, security scans, unit tests, and code
  coverage gates).
- **`setup-dependencies`**: [`.github/skills/setup-dependencies/SKILL.md`](.github/skills/setup-dependencies/SKILL.md)
  (Bootstrap Python 3.12+, Poetry, PyTorch CUDA 13.2, and local FFmpeg).

______________________________________________________________________

## Instructions & Customizations

- [`.github/copilot-instructions.md`](.github/copilot-instructions.md)
- [`.github/instructions/python.instructions.md`](.github/instructions/python.instructions.md)
- [`.github/instructions/tests.instructions.md`](.github/instructions/tests.instructions.md)
- [`.github/instructions/powershell.instructions.md`](.github/instructions/powershell.instructions.md)
- [`.github/instructions/architecture.instructions.md`](.github/instructions/architecture.instructions.md)
- [`.github/instructions/setup.instructions.md`](.github/instructions/setup.instructions.md)
