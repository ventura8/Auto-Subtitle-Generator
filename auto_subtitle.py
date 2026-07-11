"""
Auto Subtitle Generator - Optimized Edition
============================================
High-performance multilingual subtitle generation with hardware auto-tuning.

Features:
- Auto-detects GPU/CPU and applies optimal settings
- Transcribes audio using Faster-Whisper (large-v3)
- Translates to 30+ languages using NLLB-200
- Embeds all subtitles into the video container

Prerequisites:
    Run 'install_dependencies.ps1' to setup the environment.
"""

import argparse
import gc
import importlib
import logging
import multiprocessing
import os
import site
import sys
import time
import warnings

from modules import config
from modules import models
from modules import utils
from modules.models import ModelManager, OPTIMIZER
from modules.transcription import transcribe_video_audio
from modules.translation import translate_segments
from modules.utils import log, print_progress_bar

warnings.filterwarnings("ignore", category=UserWarning, message=".*expandable_segments not supported.*")
warnings.filterwarnings("ignore", message=".*The following generation flags are not valid.*")


# Torch runtime holder; tests may monkeypatch module-level "torch".
_RUNTIME_STATE = {"torch": None}


def _get_torch_module():
    """Return torch module reference, honoring any test monkeypatch on module attribute."""
    patched_torch = globals().get("torch")
    if patched_torch is not None:
        return patched_torch
    return _RUNTIME_STATE["torch"]


def _set_torch_module(torch_module):
    """Persist torch module reference for runtime and tests."""
    _RUNTIME_STATE["torch"] = torch_module
    globals()["torch"] = torch_module


logging.getLogger("transformers").addFilter(lambda record: "tied weights" not in record.getMessage())

INIT_TOTAL_STEPS = 6


# =============================================================================
# AI ENGINE INITIALIZATION
# =============================================================================


def _render_init_progress(step, total_steps, stage, status="OK"):
    """Render a consistent initialization progress update."""
    suffix = f"{stage:<35} [{status}]"
    print_progress_bar(step, total_steps, prefix="[Init] ", suffix=suffix, length=25, decimals=1)


def _init_torch_and_hardware(step, total_steps):
    """Initializes PyTorch and hardware detection."""
    # Step 1: PyTorch
    try:
        torch_module = importlib.import_module("torch")
        if _get_torch_module() is None:
            _set_torch_module(torch_module)
        _render_init_progress(step, total_steps, "Loading PyTorch")
    except ImportError as e:
        _render_init_progress(step, total_steps, "Loading PyTorch", "FAIL")
        print("")
        log(f"[Fatal] PyTorch missing: {e}", "CRITICAL")
        sys.exit(1)

    # Step 2: Hardware Detection
    step += 1
    OPTIMIZER.detect_hardware(verbose=False)
    _render_init_progress(step, total_steps, "Detecting Hardware")
    return step


def _init_nvidia_and_transformers(step, total_steps):
    """Initializes NVIDIA paths and Transformers."""
    # Step 3: NVIDIA Paths
    step += 1
    load_nvidia_paths()
    _render_init_progress(step, total_steps, "Configuring NVIDIA Runtime")

    # Step 4: Transformers
    step += 1
    try:
        importlib.import_module("transformers")

        _render_init_progress(step, total_steps, "Loading Transformers (NLLB)")
    except ImportError:
        _render_init_progress(step, total_steps, "Loading Transformers (NLLB)", "FAIL")
        print("")
        log("[Fatal] Transformers missing.", "CRITICAL")
        sys.exit(1)
    return step


def _init_whisper_and_separator(step, total_steps):
    """Initializes Faster-Whisper and Audio-Separator."""
    # Step 5: Faster-Whisper
    step += 1
    try:
        importlib.import_module("faster_whisper")

        _render_init_progress(step, total_steps, "Loading Faster-Whisper")
    except ImportError:
        _render_init_progress(step, total_steps, "Loading Faster-Whisper", "FAIL")
        print("")
        log("[Fatal] Faster-Whisper missing.", "CRITICAL")
        sys.exit(1)

    # Step 6: Audio-Separator
    step += 1
    try:
        importlib.import_module("audio_separator.separator")

        _render_init_progress(step, total_steps, "Loading Audio-Separator")
    except ImportError:
        _render_init_progress(step, total_steps, "Loading Audio-Separator", "SKIP")
        log("[Warning] audio-separator not installed. Vocal separation will be skipped.", "WARNING")

    return step


def init_ai_engine():
    """Lazily loads all AI dependencies with a progress indicator."""
    if _get_torch_module() is not None:
        return

    total_steps = INIT_TOTAL_STEPS
    step = 1

    step = _init_torch_and_hardware(step, total_steps)
    step = _init_nvidia_and_transformers(step, total_steps)
    step = _init_whisper_and_separator(step, total_steps)

    if step == total_steps:
        _render_init_progress(step, total_steps, "Initialization Complete")


def _get_nvidia_bin_lib_paths(sp):
    """Internal helper to find bin/lib in nvidia subdirs."""
    paths = []
    nvidia_path = os.path.join(sp, "nvidia")
    if os.path.exists(nvidia_path):
        for item in os.listdir(nvidia_path):
            sub_path = os.path.join(nvidia_path, item)
            if os.path.isdir(sub_path):
                for d in ["bin", "lib"]:
                    p = os.path.join(sub_path, d)
                    if os.path.exists(p):
                        paths.append(p)
    return paths


def _apply_paths_to_env(paths):
    """Internal helper to update PATH and DLL directories."""
    raw_path = os.environ.get("PATH", "")
    normalized_entries = {os.path.normcase(os.path.normpath(entry)) for entry in raw_path.split(os.pathsep) if entry}

    for p in paths:
        normalized_p = os.path.normcase(os.path.normpath(p))
        if normalized_p not in normalized_entries:
            os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
            normalized_entries.add(normalized_p)
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(p)
                except (AttributeError, OSError):
                    pass


def load_nvidia_paths():
    """Adds Torch/NVIDIA DLLs to PATH to fix ONNX Runtime 'CUDAExecutionProvider not available'."""
    paths_to_add = []

    # 1. Site Packages
    site_packages = site.getsitepackages()
    manual_site = os.path.join(sys.prefix, "Lib", "site-packages")
    if manual_site not in site_packages:
        site_packages.append(manual_site)

    for sp in site_packages:
        paths_to_add.extend(_get_nvidia_bin_lib_paths(sp))

    # 2. Torch libs
    torch_module = _get_torch_module()
    if torch_module is None:
        try:
            torch_module = importlib.import_module("torch")
            _set_torch_module(torch_module)
        except ImportError:
            torch_module = None

    if torch_module is not None and hasattr(torch_module, "__path__"):
        for q in torch_module.__path__:
            lib_path = os.path.join(q, "lib")
            if os.path.exists(lib_path):
                paths_to_add.append(lib_path)

    # 3. Apply
    _apply_paths_to_env(paths_to_add)

    try:
        importlib.import_module("onnxruntime")
    except ImportError:
        pass


# =============================================================================
# PIPELINE FUNCTIONS
# =============================================================================


def _check_resume(folder, base_name, forced_lang=None):
    """Checks if a valid SRT exists to skip transcription."""
    if forced_lang:
        srt_path = os.path.join(folder, f"{base_name}.{forced_lang}.srt")
        if os.path.exists(srt_path):
            segs = utils.parse_srt(srt_path)
            if segs:
                log(f"  [Resume] Found valid SRT: {srt_path}")
                return segs, forced_lang, srt_path
            log(f"  [Resume] SRT {srt_path} is empty or corrupted. Skipping.", "WARNING")
        return None, None, None
    # Check commonly generated ones
    for lang_code in ["en", "ro", "es", "fr"]:
        srt_path = os.path.join(folder, f"{base_name}.{lang_code}.srt")
        if os.path.exists(srt_path):
            segs = utils.parse_srt(srt_path)
            if segs:
                log(f"  [Resume] Found valid SRT: {srt_path}")
                return segs, lang_code, srt_path
            log(f"  [Resume] SRT {srt_path} is empty or corrupted. Skipping.", "WARNING")

    return None, None, None


def _get_output_filenames(video_path, folder, forced_lang):
    """Determines filenames based on video path and language."""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    final_output = os.path.abspath(os.path.join(folder, f"{base_name}_multilang.mp4"))

    lang = forced_lang or "en"
    srt_path = os.path.abspath(os.path.join(folder, f"{base_name}.{lang}.srt"))

    return final_output, srt_path, base_name


def embed_subtitles(video_path, srt_files):
    """Embeds all subtitle tracks into the video container using FFmpeg."""
    if not srt_files:
        return

    dir_name = os.path.dirname(video_path)
    file_name = os.path.basename(video_path)
    name_no_ext, ext = os.path.splitext(file_name)
    normalized_ext = ext.lower()
    output_path = os.path.join(dir_name, f"{name_no_ext}_multilang{ext}")

    cmd = [utils.FFMPEG_CMD, "-y", "-i", video_path]

    for srt, _, _ in srt_files:
        cmd.extend(["-sub_charenc", "UTF-8", "-i", srt])

    cmd.extend(["-map", "0:v", "-map", "0:a"])
    for i in range(len(srt_files)):
        cmd.extend(["-map", f"{i + 1}"])

    cmd.extend(
        [
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            "-c:s",
            "mov_text" if normalized_ext in [".mp4", ".m4v", ".mov"] else "srt",
        ]
    )

    for i, (_, lang, label) in enumerate(srt_files):
        cmd.extend([f"-metadata:s:s:{i}", f"language={lang}", f"-metadata:s:s:{i}", f"title={label}"])

    cmd.extend(["-loglevel", "info", output_path])

    try:
        total_dur = utils.get_audio_duration(video_path)
        utils.run_ffmpeg_progress(cmd, "  [Finalizing] Muxing Video", total_dur)
    except (OSError, RuntimeError, ValueError) as e:
        log(f"Embedding failed: {e}", "ERROR")
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass


def _obtain_segments(transcription_context, model_mgr, forced_lang, forced_prompt):
    """Internal helper to either load an existing SRT or run transcription."""
    folder = transcription_context["folder"]
    base_name = transcription_context["base_name"]
    video_path = transcription_context["video_path"]
    lang_hint = forced_lang if forced_lang else config.FORCED_LANGUAGE
    check_lang_code = lang_hint if lang_hint else None

    # Try to find existing output
    loaded_segments, loaded_lang, resume_srt_path = _check_resume(
        folder,
        base_name,
        check_lang_code,
    )

    if loaded_segments:
        log(f"  [Step 1] Skipping Transcription. Found valid SRT for {loaded_lang}.")
        return loaded_segments, loaded_lang, resume_srt_path

    # Need to Transcribe
    return transcribe_video_audio(video_path, model_mgr, forced_lang, forced_prompt)


def _finalize_video_processing(video_path, folder, base_name, src_lang, src_srt_path):
    """Internal helper to gather SRTs, embed them, and cleanup."""
    # Gather all generated SRTs
    generated_srts = []
    # Add Source
    if os.path.exists(src_srt_path):
        src_label = config.TARGET_LANGUAGES.get(src_lang, {}).get("label", src_lang.upper())
        generated_srts.append((src_srt_path, src_lang, src_label))

    # Add Translations
    for lang, info in config.TARGET_LANGUAGES.items():
        if lang == src_lang:
            continue
        lang_srt = os.path.join(folder, f"{base_name}.{lang}.srt")
        if os.path.exists(lang_srt):
            label = info.get("label", lang.upper()) if isinstance(info, dict) else lang.upper()
            generated_srts.append((lang_srt, lang, label))

    embed_subtitles(video_path, generated_srts)


def process_video(video_path, model_mgr, forced_lang=None, forced_prompt=None):
    """Orchestrates the full processing pipeline for a single video."""
    config.load_config(OPTIMIZER, log)
    folder = os.path.dirname(video_path) or "."
    final_output, _source_srt_path, base_name = _get_output_filenames(video_path, folder, None)

    # Check if this video is already done
    if os.path.exists(final_output):
        log(f"  [Skip] Output already exists: {final_output}", "INFO")
        return None, None, final_output

    try:
        utils.init_console()
        # Step 1: Transcribe (or Resume)
        transcription_context = {
            "folder": folder,
            "base_name": base_name,
            "video_path": video_path,
        }
        segments, src_lang, source_artifact_path = _obtain_segments(
            transcription_context,
            model_mgr,
            forced_lang,
            forced_prompt,
        )

        if not segments:
            log("No speech detected.", "WARNING")
            utils.cleanup_temp_files(folder, base_name, os.path.basename(video_path))
            # Fix: Return 3 values as expected by callers
            return [], None, None

        src_srt_path = os.path.join(folder, f"{base_name}.{src_lang}.srt")
        if source_artifact_path and source_artifact_path.endswith(".srt"):
            log("  [Resume] Reusing existing subtitle file. Continuing translation and muxing.", "INFO")
            src_srt_path = source_artifact_path
        else:
            # Immediate Save: Source SRT
            try:
                utils.save_srt(segments, src_srt_path)
            except (OSError, ValueError) as e:
                log(f"  [Error] Failed to save source SRT: {e}", "ERROR")
                return None, None, None

        # Step 2: Translate
        try:
            # Proactively offload transcription/separation models to clear VRAM
            model_mgr.offload_whisper()
            model_mgr.offload_separator()

            # FORCE CLEAN STATE (Paranoid Mode)
            gc.collect()
            torch_module = _get_torch_module()
            if torch_module is not None and hasattr(torch_module, "cuda") and torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()

            # Refactored: Calls module function
            translate_segments(segments, src_lang, model_mgr, folder, base_name)
        except (RuntimeError, OSError, ValueError) as e:
            log(f"Translation failed: {e}", "ERROR")
            # Continue to finalizing even if translation fails

        # Step 3: Finalize (Embed Subtitles)
        _finalize_video_processing(video_path, folder, base_name, src_lang, src_srt_path)

        # Return results for callers (e.g. tests)
        return segments, src_lang, final_output

    except (RuntimeError, OSError, ValueError, TypeError, KeyError) as e:
        log(f"Processing failed for {video_path}: {e}", "ERROR")
        return None, None, None

    finally:
        # Cleanup
        utils.cleanup_temp_files(folder, base_name, os.path.basename(video_path))


def get_input_files():
    """Parses command line args or prompts user for input."""
    parser = argparse.ArgumentParser(description="Auto Subtitle Generator")
    parser.add_argument("input_path", nargs="?", help="Video file or folder path")
    parser.add_argument("--lang", help="Force source language (e.g., 'en', 'ro')")
    parser.add_argument("--prompt", help="Custom initial prompt for Whisper")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")

    args = parser.parse_args()

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    path = utils.resolve_input_path(args.input_path)
    files = utils.collect_video_files(path)

    return files, args.lang, args.prompt


def setup_environment():
    """Global setup for multiprocessing and signals."""
    multiprocessing.freeze_support()
    utils.init_console()
    utils.setup_signal_handlers()


def _process_batch_video(video_path, index, total_files, process_context):
    """Process one video in a batch and return status plus per-file metrics."""
    model_mgr, forced_lang, forced_prompt = process_context
    print(f"\n[{index + 1}/{total_files}] Processing: {video_path}")
    start_time = time.time()
    process_result = process_video(video_path, model_mgr, forced_lang, forced_prompt)
    status = utils.classify_batch_result(process_result)
    elapsed_seconds = time.time() - start_time
    summary_message, media_seconds, item_stats = utils.build_file_summary(video_path, elapsed_seconds, status)
    log(summary_message, "INFO")
    return status, media_seconds, item_stats


def process_video_batch(video_files, model_mgr, forced_lang, forced_prompt):
    """Processes a list of video files."""
    batch_start_time = time.time() if len(video_files) > 1 else None
    process_context = (model_mgr, forced_lang, forced_prompt)
    total_media_seconds = 0.0
    counters = {"succeeded": 0, "no_speech": 0, "failed": 0}
    file_stats = []

    for i, video_path in enumerate(video_files):
        status, media_seconds, item_stats = _process_batch_video(
            video_path,
            i,
            len(video_files),
            process_context,
        )
        counters[status] += 1
        total_media_seconds += media_seconds
        file_stats.append(item_stats)

    utils.log_batch_summary(
        len(video_files),
        counters,
        total_media_seconds,
        batch_start_time,
        file_stats,
    )


def main():
    """Initialize environment and process all discovered videos."""
    print("[AI ENGINE INITIALIZATION]")
    _render_init_progress(0, INIT_TOTAL_STEPS, "Starting", "RUN")
    setup_environment()
    init_ai_engine()
    utils.print_banner(models.OPTIMIZER)

    video_files, forced_lang, forced_prompt = get_input_files()

    if not video_files:
        log("No video files found.", "WARNING")
        sys.exit(0)

    log(f"Found {len(video_files)} videos to process.", "INFO")

    model_mgr = ModelManager()
    process_video_batch(video_files, model_mgr, forced_lang, forced_prompt)

    print("\n[Done] All tasks completed.")


if __name__ == "__main__":
    main()
