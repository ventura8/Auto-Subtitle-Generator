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
from typing import Any

from modules import models, utils
from modules.configuration import config
from modules.models import OPTIMIZER, ModelManager
from modules.pipeline.transcription import transcribe_video_audio
from modules.pipeline.translation import translate_segments
from modules.utils import log, print_progress_bar

# Torch runtime holder; tests may monkeypatch module-level "torch".
_RUNTIME_STATE = {"torch": None}
_DLL_DIRECTORY_HANDLES: list[Any] = []


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
    if not os.path.exists(nvidia_path):
        return paths

    for item in os.listdir(nvidia_path):
        sub_path = os.path.join(nvidia_path, item)
        if not os.path.isdir(sub_path):
            continue
        paths.extend(_collect_existing_runtime_dirs(sub_path))
    return paths


def _collect_existing_runtime_dirs(base_dir):
    """Collect existing runtime subdirectories used by CUDA providers."""
    existing = []
    for dir_name in ["bin", "lib"]:
        candidate = os.path.join(base_dir, dir_name)
        if os.path.exists(candidate):
            existing.append(candidate)
    return existing


def _add_dll_directory_if_supported(path):
    """Register DLL directory on supported platforms without hard failure."""
    if not hasattr(os, "add_dll_directory"):
        return
    try:
        handle = os.add_dll_directory(path)
        if handle is not None:
            _DLL_DIRECTORY_HANDLES.append(handle)
    except (AttributeError, OSError):
        pass


def _collect_site_packages_paths():
    """Collect site-packages paths for runtime DLL discovery."""
    site_packages = site.getsitepackages()
    manual_site = os.path.join(sys.prefix, "Lib", "site-packages")
    if manual_site not in site_packages:
        site_packages.append(manual_site)
    return site_packages


def _get_or_import_torch_module():
    """Return cached torch module or import it when available."""
    torch_module = _get_torch_module()
    if torch_module is not None:
        return torch_module
    # Ensure runtime DLL paths are prepared before importing torch.
    paths_to_add = []
    for site_package in _collect_site_packages_paths():
        paths_to_add.extend(_get_nvidia_bin_lib_paths(site_package))
    _apply_paths_to_env(paths_to_add)
    try:
        torch_module = importlib.import_module("torch")
        _set_torch_module(torch_module)
        return torch_module
    except ImportError:
        return None


def _collect_torch_lib_paths(torch_module):
    """Collect torch lib paths needed for CUDA runtime DLL discovery."""
    if torch_module is None or not hasattr(torch_module, "__path__"):
        return []

    lib_paths = []
    for package_path in torch_module.__path__:
        lib_path = os.path.join(package_path, "lib")
        if os.path.exists(lib_path):
            lib_paths.append(lib_path)
    return lib_paths


def _apply_paths_to_env(paths):
    """Internal helper to update PATH and DLL directories."""
    raw_path = os.environ.get("PATH", "")
    normalized_entries = {os.path.normcase(os.path.normpath(entry)) for entry in raw_path.split(os.pathsep) if entry}

    for path in paths:
        _add_dll_directory_if_supported(path)
        normalized_path = os.path.normcase(os.path.normpath(path))
        if normalized_path in normalized_entries:
            continue
        os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
        normalized_entries.add(normalized_path)


def load_nvidia_paths():
    """Adds Torch/NVIDIA DLLs to PATH to fix ONNX Runtime 'CUDAExecutionProvider not available'."""
    paths_to_add = []
    for site_package in _collect_site_packages_paths():
        paths_to_add.extend(_get_nvidia_bin_lib_paths(site_package))

    torch_module = _get_or_import_torch_module()
    paths_to_add.extend(_collect_torch_lib_paths(torch_module))

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
    candidates = _get_resume_candidates(folder, base_name, forced_lang)
    for lang_code in candidates:
        if not lang_code:
            continue
        segments, srt_path = _read_resume_srt(folder, base_name, lang_code)
        if segments:
            return segments, lang_code, srt_path

    return None, None, None


def _get_resume_candidates(folder, base_name, forced_lang):
    """Return resume-language candidates honoring forced and recorded source language."""
    if forced_lang:
        return [forced_lang]

    recorded_source_lang = _read_recorded_source_language(folder, base_name)
    return [recorded_source_lang] if recorded_source_lang else []


def _read_resume_srt(folder, base_name, lang_code):
    """Read and validate a candidate resume SRT file."""
    srt_path = os.path.join(folder, f"{base_name}.{lang_code}.srt")
    if not os.path.exists(srt_path):
        return None, srt_path

    segments = utils.parse_srt(srt_path)
    if segments:
        log(f"  [Resume] Found valid SRT: {srt_path}")
        return segments, srt_path

    log(f"  [Resume] SRT {srt_path} is empty or corrupted. Skipping.", "WARNING")
    return None, srt_path


def _get_source_language_artifact_path(folder, base_name):
    """Return sidecar path used to persist the detected source language."""
    return os.path.join(folder, f"{base_name}.source_lang.txt")


def _read_recorded_source_language(folder, base_name):
    """Read previously recorded source language for resume purposes."""
    artifact_path = _get_source_language_artifact_path(folder, base_name)
    if not os.path.exists(artifact_path):
        return None
    try:
        with open(artifact_path, "r", encoding="utf-8") as file_handle:
            return file_handle.read().strip() or None
    except OSError:
        return None


def _write_recorded_source_language(folder, base_name, src_lang):
    """Persist detected source language for safe resume selection."""
    artifact_path = _get_source_language_artifact_path(folder, base_name)
    temp_path = f"{artifact_path}.tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as file_handle:
            file_handle.write(src_lang)
        os.replace(temp_path, artifact_path)
    except OSError:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _get_output_filenames(video_path, folder, forced_lang):
    """Determines filenames based on video path and language."""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    _, extension = os.path.splitext(video_path)
    final_output = os.path.abspath(os.path.join(folder, f"{base_name}_multilang{extension}"))

    lang = forced_lang or "en"
    srt_path = os.path.abspath(os.path.join(folder, f"{base_name}.{lang}.srt"))

    return final_output, srt_path, base_name


def embed_subtitles(video_path, srt_files):
    """Embeds all subtitle tracks into the video container using FFmpeg."""
    if not srt_files:
        return None

    dir_name = os.path.dirname(video_path)
    file_name = os.path.basename(video_path)
    name_no_ext, ext = os.path.splitext(file_name)
    normalized_ext = ext.lower()
    output_path = os.path.join(dir_name, f"{name_no_ext}_multilang{ext}")

    cmd = _build_embed_command(video_path, srt_files, normalized_ext, output_path)

    try:
        total_dur = utils.get_audio_duration(video_path)
        utils.run_ffmpeg_progress(cmd, "  [Finalizing] Muxing Video", total_dur)
        return output_path
    except (OSError, RuntimeError, ValueError) as e:
        log(f"Embedding failed: {e}", "ERROR")
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        return None


def _build_embed_command(video_path, srt_files, normalized_ext, output_path):
    """Build FFmpeg command for multi-language subtitle muxing."""
    cmd = [utils.FFMPEG_CMD, "-y", "-i", video_path]
    for track in srt_files:
        srt_path = track[0]
        cmd.extend(["-sub_charenc", "UTF-8", "-i", srt_path])

    cmd.extend(["-map", "0:v", "-map", "0:a"])
    for index in range(len(srt_files)):
        cmd.extend(["-map", f"{index + 1}"])

    subtitle_codec = "mov_text" if normalized_ext in [".mp4", ".m4v", ".mov"] else "srt"
    cmd.extend(["-c:v", "copy", "-c:a", "copy", "-c:s", subtitle_codec])
    cmd.extend(_build_embed_metadata_args(srt_files))
    cmd.extend(["-loglevel", "info", output_path])
    return cmd


def _build_embed_metadata_args(srt_files):
    """Build subtitle metadata arguments for language and track title."""
    metadata_args = []
    for index, track in enumerate(srt_files):
        _, lang, label, *optional = track
        mux_lang = optional[0] if optional else lang
        metadata_args.extend(
            [
                f"-metadata:s:s:{index}",
                f"language={mux_lang}",
                f"-metadata:s:s:{index}",
                f"title={label}",
            ]
        )
    return metadata_args


def _to_mux_language_code(lang):
    """Convert ISO language key into ISO 639-2/B-like 3-letter mux code."""
    lang_info = config.TARGET_LANGUAGES.get(lang)
    if isinstance(lang_info, dict) and lang_info.get("code"):
        nllb_code = lang_info["code"]
    else:
        nllb_code = config.ISO_TO_NLLB.get(lang, "eng_Latn")
    return nllb_code.split("_", maxsplit=1)[0] if nllb_code else lang


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
    generated_srts = _collect_generated_srt_tracks(folder, base_name, src_lang, src_srt_path)
    return embed_subtitles(video_path, generated_srts)


def _collect_generated_srt_tracks(folder, base_name, src_lang, src_srt_path):
    """Collect source and translated SRT tracks for final muxing."""
    generated_srts = []
    if os.path.exists(src_srt_path):
        src_label = config.TARGET_LANGUAGES.get(src_lang, {}).get("label", src_lang.upper())
        generated_srts.append((src_srt_path, src_lang, src_label, _to_mux_language_code(src_lang)))

    for lang, info in config.TARGET_LANGUAGES.items():
        track = _build_translation_srt_track(folder, base_name, src_lang, lang, info)
        if track is not None:
            generated_srts.append(track)

    return generated_srts


def _build_translation_srt_track(folder, base_name, src_lang, lang, info):
    """Build one translated SRT track tuple or return None when unavailable."""
    if lang == src_lang:
        return None

    lang_srt = os.path.join(folder, f"{base_name}.{lang}.srt")
    if not os.path.exists(lang_srt):
        return None
    if not utils.validate_srt(lang_srt):
        log(f"  [Mux] Skipping invalid translated SRT: {lang_srt}", "WARNING")
        return None

    label = info.get("label", lang.upper()) if isinstance(info, dict) else lang.upper()
    return (lang_srt, lang, label, _to_mux_language_code(lang))


def _build_transcription_context(folder, base_name, video_path):
    """Build context payload passed into transcription/resume helper."""
    return {
        "folder": folder,
        "base_name": base_name,
        "video_path": video_path,
    }


def _prepare_source_srt_path(folder, base_name, src_lang, source_artifact_path, segments):
    """Resolve source subtitle path from resumed artifact or persist newly generated SRT."""
    src_srt_path = os.path.join(folder, f"{base_name}.{src_lang}.srt")
    if _is_reused_srt_artifact(source_artifact_path):
        _write_recorded_source_language(folder, base_name, src_lang)
        log("  [Resume] Reusing existing subtitle file. Continuing translation and muxing.", "INFO")
        return source_artifact_path
    if _save_source_srt_file(segments, src_srt_path):
        _write_recorded_source_language(folder, base_name, src_lang)
        return src_srt_path
    return None


def _save_source_srt_file(segments, src_srt_path):
    """Persist source SRT and return True on success."""
    try:
        utils.save_srt(segments, src_srt_path)
        return True
    except (OSError, ValueError) as e:
        log(f"  [Error] Failed to save source SRT: {e}", "ERROR")
        return False


def _is_reused_srt_artifact(source_artifact_path):
    """Return True when the transcription step reused an existing SRT artifact."""
    return bool(source_artifact_path and source_artifact_path.endswith(".srt"))


def _clear_cuda_cache_if_available():
    """Proactively clear CUDA cache before heavy translation model loads."""
    gc.collect()
    torch_module = _get_torch_module()
    if torch_module is not None and hasattr(torch_module, "cuda") and torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()


def _run_translation_step(segments, src_lang, model_mgr, folder, base_name):
    """Run translation stage and return True when it completes successfully."""
    try:
        model_mgr.offload_whisper()
        model_mgr.offload_separator()
        _clear_cuda_cache_if_available()
        translate_segments(segments, src_lang, model_mgr, folder, base_name)
        return True
    except (RuntimeError, OSError, ValueError) as e:
        log(f"Translation failed: {e}", "ERROR")
        return False


def _process_video_pipeline(video_path, model_mgr, pipeline_context):
    """Run transcription, translation, and muxing stages for one video."""
    utils.init_console()
    forced_lang = pipeline_context["forced_lang"]
    forced_prompt = pipeline_context["forced_prompt"]
    folder = pipeline_context["folder"]
    base_name = pipeline_context["base_name"]

    transcription_context = _build_transcription_context(folder, base_name, video_path)
    segments, src_lang, source_artifact_path = _obtain_segments(
        transcription_context,
        model_mgr,
        forced_lang,
        forced_prompt,
    )

    if not segments:
        log("No speech detected.", "WARNING")
        return [], None, None

    src_srt_path = _prepare_source_srt_path(folder, base_name, src_lang, source_artifact_path, segments)
    if src_srt_path is None:
        return None, None, None

    if not _run_translation_step(segments, src_lang, model_mgr, folder, base_name):
        return None, None, None
    finalized_output_path = _finalize_video_processing(video_path, folder, base_name, src_lang, src_srt_path)
    if not finalized_output_path:
        return None, None, None
    return segments, src_lang, finalized_output_path


def process_video(video_path, model_mgr, forced_lang=None, forced_prompt=None):
    """Orchestrates the full processing pipeline for a single video."""
    config.load_config(OPTIMIZER, log)
    folder = os.path.dirname(video_path) or "."
    output_path, _source_srt_path, base_name = _get_output_filenames(video_path, folder, None)

    # Check if this video is already done
    if os.path.exists(output_path):
        log(f"  [Skip] Output already exists: {output_path}", "INFO")
        return None, None, output_path

    try:
        pipeline_context = {
            "forced_lang": forced_lang,
            "forced_prompt": forced_prompt,
            "folder": folder,
            "base_name": base_name,
            "output_path": output_path,
        }
        return _process_video_pipeline(video_path, model_mgr, pipeline_context)

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
