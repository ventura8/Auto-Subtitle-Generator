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

import os
import sys

# Ensure the root directory is in sys.path for internal module imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import multiprocessing  # noqa: E402
import argparse  # noqa: E402
import gc  # noqa: E402
import logging  # noqa: E402

# New Modules
from modules import config  # noqa: E402
from modules import utils  # noqa: E402
from modules.utils import log, print_progress_bar  # noqa: E402
from modules import models  # noqa: E402
from modules.models import OPTIMIZER, ModelManager  # noqa: E402
from modules.transcription import transcribe_video_audio  # noqa: E402
from modules.translation import translate_segments  # noqa: E402


# Lazy imports handle
torch = None

logging.getLogger("transformers").addFilter(
    lambda record: "tied weights" not in record.getMessage()
)


# =============================================================================
# AI ENGINE INITIALIZATION
# =============================================================================


def _init_torch_and_hardware(step, total_steps):
    """Initializes PyTorch and hardware detection."""
    # Step 1: PyTorch
    try:
        global torch
        import torch as _torch
        if torch is None:
            torch = _torch
    except ImportError as e:
        suffix_fail = f"{'Loading PyTorch':<35} [FAIL]"
        print_progress_bar(step, total_steps, prefix="[Init] ", suffix=suffix_fail, length=25, decimals=0)
        print("")
        log(f"[Fatal] PyTorch missing: {e}", "CRITICAL")
        sys.exit(1)

    # Step 2: Hardware Detection
    step += 1
    OPTIMIZER.detect_hardware(verbose=False)
    return step


def _init_nvidia_and_transformers(step, total_steps):
    """Initializes NVIDIA paths and Transformers."""
    # Step 3: NVIDIA Paths
    step += 1
    load_nvidia_paths()

    # Step 4: Transformers
    step += 1
    try:
        models.AutoTokenizer  # Trigger lazy load validation if we wanted
        # Verification import
        import transformers  # noqa: F401
    except ImportError:
        suffix_tr_fail = f"{'Loading Transformers (NLLB)':<35} [FAIL]"
        print_progress_bar(step, total_steps, prefix="[Init] ", suffix=suffix_tr_fail, length=25, decimals=0)
        print("")
        log("[Fatal] Transformers missing.", "CRITICAL")
        sys.exit(1)
    return step


def _init_whisper_and_separator(step, total_steps):
    """Initializes Faster-Whisper and Audio-Separator."""
    # Step 5: Faster-Whisper
    step += 1
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        suffix_wh_fail = f"{'Loading Faster-Whisper':<35} [FAIL]"
        print_progress_bar(step, total_steps, prefix="[Init] ", suffix=suffix_wh_fail, length=25, decimals=0)
        print("")
        log("[Fatal] Faster-Whisper missing.", "CRITICAL")
        sys.exit(1)

    # Step 6: Audio-Separator
    step += 1
    try:
        import audio_separator.separator  # noqa: F401
    except ImportError:
        suffix_sep_skip = f"{'Loading Audio-Separator':<35} [SKIP]"
        print_progress_bar(step, total_steps, prefix="[Init] ", suffix=suffix_sep_skip, length=25, decimals=0)
        log("[Warning] audio-separator not installed. Vocal separation will be skipped.", "WARNING")


def init_ai_engine():
    """Lazily loads all AI dependencies with a progress indicator."""
    if torch is not None:
        return

    print("[AI ENGINE INITIALIZATION]")

    total_steps = 6
    step = 1

    step = _init_torch_and_hardware(step, total_steps)
    step = _init_nvidia_and_transformers(step, total_steps)
    _init_whisper_and_separator(step, total_steps)


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
    import os
    for p in paths:
        if p not in os.environ['PATH']:
            os.environ['PATH'] = p + os.pathsep + os.environ['PATH']
            if hasattr(os, 'add_dll_directory'):
                try:
                    os.add_dll_directory(p)
                except (AttributeError, OSError):
                    pass


def load_nvidia_paths():
    """Adds Torch/NVIDIA DLLs to PATH to fix ONNX Runtime 'CUDAExecutionProvider not available'."""
    import site
    import os

    paths_to_add = []

    # 1. Site Packages
    site_packages = site.getsitepackages()
    manual_site = os.path.join(sys.prefix, "Lib", "site-packages")
    if manual_site not in site_packages:
        site_packages.append(manual_site)

    for sp in site_packages:
        paths_to_add.extend(_get_nvidia_bin_lib_paths(sp))

    # 2. Torch libs
    try:
        import torch
        if hasattr(torch, '__path__'):
            for q in torch.__path__:
                lib_path = os.path.join(q, "lib")
                if os.path.exists(lib_path):
                    paths_to_add.append(lib_path)
    except ImportError:
        pass

    # 3. Apply
    _apply_paths_to_env(paths_to_add)

    try:
        import onnxruntime as ort  # noqa: F401
    except Exception:
        pass


# =============================================================================
# PIPELINE FUNCTIONS
# =============================================================================


def _check_resume(folder, base_name, video_path, forced_lang=None):
    """Checks if a valid SRT exists to skip transcription."""
    if forced_lang:
        srt_path = os.path.join(folder, f"{base_name}.{forced_lang}.srt")
        if os.path.exists(srt_path):
            segs = utils.parse_srt(srt_path)
            if segs:
                log(f"  [Resume] Found valid SRT: {srt_path}")
                return segs, forced_lang, None
            else:
                log(f"  [Resume] SRT {srt_path} is empty or corrupted. Skipping.", "WARNING")
    else:
        # Check commonly generated ones
        for lang_code in ["en", "ro", "es", "fr"]:
            srt_path = os.path.join(folder, f"{base_name}.{lang_code}.srt")
            if os.path.exists(srt_path):
                segs = utils.parse_srt(srt_path)
                if segs:
                    log(f"  [Resume] Found valid SRT: {srt_path}")
                    return segs, lang_code, None
                else:
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
    output_path = os.path.join(dir_name, f"{name_no_ext}_multilang{ext}")

    cmd = [utils.FFMPEG_CMD, "-y", "-i", video_path]

    for srt, _, _ in srt_files:
        cmd.extend(["-sub_charenc", "UTF-8", "-i", srt])

    cmd.extend(["-map", "0:v", "-map", "0:a"])
    for i in range(len(srt_files)):
        cmd.extend(["-map", f"{i + 1}"])

    cmd.extend(["-c:v", "copy", "-c:a", "copy", "-c:s",
                "mov_text" if ext in [".mp4", ".m4v", ".mov"] else "srt"])

    for i, (_, lang, label) in enumerate(srt_files):
        cmd.extend([
            f"-metadata:s:s:{i}", f"language={lang}",
            f"-metadata:s:s:{i}", f"title={label}"
        ])

    cmd.extend(["-loglevel", "info", output_path])

    try:
        total_dur = utils.get_audio_duration(video_path)
        utils.run_ffmpeg_progress(cmd, "  [Finalizing] Muxing Video", total_dur)
    except Exception as e:
        log(f"Embedding failed: {e}", "ERROR")
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except Exception:
                pass


def _obtain_segments(folder, base_name, video_path, model_mgr, forced_lang, forced_prompt):
    """Internal helper to either load existing SRT or run transcription."""
    lang_hint = forced_lang if forced_lang else config.FORCED_LANGUAGE
    check_lang_code = lang_hint if lang_hint else None

    # Try to find existing output
    loaded_segments, loaded_lang, _ = _check_resume(folder, base_name, video_path, check_lang_code)

    if loaded_segments:
        log(f"  [Step 1] Skipping Transcription. Found valid SRT for {loaded_lang}.")
        return loaded_segments, loaded_lang, None

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
    for lang in config.TARGET_LANGUAGES:
        if lang == src_lang:
            continue
        lang_srt = os.path.join(folder, f"{base_name}.{lang}.srt")
        if os.path.exists(lang_srt):
            info = config.TARGET_LANGUAGES[lang]
            generated_srts.append((lang_srt, lang, info["label"]))

    embed_subtitles(video_path, generated_srts)


def process_video(video_path, model_mgr, forced_lang=None, forced_prompt=None):
    """Orchestrates the full processing pipeline for a single video."""
    config.load_config(OPTIMIZER, log)
    folder = os.path.dirname(video_path) or "."
    final_output, srt_path, base_name = _get_output_filenames(
        video_path, folder, None
    )

    # Check if this video is already done
    if os.path.exists(final_output):
        log(f"  [Skip] Output already exists: {final_output}", "INFO")
        return None, None, final_output

    try:
        utils.init_console()
        # Step 1: Transcribe (or Resume)
        segments, src_lang, audio_path = _obtain_segments(
            folder, base_name, video_path, model_mgr, forced_lang, forced_prompt
        )

        if not segments:
            log("No speech detected.", "WARNING")
            utils.cleanup_temp_files(folder, base_name, os.path.basename(video_path))
            # Fix: Return 3 values as expected by callers
            return [], None, None

        # Immediate Save: Source SRT
        # ... (rest of function)
        src_srt_path = os.path.join(folder, f"{base_name}.{src_lang}.srt")
        try:
            utils.save_srt(segments, src_srt_path)
        except Exception as e:
            log(f"  [Error] Failed to save source SRT: {e}", "ERROR")

        # Step 2: Translate
        try:
            # Proactively offload transcription/separation models to clear VRAM
            model_mgr.offload_whisper()
            model_mgr.offload_separator()

            # FORCE CLEAN STATE (Paranoid Mode)
            gc.collect()
            if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Refactored: Calls module function
            translate_segments(segments, src_lang, model_mgr, folder, base_name)
        except Exception as e:
            log(f"Translation failed: {e}", "ERROR")
            # Continue to finalizing even if translation fails
            pass

        # Step 3: Finalize (Embed Subtitles)
        _finalize_video_processing(video_path, folder, base_name, src_lang, src_srt_path)

        # Return results for callers (e.g. tests)
        return segments, src_lang, final_output

    except Exception as e:
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

    path = args.input_path
    if not path:
        print(">> Please Drag & Drop a video file here and press Enter:")
        path = input(">>Path: ").strip().strip('"')

    if not path:
        path = "input"  # Default folder

    files = []
    if os.path.isfile(path):
        files.append(os.path.abspath(path))
    elif os.path.isdir(path):
        for root, _, filenames in os.walk(path):
            for f in filenames:
                if f.lower().endswith(('.mp4', '.mkv', '.avi', '.mov', '.flv', '.webm')):
                    # Exclude our own output files
                    if "_multilang" in f:
                        continue
                    files.append(os.path.abspath(os.path.join(root, f)))
    else:
        log(f"Error: Path not found: {path}", "CRITICAL")
        sys.exit(1)

    return files, args.lang, args.prompt


def setup_environment():
    """Global setup for multiprocessing and signals."""
    multiprocessing.freeze_support()
    utils.init_console()
    utils.setup_signal_handlers()


def main():
    setup_environment()
    init_ai_engine()
    utils.print_banner(models.OPTIMIZER)

    video_files, forced_lang, forced_prompt = get_input_files()

    if not video_files:
        log("No video files found.", "WARNING")
        sys.exit(0)

    log(f"Found {len(video_files)} videos to process.", "INFO")

    model_mgr = ModelManager()

    for i, video_path in enumerate(video_files):
        print(f"\n[{i + 1}/{len(video_files)}] Processing: {video_path}")
        process_video(video_path, model_mgr, forced_lang, forced_prompt)

    print("\n[Done] All tasks completed.")


if __name__ == "__main__":
    main()
