import os
import sys
import gc
import subprocess
import json
import time
from modules import config, utils, models
from modules.utils import log

# Lazy imports handle
torch = None


def _handle_pivot_pass(source_data, src_lang, folder, base_name, missing_langs, src_code, segments):
    """Internal helper to handle the English pivot strategy."""
    log(f"  [Pivot] Non-English source detected ({src_lang}). Performing English Pivot Pass...")

    eng_info = config.TARGET_LANGUAGES.get("en", {"code": "eng_Latn", "label": "English"})
    pivot_tgt_code = eng_info["code"]

    # Temp files for pivot
    temp_input = os.path.join(folder, f"{base_name}.pivot_input.json")
    temp_output = os.path.join(folder, f"{base_name}.pivot_output.json")

    with open(temp_input, 'w', encoding='utf-8') as f:
        json.dump(source_data, f, ensure_ascii=False)

    worker_batch = models.OPTIMIZER.config["nllb_batch"]

    cmd = [
        sys.executable,
        os.path.join("modules", "isolated_translator.py"),
        temp_input, temp_output,
        src_code, pivot_tgt_code,
        str(worker_batch),
        "English"
    ]

    try:
        # SAFETY: Popen for cleanup
        proc = subprocess.Popen(cmd)
        utils.register_subprocess(proc)
        proc.wait()

        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)

        utils.unregister_subprocess(proc)
        with open(temp_output, 'r', encoding='utf-8') as f:
            pivot_lines = json.load(f)

        # Switch source to the Pivot (English)
        new_source_data = [
            {"text": txt, "start": item["start"], "end": item["end"]}
            for txt, item in zip(pivot_lines, source_data)
        ]

        # If English was a requested target, save it now
        if "en" in missing_langs:
            lang_srt_path = os.path.join(folder, f"{base_name}.en.srt")
            utils.save_translated_srt(segments, pivot_lines, lang_srt_path)
            missing_langs.remove("en")
            log("  [Pivot] English target satisfied via pivot pass.")

        return new_source_data, pivot_tgt_code

    except Exception as e:
        log(f"  [Pivot] Warning: Pivot pass failed ({e}). Proceeding with direct translation.", "WARNING")
        return source_data, src_code

    finally:
        for f_path in [temp_input, temp_output]:
            if os.path.exists(f_path):
                try:
                    os.remove(f_path)
                except OSError as e:
                    log(f"  [Warning] Could not remove temp file {temp_input}: {e}", "DEBUG")
                    pass


def _identify_missing_targets(src_lang, folder, base_name):
    """Identifies which languages still need translation."""
    all_targets = [lang for lang in config.TARGET_LANGUAGES if lang != src_lang]
    if not all_targets:
        return [], 0

    missing_langs = []
    skipped_count = 0
    for lang in all_targets:
        lang_srt_path = os.path.join(folder, f"{base_name}.{lang}.srt")

        # SKIP if valid
        if os.path.exists(lang_srt_path):
            if utils.validate_srt(lang_srt_path):
                skipped_count += 1
                continue
            else:
                log(f"  [Translate] Found invalid/corrupt SRT for {lang}. Re-doing.", "WARNING")

        missing_langs.append(lang)

    # User Request: Show summary at start
    log(f"  [Translate] Summary: {len(all_targets)} Total | {len(missing_langs)} To Do | {skipped_count} Skipped", "INFO")
    return missing_langs, skipped_count


def _prepare_source_data(segments):
    """Filters valid segments and prepares data structure for translation."""
    valid_segments = [s for s in segments if s.text.strip()]
    return [
        {"text": s.text.strip(), "start": s.start, "end": s.end}
        for s in valid_segments
    ]


def _process_completed_output(output_file, lang, segments, folder, base_name):
    """Helper to process a single completed translation file."""
    try:
        # Retry logic for reading
        for _ in range(3):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    translated_lines = json.load(f)
                break
            except Exception:
                time.sleep(0.1)
        else:
            raise RuntimeError(f"Could not read {output_file}")

        # Save SRT
        if len(translated_lines) == len(segments):
            lang_srt_path = os.path.join(folder, f"{base_name}.{lang}.srt")
            utils.save_translated_srt(segments, translated_lines, lang_srt_path)
            log(f"  [Success] Saved {lang} subtitles.")

            # Success - return True to indicate completion
            return True
        else:
            log(f"  [Error] Mismatch for {lang}: {len(translated_lines)} vs {len(segments)}", "ERROR")
            return False

    except Exception as e:
        log(f"  [Error] Failed to process realtime output for {lang}: {e}", "ERROR")
        return False


def _poll_translation_results(proc, missing_langs, folder, base_name, segments):
    """Helper to poll for real-time translation results."""
    # Convert list to set for O(1) lookups
    pending = set(missing_langs)

    while proc.poll() is None:
        # Check output files for all pending languages
        for lang in list(pending):
            output_file = os.path.join(folder, f".temp_output.{base_name}.{lang}.json")

            if os.path.exists(output_file):
                # Process It
                success = _process_completed_output(output_file, lang, segments, folder, base_name)

                # If processed (success or fail-mismatch), we clean up
                # Note: Currently mismatched also cleans up to unblock
                if success:
                    pending.remove(lang)

                # Cleanup file
                if os.path.exists(output_file):
                    os.remove(output_file)

        time.sleep(0.1)

    # FINAL PASS: Check one last time for any remaining files after process exit
    for lang in list(pending):
        output_file = os.path.join(folder, f".temp_output.{base_name}.{lang}.json")
        if os.path.exists(output_file):
            _process_completed_output(output_file, lang, segments, folder, base_name)


def _execute_translation_workers(missing_langs, source_data, src_code, folder, base_name, segments):
    """Orchestrates the worker processes for each missing language."""
    log(f"  [System] Offloading remaining {len(missing_langs)} targets to Batch Worker...")

    # 1. Create Manifest
    from modules import config

    manifest_jobs = []

    # Pre-write common input file
    common_input = os.path.join(folder, f"{base_name}.common_input.json")
    # Track temp files for cleanup
    temp_files = [common_input]

    with open(common_input, 'w', encoding='utf-8') as f:
        json.dump(source_data, f, ensure_ascii=False)

    for lang in missing_langs:
        info = config.TARGET_LANGUAGES[lang]
        output_file = os.path.join(folder, f".temp_output.{base_name}.{lang}.json")
        temp_files.append(output_file)

        # Ensure cleanup of old pending files
        if os.path.exists(output_file):
            os.remove(output_file)

        manifest_jobs.append({
            "lang": lang,
            "label": info.get("label", lang),
            "tgt_code": info["code"],
            "src_code": src_code,
            "input": common_input,
            "output": output_file
        })

    manifest_path = os.path.join(folder, f"{base_name}.manifest.json")
    temp_files.append(manifest_path)

    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump({"jobs": manifest_jobs}, f, ensure_ascii=False, indent=2)

    # 2. Spawn Isolated Worker (Batch Mode)
    cmd = [
        sys.executable,
        os.path.join("modules", "isolated_translator.py"),
        "--batch",
        manifest_path
    ]

    proc = subprocess.Popen(cmd)

    # Register for cleanup
    utils.register_subprocess(proc)

    try:
        # 3. Poll for Results (Real-time)
        _poll_translation_results(proc, missing_langs, folder, base_name, segments)

        # Final check after process exit
        proc.wait()

        if proc.returncode != 0:
            log(f"!!! Translation worker failed with code {proc.returncode}", "ERROR")

    finally:
        # Cleanup Worker
        if proc.poll() is None:
            log("!   [Cleanup] Terminating orphaned translation worker...", "WARNING")
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception:
                pass
            utils.unregister_subprocess(proc)

        # Cleanup Files
        for tf in temp_files:
            if os.path.exists(tf):
                try:
                    os.remove(tf)
                except OSError:
                    pass

    # GC/Cache clear
    gc.collect()
    if 'torch' in globals() and torch is not None:
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


def translate_segments(segments, src_lang, model_mgr, folder, base_name):
    """Translates transcription segments into missing target languages."""
    global torch
    if torch is None:
        try:
            import torch
        except ImportError:
            torch = None

    missing_langs, skipped_count = _identify_missing_targets(src_lang, folder, base_name)

    if not missing_langs:
        log("  [Skip] All targets completed. Moving to next step.")
        return {}

    # CRITICAL: Offload previous models to prevent VRAM Contention/Shared Memory usage
    # The isolated worker will load NLLB, so we need to clear space in the main process first.
    if model_mgr:
        log("  [System] Offloading Whisper/Separator to free VRAM for Translation Worker...", level="DEBUG")
        model_mgr.offload_whisper()
        model_mgr.offload_separator()

    # PREPARE DATA FOR ISOLATION
    src_code = config.get_nllb_code(src_lang)
    source_data = _prepare_source_data(segments)

    if not source_data:
        log("  [Skip] No valid text to translate.")
        return {}

    # PIVOT STRATEGY: (Non-English Source -> English -> Others)
    if src_lang != "en" and missing_langs:
        source_data, src_code = _handle_pivot_pass(
            source_data, src_lang, folder, base_name, missing_langs, src_code, segments
        )

    # PROCESS REMAINING LANGUAGES
    if not missing_langs:
        return {}

    _execute_translation_workers(missing_langs, source_data, src_code, folder, base_name, segments)
