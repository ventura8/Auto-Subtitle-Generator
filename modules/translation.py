"""Translation orchestration, pivot pass, and isolated worker coordination."""

import os
import sys
import gc
import subprocess
import json
import time
from modules import config, utils
from modules.utils import log

try:
    import torch
except ImportError:
    torch = None


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

            log(f"  [Translate] Found invalid/corrupt SRT for {lang}. Re-doing.", "WARNING")

        missing_langs.append(lang)

    # User Request: Show summary at start
    log(f"  [Translate] Summary: {len(all_targets)} Total | {len(missing_langs)} To Do | {skipped_count} Skipped", "INFO")
    return missing_langs, skipped_count


def _prepare_source_data(segments):
    """Filters valid segments and prepares data structure for translation."""
    valid_segments = [s for s in segments if s.text.strip()]
    return [{"text": s.text.strip(), "start": s.start, "end": s.end} for s in valid_segments]


def _process_completed_output(output_file, lang, segments, folder, base_name):
    """Helper to process a single completed translation file."""
    try:
        # Retry logic for reading
        for _ in range(3):
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    translated_lines = json.load(f)
                break
            except (OSError, json.JSONDecodeError):
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

        log(f"  [Error] Mismatch for {lang}: {len(translated_lines)} vs {len(segments)}", "ERROR")
        return False

    except (OSError, RuntimeError, json.JSONDecodeError) as e:
        log(f"  [Error] Failed to process realtime output for {lang}: {e}", "ERROR")
        return False


def _wait_worker_tick(proc):
    """Return True when worker has exited; False when still running after timeout."""
    timeout_expired = getattr(subprocess, "TimeoutExpired", TimeoutError)
    if not (isinstance(timeout_expired, type) and issubclass(timeout_expired, BaseException)):
        timeout_expired = TimeoutError

    try:
        proc.wait(timeout=0.1)
        return True
    except (TimeoutError, timeout_expired):
        return False


def _scan_pending_outputs(pending, folder, base_name, segments):
    """Process currently available worker outputs and return remaining pending languages."""
    for lang in list(pending):
        output_file = os.path.join(folder, f".temp_output.{base_name}.{lang}.json")

        if not os.path.exists(output_file):
            continue

        success = _process_completed_output(output_file, lang, segments, folder, base_name)
        pending.remove(lang)
        if success:
            if os.path.exists(output_file):
                os.remove(output_file)

    return pending


def _poll_translation_results(proc, missing_langs, folder, base_name, segments):
    """Helper to poll for real-time translation results."""
    # Convert list to set for O(1) lookups
    pending = set(missing_langs)

    while proc.poll() is None:
        if _wait_worker_tick(proc):
            break

        pending = _scan_pending_outputs(pending, folder, base_name, segments)

        time.sleep(0.1)

    # FINAL PASS: Check one last time for any remaining files after process exit
    for lang in list(pending):
        output_file = os.path.join(folder, f".temp_output.{base_name}.{lang}.json")
        if os.path.exists(output_file):
            _process_completed_output(output_file, lang, segments, folder, base_name)


def _safe_remove(file_path):
    """Remove a temporary file when it exists."""
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except OSError:
            pass


def _build_pivot_config(worker_context, common_input, temp_files):
    """Build optional pivot job metadata for non-English sources."""
    if worker_context["src_lang"] == "en":
        return None, worker_context["src_code"]

    folder = worker_context["folder"]
    base_name = worker_context["base_name"]
    missing_langs = worker_context["missing_langs"]

    pivot_src_code = worker_context["src_code"]
    pivot_tgt_code = config.TARGET_LANGUAGES.get("en", {"code": "eng_Latn"})["code"]
    pivot_output = os.path.join(folder, f"{base_name}.pivot_pivoted.json")
    temp_files.append(pivot_output)

    pivot_config = {
        "input": common_input,
        "output": pivot_output,
        "src_code": pivot_src_code,
        "tgt_code": pivot_tgt_code,
        "emit_en_output": "en" in missing_langs,
    }

    if pivot_config["emit_en_output"]:
        en_output_file = os.path.join(folder, f".temp_output.{base_name}.en.json")
        temp_files.append(en_output_file)
        pivot_config["en_output"] = en_output_file
        _safe_remove(en_output_file)

    return pivot_config, pivot_tgt_code


def _build_manifest_jobs(worker_context, source_code_for_jobs, common_input, pivot_config, temp_files):
    """Build translation jobs for all target languages still pending."""
    jobs = []
    source_lang_for_jobs = config.nllb_to_iso(source_code_for_jobs)
    input_file = pivot_config["output"] if pivot_config else common_input

    for lang in worker_context["missing_langs"]:
        if pivot_config and lang == "en":
            continue

        output_file = os.path.join(worker_context["folder"], f".temp_output.{worker_context['base_name']}.{lang}.json")
        temp_files.append(output_file)
        _safe_remove(output_file)

        lang_info = config.TARGET_LANGUAGES[lang]
        jobs.append(
            {
                "lang": lang,
                "label": lang_info.get("label", lang),
                "tgt_code": lang_info["code"],
                "src_code": source_code_for_jobs,
                "src_lang": source_lang_for_jobs,
                "input": input_file,
                "output": output_file,
            }
        )

    return jobs


def _create_translation_manifest(worker_context):
    """Creates the job manifest and input files for the worker."""
    folder = worker_context["folder"]
    base_name = worker_context["base_name"]
    common_input = os.path.join(folder, f"{base_name}.common_input.json")
    temp_files = [common_input]

    with open(common_input, "w", encoding="utf-8") as file_handle:
        json.dump(worker_context["source_data"], file_handle, ensure_ascii=False)

    pivot_config, source_code_for_jobs = _build_pivot_config(worker_context, common_input, temp_files)
    manifest_jobs = _build_manifest_jobs(
        worker_context,
        source_code_for_jobs,
        common_input,
        pivot_config,
        temp_files,
    )

    manifest_path = os.path.join(folder, f"{base_name}.manifest.json")
    temp_files.append(manifest_path)

    with open(manifest_path, "w", encoding="utf-8") as file_handle:
        json.dump({"jobs": manifest_jobs, "pivot": pivot_config}, file_handle, ensure_ascii=False, indent=2)

    return manifest_path, temp_files


def _cleanup_worker_process(proc):
    """Safely terminates the worker process."""
    try:
        if proc.poll() is None:
            log("!   [Cleanup] Terminating orphaned translation worker...", "WARNING")
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except OSError:
                pass
    finally:
        utils.unregister_subprocess(proc)


def _cleanup_temp_files(temp_files):
    """Removes temporary files."""
    for tf in temp_files:
        if os.path.exists(tf):
            try:
                os.remove(tf)
            except OSError:
                pass


def _run_worker_process(worker_context):
    """Spawns and manages the isolated translation worker process."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    worker_path = os.path.join(project_root, "modules", "isolated_translator.py")
    cmd = [sys.executable, worker_path, "--batch", worker_context["manifest_path"]]

    env = os.environ.copy()
    env["IS_SUBPROCESS"] = "1"
    # Ensure subprocess can import modules by adding project root to PYTHONPATH
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = project_root if not existing_pythonpath else project_root + os.pathsep + existing_pythonpath
    with subprocess.Popen(cmd, env=env) as proc:
        utils.register_subprocess(proc)

        try:
            _poll_translation_results(
                proc,
                worker_context["missing_langs"],
                worker_context["folder"],
                worker_context["base_name"],
                worker_context["segments"],
            )
            proc.wait()

            if proc.returncode != 0:
                log(f"!!! Translation worker failed with code {proc.returncode}", "ERROR")

        finally:
            _cleanup_worker_process(proc)
            # Ensure temporary files are cleaned regardless of worker outcome
            try:
                _cleanup_temp_files(worker_context["temp_files"])
            except OSError:
                pass

    gc.collect()
    if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _execute_translation_workers(*worker_args):
    """Orchestrates the worker processes for each missing language."""
    if len(worker_args) == 1 and isinstance(worker_args[0], dict):
        worker_context = worker_args[0]
    else:
        worker_context = {
            "missing_langs": worker_args[0],
            "source_data": worker_args[1],
            "src_code": worker_args[2],
            "src_lang": config.nllb_to_iso(worker_args[2]),
            "folder": worker_args[3],
            "base_name": worker_args[4],
            "segments": worker_args[5],
        }

    log(f"  [System] Offloading remaining {len(worker_context['missing_langs'])} targets to Batch Worker...")

    manifest_path, temp_files = _create_translation_manifest(
        {
            **worker_context,
            "src_lang": worker_context.get("src_lang", config.nllb_to_iso(worker_context["src_code"])),
        }
    )
    worker_context = {
        **worker_context,
        "manifest_path": manifest_path,
        "temp_files": temp_files,
    }
    _run_worker_process(worker_context)


def translate_segments(segments, src_lang, model_mgr, folder, base_name):
    """Translates transcription segments into missing target languages."""
    missing_langs, _skipped_count = _identify_missing_targets(src_lang, folder, base_name)

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

    _execute_translation_workers(
        {
            "missing_langs": missing_langs,
            "source_data": source_data,
            "src_code": src_code,
            "src_lang": src_lang,
            "folder": folder,
            "base_name": base_name,
            "segments": segments,
        }
    )
    return {}
