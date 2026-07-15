"""Transcription pipeline and vocal-separation orchestration."""

import gc
import math
import os
import sys
import time
from typing import Any

from modules import utils
from modules.configuration import config
from modules.models import OPTIMIZER
from modules.runtime.optional_imports import load_optional_torch
from modules.utils import log

torch: Any | None = load_optional_torch()


def _get_separated_vocal_path(video_path):
    """Internal helper to determine vocal separation output path."""
    target_dir = os.path.abspath(os.path.dirname(video_path))
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    separator_prefix = f"{base_name}_"
    # Audio-Separator naming: {base_name}_(Vocals)_...
    try:
        for f in os.listdir(target_dir):
            if f.startswith(separator_prefix) and "(Vocals)" in f:
                return os.path.join(target_dir, f)
    except OSError:
        pass
    return None


def _process_separator_outputs(output_files, target_dir):
    """Persist only vocal output files and ignore non-vocal stems."""
    vocal_file = None
    for output_file in output_files:
        src_path = _resolve_separator_output_path(output_file, target_dir)
        file_name = os.path.basename(output_file)
        if "Vocals" not in file_name:
            continue

        dst_path = os.path.join(target_dir, file_name)
        _move_separator_output(src_path, dst_path)
        vocal_file = dst_path
    return vocal_file


def _resolve_separator_output_path(output_file, target_dir):
    """Resolve separator output path for absolute, cwd-relative, and target-dir-relative files."""
    if os.path.isabs(output_file):
        return output_file

    abs_from_cwd = os.path.abspath(output_file)
    if os.path.exists(abs_from_cwd):
        return abs_from_cwd

    return os.path.join(target_dir, output_file)


def _move_separator_output(src_path, dst_path):
    """Move a separator output file into target directory, replacing stale outputs."""
    if not os.path.exists(src_path) or src_path == dst_path:
        return
    if os.path.exists(dst_path):
        os.remove(dst_path)
    os.rename(src_path, dst_path)


def _detect_and_separate_vocals(video_path, model_mgr):
    """Handles vocal separation if configured."""
    if not config.USE_VOCAL_SEPARATION:
        return video_path

    existing_vocal = _get_separated_vocal_path(video_path)
    if existing_vocal:
        log(f"  [AI] Resuming with existing vocals: {os.path.basename(existing_vocal)}")
        return existing_vocal

    try:
        vocal_file = _run_vocal_separation(video_path, model_mgr)
        if vocal_file:
            return vocal_file

    except (RuntimeError, OSError) as e:
        log(f"  [Sep] Warning: Separation failed ({e}). Using original audio.", "WARNING")

    return video_path


def _run_vocal_separation(video_path, model_mgr):
    """Execute separator model and return isolated vocals path when successful."""
    log("  [Task 0/4] Separating Vocals (BS-Roformer)...")
    audio_input_path = utils.extract_clean_audio(video_path)
    target_dir = os.path.abspath(os.path.dirname(video_path))
    separator = model_mgr.get_separator(output_dir=target_dir)
    output_files = separator.separate(audio_input_path)
    vocal_file = _process_separator_outputs(output_files, target_dir)
    if vocal_file and os.path.exists(vocal_file):
        log(f"  [Sep] Vocal track isolated: {os.path.basename(vocal_file)}")
        return vocal_file
    return None


def _filter_hallucinations(segments, hallucination_phrases):
    """Internal helper to filter out hallucinated segments."""
    filtered_segments = []
    hallucinated_count = 0

    for segment in segments:
        text_clean = segment.text.strip().lower().strip(".,!?;: ")
        if _is_hallucinated_text(text_clean, hallucination_phrases):
            hallucinated_count += 1
            continue
        filtered_segments.append(segment)

    return filtered_segments, hallucinated_count


def _is_hallucinated_text(text_clean, hallucination_phrases):
    """Return True when segment text matches known short hallucination patterns."""
    for phrase in hallucination_phrases:
        if phrase in text_clean and len(text_clean) < len(phrase) + 5:
            return True
    return False


def _process_transcription_segments(segments_gen, total_dur, start_time):
    """Internal helper to process segments and update progress."""
    segments = []
    for segment in segments_gen:
        segments.append(segment)
        elapsed = time.time() - start_time
        # Calculate speed (audio seconds per real second)
        speed = segment.end / elapsed if elapsed > 0 else 0
        eta = (total_dur - segment.end) / speed if speed > 0 and total_dur > segment.end else 0

        _print_transcription_segment(segment)
        _print_transcription_progress(segment.end, total_dur, speed, eta)
    return segments


def _print_transcription_segment(segment):
    """Print one transcription segment with timestamp and confidence."""
    sys.stdout.write("\r\033[K")
    prob = math.exp(segment.avg_logprob) if hasattr(segment, "avg_logprob") else 1.0
    ts_start = utils.format_timestamp(segment.start)
    ts_end = utils.format_timestamp(segment.end)
    print(f"[{ts_start}->{ts_end}] ({prob:.0%}) {segment.text.strip()}")


def _print_transcription_progress(current_end, total_dur, speed, eta):
    """Print transcription progress bar with timing context."""
    utils.print_progress_bar(
        current_end,
        total_dur,
        prefix="  [Whisper] Transcribing",
        timestamp_str=f"{utils.format_timestamp(current_end)} / {utils.format_timestamp(total_dur)}",
        speed=speed,
        eta=eta,
    )


def _prepare_audio(video_path, model_mgr):
    """Prepares audio for transcription (Separation or Extraction)."""
    transcribe_path = video_path
    if config.USE_VOCAL_SEPARATION:
        transcribe_path = _detect_and_separate_vocals(video_path, model_mgr)
        model_mgr.offload_separator()

    if transcribe_path == video_path:
        transcribe_path = utils.extract_clean_audio(video_path)
    return transcribe_path


def _run_whisper_transcribe_call(whisper_model, path, vad_params, transcription_options):
    """Execute one Whisper transcribe call for a specific beam size."""
    return whisper_model.transcribe(
        path,
        beam_size=transcription_options["beam_size"],
        initial_prompt=transcription_options["prompt"],
        vad_filter=True,
        vad_parameters=vad_params,
        language=transcription_options["language"],
        condition_on_previous_text=True,
        no_speech_threshold=0.6,
    )


def _perform_transcription(whisper_model, path, vad_params, transcription_options, start_time):
    """Execute Whisper transcription and consume segments with OOM retry support."""
    initial_beam = int(transcription_options["beam_size"])
    try:
        first_pass_options = {
            **transcription_options,
            "beam_size": initial_beam,
        }
        segments_gen, info = _run_whisper_transcribe_call(whisper_model, path, vad_params, first_pass_options)
        segments = _process_transcription_segments(segments_gen, info.duration, start_time)
        return segments, info
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            reduced_beam = max(1, initial_beam // 2)
            if reduced_beam >= initial_beam:
                raise
            log("  [Whisper] OOM detected. Clearing cache and retrying...", "WARNING")
            if torch is not None:
                torch.cuda.empty_cache()
            gc.collect()
            time.sleep(1)
            retry_options = {
                **transcription_options,
                "beam_size": reduced_beam,
            }
            segments_gen, info = _run_whisper_transcribe_call(whisper_model, path, vad_params, retry_options)
            segments = _process_transcription_segments(segments_gen, info.duration, start_time)
            return segments, info
        raise


def _log_transcription_config(lang_to_use, current_prompt):
    """Log the language and prompt configuration used for transcription."""
    if lang_to_use:
        log(f"  [Whisper] Config: Forced Language='{lang_to_use}'")
    else:
        log("  [Whisper] Config: Language Auto-Detection Enabled")

    if current_prompt:
        log("  [Whisper] Config: Input Prompt=Enabled")
    else:
        log("  [Whisper] Config: No Input Prompt")


def _finalize_transcription(segments, info, start_time):
    """Finalize Whisper output, filter segments, and free model resources."""
    elapsed = time.time() - start_time
    utils.print_progress_bar(
        info.duration,
        info.duration,
        prefix="  [Whisper] Transcribing",
        elapsed=elapsed,
        speed=info.duration / elapsed if elapsed > 0 else 1.0,
    )

    filtered_segments, hallucinated_count = _filter_hallucinations(
        segments,
        config.HALLUCINATION_PHRASES,
    )
    if hallucinated_count > 0:
        log(f"  [Whisper] Filtered {hallucinated_count} hallucinated segments.", "WARNING")

    detected_lang = info.language
    probability = info.language_probability
    log(f"  [Whisper] Detected Language: {detected_lang} (Conf: {probability:.2f})")
    if probability < 0.4:
        log(f"  [Warning] Low language confidence ({probability:.2f}).", "WARNING")

    filtered_segments.sort(key=lambda segment: segment.start)
    return filtered_segments, detected_lang


def transcribe_video_audio(video_path, model_mgr, forced_lang=None, forced_prompt=None):
    """Runs Whisper transcription on the video (or vocal track)."""
    # 1. Prepare Audio
    transcribe_path = _prepare_audio(video_path, model_mgr)

    # 2. Transcribe
    log(f"  [Task 1/4] Transcribing '{os.path.basename(transcribe_path)}'...")
    current_prompt = forced_prompt if forced_prompt else config.INITIAL_PROMPT
    lang_to_use = forced_lang if forced_lang else config.FORCED_LANGUAGE

    _log_transcription_config(lang_to_use, current_prompt)

    start_time = time.time()

    try:
        whisper_model = model_mgr.get_whisper()
        vad_params = {
            "threshold": 0.35,
            "min_silence_duration_ms": config.VAD_MIN_SILENCE_MS,
            "speech_pad_ms": 500,
        }
        transcription_options = {
            "prompt": current_prompt,
            "language": lang_to_use,
            "beam_size": int(OPTIMIZER.config["whisper_beam"]),
        }

        segments, info = _perform_transcription(
            whisper_model,
            transcribe_path,
            vad_params,
            transcription_options,
            start_time,
        )

        segments, detected_lang = _finalize_transcription(
            segments,
            info,
            start_time,
        )
        return segments, detected_lang, transcribe_path

    except (RuntimeError, OSError, ValueError) as e:
        log(f"Transcription failed: {e}", "ERROR")
        raise
    finally:
        model_mgr.offload_whisper()
