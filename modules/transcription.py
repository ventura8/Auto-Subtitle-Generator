"""Transcription pipeline and vocal-separation orchestration."""

import gc
import math
import os
import sys
import time

try:
    import torch
except ImportError:
    torch = None

from modules import config, utils
from modules.utils import log
from modules.models import OPTIMIZER


def _get_separated_vocal_path(video_path):
    """Internal helper to determine vocal separation output path."""
    target_dir = os.path.abspath(os.path.dirname(video_path))
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    # Audio-Separator naming: {base_name}_(Vocals)_...
    try:
        for f in os.listdir(target_dir):
            if f.startswith(base_name) and "(Vocals)" in f:
                return os.path.join(target_dir, f)
    except OSError:
        pass
    return None


def _process_separator_outputs(output_files, target_dir):
    """Handles renaming and moving separator output files."""
    vocal_file = None
    for f in output_files:
        src_path = os.path.abspath(f)
        base = os.path.basename(f)

        # Rename Instrumental -> Background
        if "(Instrumental)" in base:
            base = base.replace("(Instrumental)", "(Background)")

        dst_path = os.path.join(target_dir, base)
        if os.path.exists(src_path) and src_path != dst_path:
            if os.path.exists(dst_path):
                os.remove(dst_path)
            os.rename(src_path, dst_path)

        if "Vocals" in base:
            vocal_file = dst_path
    return vocal_file


def _detect_and_separate_vocals(video_path, model_mgr):
    """Handles vocal separation if configured."""
    if not config.USE_VOCAL_SEPARATION:
        return video_path
    try:
        # 1. Check for existing output
        existing_vocal = _get_separated_vocal_path(video_path)
        if existing_vocal:
            log(f"  [AI] Resuming with existing vocals: {os.path.basename(existing_vocal)}")
            return existing_vocal

        # 2. Run separation
        log("  [Task 0/4] Separating Vocals (BS-Roformer)...")
        audio_input_path = utils.extract_clean_audio(video_path)

        target_dir = os.path.abspath(os.path.dirname(video_path))
        separator = model_mgr.get_separator(output_dir=target_dir)
        output_files = separator.separate(audio_input_path)

        vocal_file = _process_separator_outputs(output_files, target_dir)

        if vocal_file and os.path.exists(vocal_file):
            log(f"  [Sep] Vocal track isolated: {os.path.basename(vocal_file)}")
            return vocal_file

    except (RuntimeError, OSError) as e:
        log(f"  [Sep] Warning: Separation failed ({e}). Using original audio.", "WARNING")

    return video_path


def _filter_hallucinations(segments, hallucination_phrases):
    """Internal helper to filter out hallucinated segments."""
    filtered_segments = []
    hallucinated_count = 0

    for s in segments:
        text_clean = s.text.strip().lower().strip(".,!?;: ")
        is_hallucination = False
        for phrase in hallucination_phrases:
            if phrase in text_clean and len(text_clean) < len(phrase) + 5:
                is_hallucination = True
                break

        if is_hallucination:
            hallucinated_count += 1
            continue
        filtered_segments.append(s)

    return filtered_segments, hallucinated_count


def _process_transcription_segments(segments_gen, total_dur, start_time):
    """Internal helper to process segments and update progress."""
    segments = []
    for segment in segments_gen:
        segments.append(segment)
        elapsed = time.time() - start_time
        # Calculate speed (audio seconds per real second)
        speed = segment.end / elapsed if elapsed > 0 else 0
        eta = (total_dur - segment.end) / speed if speed > 0 and total_dur > segment.end else 0

        # Print segment text (Verbose UI)
        sys.stdout.write("\r\033[K")

        # fast-whisper segments usually have 'avg_logprob'
        # standard openai-whisper segments also have 'avg_logprob'
        prob = math.exp(segment.avg_logprob) if hasattr(segment, "avg_logprob") else 1.0

        ts_start = utils.format_timestamp(segment.start)
        ts_end = utils.format_timestamp(segment.end)
        print(f"[{ts_start}->{ts_end}] ({prob:.0%}) {segment.text.strip()}")

        utils.print_progress_bar(
            segment.end,
            total_dur,
            prefix="  [Whisper] Transcribing",
            timestamp_str=f"{utils.format_timestamp(segment.end)} / {utils.format_timestamp(total_dur)}",
            speed=speed,
            eta=eta,
        )
    return segments


def _prepare_audio(video_path, model_mgr):
    """Prepares audio for transcription (Separation or Extraction)."""
    transcribe_path = video_path
    if config.USE_VOCAL_SEPARATION:
        transcribe_path = _detect_and_separate_vocals(video_path, model_mgr)
        model_mgr.offload_separator()

    if transcribe_path == video_path:
        transcribe_path = utils.extract_clean_audio(video_path)
    return transcribe_path


def _perform_transcription(whisper_model, path, prompt, lang, vad_params):
    """Executes the raw Whisper transcription call."""
    try:
        return whisper_model.transcribe(
            path,
            beam_size=OPTIMIZER.config["whisper_beam"],
            initial_prompt=prompt,
            vad_filter=True,
            vad_parameters=vad_params,
            language=lang,
            condition_on_previous_text=True,
            no_speech_threshold=0.6,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            log("  [Whisper] OOM detected. Clearing cache and retrying...", "WARNING")
            if torch is not None:
                torch.cuda.empty_cache()
            gc.collect()
            time.sleep(1)
            return whisper_model.transcribe(
                path,
                beam_size=max(1, OPTIMIZER.config["whisper_beam"] // 2),
                initial_prompt=prompt,
                vad_filter=True,
                vad_parameters=vad_params,
                language=lang,
                condition_on_previous_text=True,
                no_speech_threshold=0.6,
            )
        raise


def _log_transcription_config(lang_to_use, current_prompt):
    """Log the language and prompt configuration used for transcription."""
    if lang_to_use:
        log(f"  [Whisper] Config: Forced Language='{lang_to_use}'")
    else:
        log("  [Whisper] Config: Language Auto-Detection Enabled")

    if current_prompt:
        log(f"  [Whisper] Config: Input Prompt='{current_prompt}'")
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
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 500,
        }

        segments_gen, info = _perform_transcription(whisper_model, transcribe_path, current_prompt, lang_to_use, vad_params)

        segments = _process_transcription_segments(segments_gen, info.duration, start_time)

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
