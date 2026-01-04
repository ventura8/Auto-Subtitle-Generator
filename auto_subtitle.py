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
import math
import subprocess
import warnings
import gc
import multiprocessing
import platform
import time
import yaml
import logging
import signal
import re
import threading

warnings.filterwarnings("ignore")
logging.getLogger("transformers").addFilter(lambda record: "tied weights" not in record.getMessage())

# =============================================================================
# CONFIGURATION
# =============================================================================

LOG_FILE = "subtitle_gen.log"
WHISPER_MODEL_SIZE = "large-v3"

# Multilingual prompt to help Whisper recognize expected languages
INITIAL_PROMPT = (
    "This video contains speech in multiple languages including "
    "Romanian, English, French, Italian, German, and Spanish."
)

# Anti-hallucination thresholds (tuned for singing/music content)
HALLUCINATION_SILENCE_THRESHOLD = 0.1   # Skip segments with >10% no-speech probability
HALLUCINATION_REPETITION_THRESHOLD = 5  # Flag if same segment repeats 5+ times

# Known hallucination phrases that Whisper outputs on unintelligible audio
HALLUCINATION_PHRASES = [
    # Romanian
    "nu uitați să dați like", "nu uitati sa dati like",
    "să lăsați un comentariu", "sa lasati un comentariu",
    "să distribuiți", "sa distribuiti",
    "abonați-vă la canal", "abonati-va la canal",
    "mulțumesc pentru vizionare", "multumesc pentru vizionare",
    # English
    "thank you for watching", "thanks for watching",
    "don't forget to subscribe", "please subscribe",
    "like and subscribe", "hit the like button",
    "leave a comment", "share this video",
    "see you in the next", "bye bye",
    # French
    "merci d'avoir regardé", "n'oubliez pas de vous abonner",
    "laissez un commentaire", "à bientôt",
    # German
    "danke fürs zuschauen", "vergisst nicht zu abonnieren",
    # Spanish
    "gracias por ver", "no olvides suscribirte",
    # Italian
    "grazie per aver guardato", "non dimenticare di iscriverti",
]

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".m4v", ".ts", ".mts"}

# AI Model settings (Defaults)
NLLB_MODEL_ID = "facebook/nllb-200-3.3B"
AUDIO_SEPARATOR_MODEL_ID = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
VAD_MIN_SILENCE_MS = 500

# NLLB language codes mapped by ISO 639-1
# Loaded from config.yaml dynamically
TARGET_LANGUAGES = {}


# =============================================================================
# LAZY IMPORTS (loaded on demand to speed up startup)
# =============================================================================

torch = None
AutoTokenizer = None
AutoModelForSeq2SeqLM = None
WhisperModel = None

# =============================================================================
# UTILITIES
# =============================================================================


def handle_shutdown(signum, frame):
    """Handles termination signals for graceful shutdown."""
    print("\n\n[!] Termination detected. Stopping process...")
    sys.exit(0)


def setup_signal_handlers():
    """Registers signal handlers for SIGINT and SIGTERM."""
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)


def log(message, level="INFO", to_console=True):
    """Logs a message to both console and log file.

    Args:
        message: The message to log.
        level: Log level - INFO, WARNING, ERROR, CRITICAL, or DEBUG.
               DEBUG level only writes to log file, never to console.
        to_console: If True, also prints to console (ignored for DEBUG level).
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] [{level}] {message}"

    # DEBUG level messages go only to the log file, never to console
    if level != "DEBUG" and to_console:
        prefix = {"ERROR": "!!! ", "WARNING": "! ", "CRITICAL": "XXX "}.get(level, "")
        # Use \r\033[K to clear any active progress bar on the current line before logging
        print(f"\r\033[K{prefix}{message}")

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry + "\n")


def print_progress_bar(
    iteration, total, prefix='', suffix='', decimals=1, length=30,
    fill='█', empty='░', elapsed=None, speed=None, eta=None
):
    """
    Call in a loop to create terminal progress bar.
    Format: [███████████████░░░░░░░░░░░░░░░]  53.3% | 00:17:02.28 | 2.57x | ETA 00:15:30 | Rendering...

    Args:
        elapsed: Elapsed time in seconds (float). Displayed as HH:MM:SS.cc
        speed: Speed multiplier (float). Displayed as X.XXx
        eta: Estimated time remaining in seconds (float). Displayed as ETA HH:MM:SS
    """
    if total == 0:
        total = 1

    percent_float = 100 * (iteration / float(total))
    percent_str = ("{0:." + str(decimals) + "f}").format(percent_float)
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + empty * (length - filled_length)

    # Build info parts
    info_parts = [f'{percent_str:>5}%']

    if elapsed is not None:
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = elapsed % 60
        elapsed_str = f'{hours:02d}:{minutes:02d}:{seconds:05.2f}'
        info_parts.append(elapsed_str)

    if speed is not None:
        speed_str = f'{speed:.2f}x'
        info_parts.append(speed_str)

    if eta is not None and eta > 0:
        eta_hours = int(eta // 3600)
        eta_minutes = int((eta % 3600) // 60)
        eta_seconds = int(eta % 60)
        eta_str = f'ETA {eta_hours:02d}:{eta_minutes:02d}:{eta_seconds:02d}'
        info_parts.append(eta_str)

    if suffix:
        info_parts.append(suffix)

    info_display = ' | '.join(info_parts)

    # Clear line before printing to handle varying line lengths
    # \033[K clears from cursor to end of line, preventing leftover text artifacts
    sys.stdout.write(f'\r\033[K{prefix}[{bar}] {info_display}')
    sys.stdout.flush()

    # Print new line on complete
    if iteration >= total:
        print()


# =============================================================================
# HARDWARE OPTIMIZATION
# =============================================================================

class SystemOptimizer:
    """Auto-detects hardware and applies optimal settings for AI workloads."""

    def __init__(self):
        self.profile = "STANDARD"
        self.vram_gb = 0
        self.ram_gb = 0
        self.cpu_cores = multiprocessing.cpu_count()
        self.gpu_name = "None"
        self.config = {
            "whisper_beam": 5,
            "whisper_compute": "float16",
            "nllb_batch": 16,
            "ffmpeg_threads": max(1, self.cpu_cores - 2),
            "device": "cpu"
        }

    def detect_hardware(self, verbose=True):
        """Probes system for GPU/VRAM and assigns performance profile."""
        if verbose:
            log("[Auto-Detect] Scanning Hardware...")
            log(f"[Auto-Detect] CPU Cores: {self.cpu_cores}")

        try:
            import torch as t
            if t.cuda.is_available():
                props = t.cuda.get_device_properties(0)
                self.vram_gb = round(props.total_memory / (1024**3), 2)
                self.config["device"] = "cuda"
                self.gpu_name = props.name
                if verbose:
                    log(f"[Auto-Detect] GPU Detected: {props.name} ({self.vram_gb} GB VRAM)")
            else:
                self.config["device"] = "cpu"
                if verbose:
                    log("[Auto-Detect] No CUDA GPU found. Falling back to CPU.")
        except ImportError:
            if verbose:
                log("[Auto-Detect] Torch not loaded yet, assuming CPU for now.")

        # Assign profile based on VRAM
        if self.config["device"] == "cuda":
            if self.vram_gb >= 22:
                self.set_profile("ULTRA", verbose=verbose)
            elif self.vram_gb >= 15:
                self.set_profile("HIGH", verbose=verbose)
            elif self.vram_gb >= 10:
                self.set_profile("MID", verbose=verbose)
            else:
                self.set_profile("LOW", verbose=verbose)
        else:
            self.set_profile("CPU_ONLY", verbose=verbose)

        return self.config

    def set_profile(self, profile_name, verbose=True):
        """Applies a named performance profile."""
        self.profile = profile_name
        if verbose:
            log(f"[Optimization] Applied Profile: {profile_name}")

        # Calculate dynamic NLLB batch size based on available VRAM
        # NLLB 3.3B in float16: ~5GB model (heavier)
        # Target: 90% VRAM utilization for high-end GPUs
        if self.vram_gb >= 8:
            available_vram = self.vram_gb * 0.90  # Use 90% of VRAM
            model_overhead = 8  # 5GB model + 3GB safety
            per_batch_memory = 1.5  # ~1.5GB per batch item (safe estimate for 3.3B)
            dynamic_nllb_batch = max(1, min(16, int((available_vram - model_overhead) / per_batch_memory)))
        else:
            dynamic_nllb_batch = 1

        profiles = {
            "ULTRA": {
                "whisper_beam": 5, "whisper_compute": "float16",
                "nllb_batch": dynamic_nllb_batch, "ffmpeg_threads": self.cpu_cores
            },
            "HIGH": {"whisper_beam": 5, "nllb_batch": dynamic_nllb_batch, "ffmpeg_threads": self.cpu_cores},
            "MID": {"whisper_beam": 5, "nllb_batch": max(1, dynamic_nllb_batch // 2)},
            "LOW": {"whisper_beam": 2, "nllb_batch": 1, "whisper_compute": "int8_float16"},
            "CPU_ONLY": {"whisper_beam": 2, "nllb_batch": 1, "whisper_compute": "int8", "device": "cpu"},
        }
        if profile_name in profiles:
            self.config.update(profiles[profile_name])
            if verbose and profile_name in ["ULTRA", "HIGH", "MID"]:
                log(f"[Optimization] Dynamic NLLB batch size: {self.config['nllb_batch']} "
                    f"(based on {self.vram_gb}GB VRAM)")


OPTIMIZER = SystemOptimizer()


def load_config():
    """Loads configuration from config.yaml."""
    global INITIAL_PROMPT, WHISPER_MODEL_SIZE, VIDEO_EXTENSIONS
    global NLLB_MODEL_ID, AUDIO_SEPARATOR_MODEL_ID, VAD_MIN_SILENCE_MS
    global HALLUCINATION_SILENCE_THRESHOLD, HALLUCINATION_REPETITION_THRESHOLD, HALLUCINATION_PHRASES

    config_path = "config.yaml"
    if not os.path.exists(config_path):
        log("[Config] config.yaml not found. Using internal defaults.", "WARNING")
        # Define default languages if config is missing (same as before)
        TARGET_LANGUAGES.update({
             "en": {"code": "eng_Latn", "label": "English"},
             "es": {"code": "spa_Latn", "label": "Spanish"},
             "fr": {"code": "fra_Latn", "label": "French"},
        })
        return

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if "target_languages" in config:
            TARGET_LANGUAGES.update(config["target_languages"])
            log(f"[Config] Loaded {len(TARGET_LANGUAGES)} languages from config.")

        if "whisper" in config:
            w_conf = config["whisper"]

            # Model Size
            if "model_size" in w_conf:
                WHISPER_MODEL_SIZE = w_conf["model_size"]
                log(f"[Config] Whisper Model: {WHISPER_MODEL_SIZE}")

            # Prompt Settings
            if w_conf.get("use_prompt", True):
                custom = w_conf.get("custom_prompt", "")
                if custom:
                    INITIAL_PROMPT = custom
                    log("[Config] Using Custom Prompt from config.")
                else:
                    log("[Config] Using Default Prompt (Enabled in config).")
            else:
                INITIAL_PROMPT = None
                log("[Config] Prompt Disabled in config.")

        if "hallucinations" in config:
            h_conf = config["hallucinations"]
            if "silence_threshold" in h_conf:
                HALLUCINATION_SILENCE_THRESHOLD = float(h_conf["silence_threshold"])
            if "repetition_threshold" in h_conf:
                HALLUCINATION_REPETITION_THRESHOLD = int(h_conf["repetition_threshold"])
            if "known_phrases" in h_conf and isinstance(h_conf["known_phrases"], list):
                HALLUCINATION_PHRASES = h_conf["known_phrases"]
            log(f"[Config] Loaded Hallucination Filters (Silence: {HALLUCINATION_SILENCE_THRESHOLD}, "
                f"Rep: {HALLUCINATION_REPETITION_THRESHOLD})")

        if "file_types" in config and "extensions" in config["file_types"]:
            exts = config["file_types"]["extensions"]
            if exts:
                VIDEO_EXTENSIONS = set(exts)
                log(f"[Config] Loaded {len(VIDEO_EXTENSIONS)} video extensions.")

        if "models" in config:
            m_conf = config["models"]
            if "nllb" in m_conf:
                NLLB_MODEL_ID = m_conf["nllb"]
            if "audio_separator" in m_conf:
                AUDIO_SEPARATOR_MODEL_ID = m_conf["audio_separator"]
            log(f"[Config] Models: NLLB={NLLB_MODEL_ID}, Separator={AUDIO_SEPARATOR_MODEL_ID}")

        if "vad" in config:
            if "min_silence_duration_ms" in config["vad"]:
                VAD_MIN_SILENCE_MS = int(config["vad"]["min_silence_duration_ms"])
                log(f"[Config] VAD Min Silence: {VAD_MIN_SILENCE_MS}ms")

        if "performance" in config:
            p_conf = config["performance"]
            if p_conf:
                updated_keys = []
                if p_conf.get("whisper_beam"):
                    OPTIMIZER.config["whisper_beam"] = int(p_conf["whisper_beam"])
                    updated_keys.append("whisper_beam")
                if p_conf.get("nllb_batch"):
                    OPTIMIZER.config["nllb_batch"] = int(p_conf["nllb_batch"])
                    updated_keys.append("nllb_batch")
                if p_conf.get("ffmpeg_threads"):
                    OPTIMIZER.config["ffmpeg_threads"] = int(p_conf["ffmpeg_threads"])
                    updated_keys.append("ffmpeg_threads")

                if updated_keys:
                    log(f"[Config] Performance Overrides Applied: {', '.join(updated_keys)}")

    except Exception as e:
        log(f"[Config] Error loading config.yaml: {e}", "ERROR")


# =============================================================================
# AI ENGINE INITIALIZATION
# =============================================================================

def init_ai_engine():
    """Lazily loads all AI dependencies with a progress indicator."""
    global torch, AutoTokenizer, AutoModelForSeq2SeqLM, WhisperModel
    if torch is not None:
        return

    print("[Init] ---------------------------------------------------------------")
    print("[Init] Initializing AI Engine Components...")
    print("[Init] ---------------------------------------------------------------")

    total_steps = 6

    def update_bar(step, msg, status="RUNNING"):
        print_progress_bar(
            step, total_steps,
            prefix="[Init] ",
            suffix=f"{msg:<35} [{status}]",
            length=25,
            decimals=0
        )

    # Step 1: PyTorch
    update_bar(1, "Loading PyTorch")
    try:
        import torch as _torch
        torch = _torch
        update_bar(1, "Loading PyTorch", "OK")
    except ImportError as e:
        update_bar(1, "Loading PyTorch", "FAIL")
        print("")
        log(f"[Fatal] PyTorch missing: {e}", "CRITICAL")
        sys.exit(1)

    # Step 2: Hardware Detection
    update_bar(2, "Detecting Hardware")
    OPTIMIZER.detect_hardware(verbose=False)
    update_bar(2, "Detecting Hardware", "OK")

    # Step 3: NVIDIA Paths
    update_bar(3, "Configuring NVIDIA Paths")
    load_nvidia_paths()
    update_bar(3, "Configuring NVIDIA Paths", "OK")

    # Step 4: Transformers
    update_bar(4, "Loading Transformers (NLLB)")
    try:
        from transformers import AutoTokenizer as _AT, AutoModelForSeq2SeqLM as _AM
        AutoTokenizer = _AT
        AutoModelForSeq2SeqLM = _AM
        update_bar(4, "Loading Transformers (NLLB)", "OK")
    except ImportError:
        update_bar(4, "Loading Transformers (NLLB)", "FAIL")
        print("")
        log("[Fatal] Transformers missing.", "CRITICAL")
        sys.exit(1)

    # Step 5: Faster-Whisper
    update_bar(5, "Loading Faster-Whisper")
    try:
        from faster_whisper import WhisperModel as _WM
        WhisperModel = _WM
        update_bar(5, "Loading Faster-Whisper", "OK")
    except ImportError:
        update_bar(5, "Loading Faster-Whisper", "FAIL")
        print("")
        log("[Fatal] Faster-Whisper missing.", "CRITICAL")
        sys.exit(1)

    # Step 6: Audio-Separator
    update_bar(6, "Loading Audio-Separator")
    try:
        import audio_separator.separator
        _ = audio_separator.separator.Separator
        update_bar(6, "Loading Audio-Separator", "OK")
    except ImportError:
        update_bar(6, "Loading Audio-Separator", "SKIP")
        log("[Warning] audio-separator not installed. Vocal separation will be skipped.", "WARNING")

    print("")
    print("[Init] AI Engine Ready.\n")


def load_nvidia_paths():
    """Adds CUDNN/CUBLAS to PATH for Windows compatibility."""
    try:
        import nvidia.cudnn
        import nvidia.cublas
        for lib in [nvidia.cudnn, nvidia.cublas]:
            if hasattr(lib, '__path__') and lib.__path__:
                path = lib.__path__[0]
            elif hasattr(lib, '__file__') and lib.__file__:
                path = os.path.dirname(lib.__file__)
            else:
                continue

            bin_path = os.path.join(path, "bin")
            if os.path.exists(bin_path):
                os.environ['PATH'] = bin_path + os.pathsep + os.environ['PATH']
                if hasattr(os, 'add_dll_directory'):
                    try:
                        os.add_dll_directory(bin_path)
                    except (AttributeError, OSError):
                        pass
    except ImportError:
        pass


def get_ffmpeg_paths():
    """Returns paths to FFmpeg binaries, preferring local venv installation."""
    base = os.path.dirname(os.path.abspath(__file__))
    venv_ffmpeg = os.path.join(base, "venv", "ffmpeg", "bin", "ffmpeg.exe")
    venv_ffprobe = os.path.join(base, "venv", "ffmpeg", "bin", "ffprobe.exe")

    if os.path.exists(venv_ffmpeg) and os.path.exists(venv_ffprobe):
        return venv_ffmpeg, venv_ffprobe
    return "ffmpeg", "ffprobe"


FFMPEG_CMD, FFPROBE_CMD = get_ffmpeg_paths()


# =============================================================================
# PROCESSING FUNCTIONS
# =============================================================================

def get_audio_duration(file_path):
    """Returns duration of audio file in seconds."""
    try:
        cmd = [
            FFPROBE_CMD, "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", file_path
        ]
        return float(subprocess.check_output(cmd).decode().strip())
    except Exception:
        return 0.0


def format_timestamp(seconds):
    """Converts seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60
    milliseconds = round((seconds - math.floor(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{math.floor(seconds):02d},{milliseconds:03d}"


def parse_timestamp(ts_str):
    """Converts SRT timestamp (HH:MM:SS,mmm) to seconds."""
    h, m, s_ms = ts_str.split(':')
    s, ms = s_ms.split(',')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


class Segment:
    """Represents a subtitle segment with timing and text."""
    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text


def parse_srt(path):
    """Parses an SRT file into a list of Segment objects."""
    segments = []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    blocks = content.split("\n\n")
    for block in blocks:
        lines = block.split("\n")
        if len(lines) >= 3:
            times = lines[1].split(" --> ")
            start = parse_timestamp(times[0])
            end = parse_timestamp(times[1])
            text = " ".join(lines[2:])
            segments.append(Segment(start, end, text))
    return segments


def extract_clean_audio(video_path):
    """Extracts audio from video, normalizes volume, and returns WAV path."""
    base_dir = os.path.dirname(video_path)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    temp_wav = os.path.join(base_dir, f"{base_name}_temp.wav")

    # Reuse existing temp file if valid
    if os.path.exists(temp_wav):
        try:
            if get_audio_duration(temp_wav) > 0:
                log("  [Pre-Process] Found valid existing temp audio.")
                return temp_wav
        except (OSError, ValueError):
            pass

    log("  [Pre-Process] Extracting & Normalizing Audio...")

    cmd = [
        FFMPEG_CMD, "-y", "-i", video_path,
        "-threads", str(OPTIMIZER.config["ffmpeg_threads"]),
        "-ar", "44100", "-ac", "2",
        "-af", "loudnorm",
        "-c:a", "pcm_f32le",
        "-loglevel", "error",
        temp_wav
    ]

    try:
        subprocess.run(cmd, check=True)
        return temp_wav
    except subprocess.CalledProcessError as e:
        log(f"  [Error] Audio extraction failed: {e}", "ERROR")
        return video_path


# =============================================================================
# NLLB TRANSLATOR
# =============================================================================

class NLLBTranslator:
    """Wrapper for NLLB-200 translation model."""

    def __init__(self):
        self.model_id = NLLB_MODEL_ID
        self.device = OPTIMIZER.config["device"]
        self.tokenizer = None
        self.model = None
        self.load()

    def load(self):
        """Loads the NLLB model and tokenizer."""
        log(f"  [Model-Load] Loading {self.model_id} on {self.device.upper()}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            # Load model in float32 for maximum stability (32GB VRAM allows this)
            if self.device == "cuda":
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                ).to("cuda")
                log("  [Model-Load] Using float32 for maximum precision/stability.")
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )

            # Warmup...
            if self.device == "cuda":
                dummy = self.tokenizer("Hello world", return_tensors="pt").to("cuda")
                _ = self.model.generate(**dummy)

            log("  [Model-Load] NLLB Loaded Successfully.")
        except Exception as e:
            log(f"  [Fatal] Failed to load NLLB: {e}", "CRITICAL")
            raise e

    def encode_batch(self, texts, src_code):
        """Runs the encoder on a batch of texts and returns encoder outputs.

        Returns a tuple of (encoder_outputs, inputs) where inputs contains
        input_ids and attention_mask needed for generation.
        """
        if not texts:
            return None

        self.tokenizer.src_lang = src_code
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            encoder_outputs = self.model.get_encoder()(**inputs)

        # Return both encoder outputs AND full inputs (needed for generate)
        return encoder_outputs, inputs

    def decode_batch(self, encoder_outputs, inputs, tgt_code):
        """Runs the decoder using pre-computed encoder outputs.

        Args:
            encoder_outputs: Pre-computed encoder hidden states
            inputs: Original tokenized inputs (contains input_ids, attention_mask)
            tgt_code: Target language NLLB code
        """
        if encoder_outputs is None:
            return []

        # Get target language token ID
        if hasattr(self.tokenizer, "lang_code_to_id"):
            bos_id = self.tokenizer.lang_code_to_id.get(tgt_code, None)
        else:
            bos_id = self.tokenizer.convert_tokens_to_ids(tgt_code)

        if bos_id is None:
            bos_id = self.tokenizer.convert_tokens_to_ids(tgt_code)

        with torch.no_grad():
            generated = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                encoder_outputs=encoder_outputs,
                forced_bos_token_id=bos_id,
                max_length=512
            )

        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)

    def translate(self, texts, src_code, tgt_code):
        """Translates a batch of texts from source to target language."""
        if not texts:
            return []

        self.tokenizer.src_lang = src_code
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)

        # Get target language token ID
        if hasattr(self.tokenizer, "lang_code_to_id"):
            bos_id = self.tokenizer.lang_code_to_id.get(tgt_code, None)
            src_lang_id = self.tokenizer.lang_code_to_id.get(src_code, None)
        else:
            bos_id = self.tokenizer.convert_tokens_to_ids(tgt_code)
            src_lang_id = self.tokenizer.convert_tokens_to_ids(src_code)

        if bos_id is None:
            bos_id = self.tokenizer.convert_tokens_to_ids(tgt_code)

        # CRITICAL FIX: Manually prepend Source Language Token if missing
        # NLLB requires the source token (e.g., ron_Latn) at the start.
        if src_lang_id is not None:
            # Check if already present (just in case)
            if inputs['input_ids'][0, 0] != src_lang_id:
                B, L = inputs['input_ids'].shape
                prefix_ids = torch.full((B, 1), src_lang_id, dtype=inputs['input_ids'].dtype, device=self.device)
                prefix_mask = torch.ones((B, 1), dtype=inputs['attention_mask'].dtype, device=self.device)

                inputs['input_ids'] = torch.cat([prefix_ids, inputs['input_ids']], dim=1)
                inputs['attention_mask'] = torch.cat([prefix_mask, inputs['attention_mask']], dim=1)

        with torch.no_grad():
            # Calculate reasonable max length to prevent infinite loops
            input_len = inputs["input_ids"].shape[1]
            max_output_len = min(512, int(input_len * 2.0) + 10)

            # High-Quality Generation Parameters (NLLB)
            # 3.3B Model requires penalties to prevent loops
            generated = self.model.generate(
                **inputs,
                forced_bos_token_id=bos_id,
                max_length=max_output_len,
                num_beams=5,                  # Beam search for best context understanding
                repetition_penalty=1.2,       # Moderate penalty for 3.3B
                no_repeat_ngram_size=3,       # Prevent phrase repetition
                early_stopping=True
            )

        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)


# =============================================================================
# MODEL MANAGER
# =============================================================================

class ModelManager:
    """Manages persistent AI model instances to avoid reloading overhead."""

    def __init__(self):
        self._whisper = None
        self._nllb_translator = None
        self._separator = None
        self._preload_thread = None

    def get_whisper(self):
        """Lazily loads and returns the Whisper model."""
        if self._whisper is None:
            log("[AI] Loading Whisper...")
            self._whisper = WhisperModel(
                WHISPER_MODEL_SIZE,
                device=OPTIMIZER.config["device"],
                compute_type=OPTIMIZER.config["whisper_compute"]
            )
        return self._whisper

    def preload_nllb(self):
        """Starts loading NLLB in a background thread."""
        if self._nllb_translator is None and self._preload_thread is None:
            self._preload_thread = threading.Thread(target=self._load_nllb_worker)
            self._preload_thread.start()

    def _load_nllb_worker(self):
        """Worker for background loading."""
        try:
            self._nllb_translator = NLLBTranslator()
        except Exception as e:
            log(f"[Optimization] Background NLLB load failed: {e}", "WARNING")
            self._nllb_translator = None

    def get_nllb(self):
        """Lazily loads and returns the NLLBTranslator."""
        if self._preload_thread is not None:
            if self._preload_thread.is_alive():
                log("[AI] Waiting for background NLLB load to finish...")
            self._preload_thread.join()
            self._preload_thread = None

        if self._nllb_translator is None:
            self._nllb_translator = NLLBTranslator()
        return self._nllb_translator

    def unload_all(self):
        """Explicitly unloads all models to free VRAM."""
        if self._whisper:
            del self._whisper
            self._whisper = None
        if self._nllb_translator:
            del self._nllb_translator
            self._nllb_translator = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# =============================================================================
# VIDEO PROCESSING PIPELINE
# =============================================================================

def cleanup_temp_files(folder, base_name, file_name):
    """Removes temporary files (audio & intermediate SRTs) created during processing."""
    temp_wav = os.path.join(folder, f"{base_name}_temp.wav")
    pattern_audio = f"{base_name}_temp"

    if os.path.exists(folder):
        for f in os.listdir(folder):
            # Remove partial audio files
            if pattern_audio in f and (".wav" in f or ".mp3" in f):
                if f == file_name:
                    continue
                try:
                    os.remove(os.path.join(folder, f))
                except OSError:
                    pass

            # Remove intermediate SRT files (e.g., vid.en.srt, vid.es.srt)
            # We look for files starting with base_name and ending in .srt
            # if f.startswith(base_name) and f.endswith(".srt"):
            #     try:
            #         os.remove(os.path.join(folder, f))
            #     except OSError:
            #         pass

    if os.path.exists(temp_wav):
        try:
            os.remove(temp_wav)
        except OSError:
            pass


def process_video(video_path, model_manager, initial_prompt=INITIAL_PROMPT):
    """Main processing pipeline for a single video file."""
    start_time = time.time()

    file_name = os.path.basename(video_path)
    base_name = os.path.splitext(file_name)[0]
    folder = os.path.dirname(video_path)

    log(f"\n>>> PROCESSING: {file_name}")
    log(f"    Profile: {OPTIMIZER.profile} | Threads: {OPTIMIZER.config['ffmpeg_threads']}")

    # Step 1: Check if already processed
    name_no_ext, ext = os.path.splitext(file_name)
    final_output_path = os.path.join(folder, f"{name_no_ext}_multilang{ext}")

    if os.path.exists(final_output_path):
        log(f"  [Skip] Final output already exists: {final_output_path}")
        return

    # Step 2: Transcription
    generated_srts = []
    original_segments = []
    detected_lang = None

    # Check for existing SRT files to enable resume
    files = os.listdir(folder)
    potential_orig_srts = [f for f in files if f.startswith(base_name) and f.endswith(".srt")]

    prior_srt = None
    prior_lang = None

    for f in potential_orig_srts:
        parts = f.split('.')
        if len(parts) >= 3:
            l_code = parts[-2]
            try:
                p = os.path.join(folder, f)
                candidate_segments = parse_srt(p)
                if candidate_segments:
                    prior_srt = p
                    prior_lang = l_code
                    original_segments = candidate_segments
                    log(f"  [Resume] Found existing SRT: {f}. Skipping Whisper.")
                    break
            except Exception:
                continue

    if not prior_srt:
        # Extract audio
        audio_path = extract_clean_audio(video_path)

        # Vocal separation
        vocal_path = None
        for f in os.listdir(folder):
            if base_name in f and "Vocals" in f and f.endswith(".wav"):
                vocal_path = os.path.join(folder, f)
                log(f"  [Resume] Found existing Vocals: {f}. Skipping separation.")
                break

        if not vocal_path:
            try:
                from audio_separator.separator import Separator
                log("  [AI] Isolating Vocals (BS-Roformer)...")

                # Audio Separator might hold state, so we initialize it here for safety
                # unless we refactor it too. For now, let's keep it local as it might be less frequent
                # or just let caching handle it.
                # However, to be consistent with "Performance Investigation", reusing it is better if possible.
                # Let's keep it local for now as it takes output_dir at init.
                # The heavy lifting is `load_model`.

                separator = Separator(
                    output_dir=folder,
                    output_format="WAV",
                    sample_rate=44100,
                    output_single_stem="Vocals"
                )
                separator.load_model(AUDIO_SEPARATOR_MODEL_ID)
                output_files = separator.separate(audio_path)

                for f in output_files:
                    if "Vocals" in f:
                        vocal_path = os.path.join(folder, f)
                        break

            except Exception as e:
                log(f"  [Error] Vocal separation failed: {e}. Aborting processing.", "ERROR")
                raise e

        if vocal_path:
            audio_path = vocal_path
            log(f"  [AI] Using Vocals: {os.path.basename(vocal_path)}")

        # Run Whisper transcription
        try:
            # OPTIMIZATION: Start loading NLLB in background if VRAM allows
            if OPTIMIZER.profile in ["ULTRA", "HIGH"]:
                log(f"[Optimization] Pre-loading NLLB model in background (Profile: {OPTIMIZER.profile})...")
                model_manager.preload_nllb()

            whisper = model_manager.get_whisper()

            log(f"[AI] Transcribing using {WHISPER_MODEL_SIZE} (Beam: {OPTIMIZER.config['whisper_beam']})...")
            segments, info = whisper.transcribe(
                audio_path,
                beam_size=OPTIMIZER.config["whisper_beam"],
                language=None,
                initial_prompt=initial_prompt,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=VAD_MIN_SILENCE_MS),
                condition_on_previous_text=False,
                no_speech_threshold=HALLUCINATION_SILENCE_THRESHOLD,
            )

            detected_lang = info.language
            log(f"[AI] Detected Language: {detected_lang.upper()} "
                f"(Probability: {info.language_probability:.2f})")

            if info.language_probability < 0.5:
                log(f"  [Warning] Low language confidence ({info.language_probability:.2f}). "
                    "May indicate music/noise.", "WARNING")

            print("    -> Transcription Progress:")
            total_dur = get_audio_duration(audio_path)
            if total_dur == 0:
                total_dur = 1

            last_texts = []
            hallucination_detected = False

            for seg in segments:
                seg_text = seg.text.strip()

                # Filter 1: Skip empty segments
                if not seg_text:
                    continue

                # Filter 2: Skip known hallucination phrases
                seg_lower = seg_text.lower()
                is_hallucination = False
                for phrase in HALLUCINATION_PHRASES:
                    if phrase in seg_lower:
                        log(f"  [Warning] Skipping hallucination: '{seg_text[:60]}...'", "WARNING")
                        is_hallucination = True
                        break
                if is_hallucination:
                    continue

                # Filter 3: Detect repetition loops
                last_texts.append(seg_text)
                if len(last_texts) > HALLUCINATION_REPETITION_THRESHOLD:
                    last_texts.pop(0)

                if len(last_texts) >= HALLUCINATION_REPETITION_THRESHOLD:
                    if len(set(last_texts)) == 1:
                        if not hallucination_detected:
                            log(f"  [Warning] Hallucination detected: '{seg_text[:50]}...' "
                                "repeating. Stopping.", "WARNING")
                            hallucination_detected = True
                        continue

                # Filter 4: Skip if still in hallucination loop
                if hallucination_detected:
                    if seg_text != last_texts[-1] if last_texts else True:
                        hallucination_detected = False
                        last_texts.clear()
                    else:
                        continue

                original_segments.append(seg)

                # Update progress with text preview
                # Update progress with text preview
                # Update progress with text preview (Safe Print)
                # Format: [MM:SS] [XX%] Text content...
                time_str = time.strftime('%M:%S', time.gmtime(seg.start))
                percent = min(100, int((seg.end / total_dur) * 100))
                print(f"    [{time_str}] [{percent}%] {seg_text}", flush=True)

            if hallucination_detected:
                print("\n    -> [!] Transcription stopped early due to hallucination loop.")

            log("Transcription Complete.")
            print("")

            # DO NOT DELETE WHISPER HERE - Managed by ModelManager
            # del whisper
            # torch.cuda.empty_cache()

            orig_srt_path = os.path.join(folder, f"{base_name}.{detected_lang}.srt")
            save_srt(original_segments, orig_srt_path)
            generated_srts.append((orig_srt_path, detected_lang, "Original"))

        except Exception as e:
            log(f"  [Error] Transcription failed: {e}", "ERROR")
            cleanup_temp_files(folder, base_name, file_name)
            return
    else:
        detected_lang = prior_lang
        generated_srts.append((prior_srt, prior_lang, "Original (Resumed)"))

    # Step 3: Translation
    # Unload Whisper to free VRAM for NLLB translation
    if model_manager._whisper is not None:
        log("  [Memory] Unloading Whisper to free VRAM for translation...")
        del model_manager._whisper
        model_manager._whisper = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    if not original_segments:
        log("  [Warning] No speech detected. Skipping translation.")
    else:
        try:
            # Map Whisper language code to NLLB code
            src_nllb = "eng_Latn"
            for k, v in TARGET_LANGUAGES.items():
                if k == detected_lang:
                    src_nllb = v["code"]
                    break
            if detected_lang == "ro":
                src_nllb = "ron_Latn"

            # Filter out existing translations
            all_tasks = [t for t in TARGET_LANGUAGES.items() if t[0] != detected_lang]
            tasks_to_run = []

            for lang_key, info in all_tasks:
                tgt_srt_path = os.path.join(folder, f"{base_name}.{lang_key}.srt")
                if os.path.exists(tgt_srt_path):
                    generated_srts.append((tgt_srt_path, lang_key, info["label"]))
                else:
                    tasks_to_run.append((lang_key, info))

            if tasks_to_run:
                translator = model_manager.get_nllb()
                # Sanitize source texts to remove any terminal control characters
                source_texts = []
                for s in original_segments:
                    txt = s.text.strip().replace('\r', '').replace('\x1b', '')
                    txt = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', txt)
                    source_texts.append(txt)

                log(f"  [AI] Translating to {len(tasks_to_run)} languages "
                    f"(Skipped {len(all_tasks) - len(tasks_to_run)})...")

                current_batch = OPTIMIZER.config["nllb_batch"]

                # Translate each language and save immediately
                for lang_idx, (lang_key, info) in enumerate(tasks_to_run):
                    tgt_code = info["code"]
                    lang_label = info["label"]
                    translated_lines = []

                    start_idx = 0
                    while start_idx < len(source_texts):
                        batch = source_texts[start_idx:start_idx + current_batch]

                        try:
                            batch_results = translator.translate(batch, src_nllb, tgt_code)
                            translated_lines.extend(batch_results)

                            # Clear cache after each batch to prevent VRAM buildup
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                            # Update Progress with clean log (Single line overwrite)
                            processed_segs = min(start_idx + current_batch, len(source_texts))
                            percentage = int((processed_segs / len(source_texts)) * 100)
                            print(f"\r    [Translation] {lang_label}: {processed_segs}/{len(source_texts)} "
                                  f"({percentage}%)" + " "*10, end="", flush=True)

                            start_idx += current_batch

                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                if current_batch <= 1:
                                    log("  [Error] OOM even at batch size 1. Skipping segment.", "ERROR")
                                    translated_lines.extend(["[Translation Failed]"] * len(batch))
                                    start_idx += current_batch
                                else:
                                    torch.cuda.empty_cache()
                                    gc.collect()
                                    current_batch = max(1, current_batch // 2)
                                    log(f"  [OOM] Reducing batch size to {current_batch} and retrying...", "WARNING")
                                    continue
                            else:
                                raise e
                        except Exception as e:
                            log(f"  [Error] Translation batch failed: {e}. Skipping.", "ERROR")
                            translated_lines.extend(["[Error]"] * len(batch))
                            start_idx += current_batch

                    print("")  # Newline after translation completes

                    # IMMEDIATELY save this language's SRT after completing translation
                    final_segments = []
                    if len(translated_lines) != len(original_segments):
                        log(f"  [Warning] Line count mismatch for {lang_key}. "
                            f"Expected {len(original_segments)}, got {len(translated_lines)}.", "WARNING")

                    for i, seg in enumerate(original_segments):
                        text = translated_lines[i] if i < len(translated_lines) else "[Missing]"
                        final_segments.append(Segment(seg.start, seg.end, text))

                    tgt_srt_path = os.path.join(folder, f"{base_name}.{lang_key}.srt")
                    save_srt(final_segments, tgt_srt_path)
                    generated_srts.append((tgt_srt_path, lang_key, info["label"]))

                    # Clear CUDA cache between languages to prevent VRAM buildup
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                print("")  # Close progress line
            else:
                log("  [AI] All translations already exist. Skipping NLLB.")

        except Exception as e:
            log(f"  [Error] Translation failed: {e}", "ERROR")

    # Step 4: Embed subtitles
    log("  [Muxing] Embedding subtitles into container...")
    embed_subtitles(video_path, generated_srts)

    cleanup_temp_files(folder, base_name, file_name)

    duration = time.time() - start_time
    log(f"  [Done] Finished {file_name} in {int(duration)}s.")


def save_srt(segments, path):
    """Saves segments to an SRT file."""
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            text = seg.text.strip() if hasattr(seg, 'text') else str(seg).strip()
            # Sanitize: remove carriage returns and control characters
            text = text.replace('\r', '').replace('\x1b', '')
            # Sanitize: remove carriage returns and control characters
            text = text.replace('\r', '').replace('\x1b', '')
            # Remove ANSI escape sequences (from progress bar)
            text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}\n")
            f.write(f"{text}\n")
            f.write("\n")


def save_translated_srt(original_segs, translated_texts, path):
    """Saves translated text with original timing to an SRT file."""
    with open(path, "w", encoding="utf-8") as f:
        for i, (seg, text) in enumerate(zip(original_segs, translated_texts), 1):
            # Sanitize: remove carriage returns and ANSI escape sequences
            clean_text = text.strip().replace('\r', '').replace('\x1b', '')
            clean_text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', clean_text)
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}\n")
            f.write(f"{clean_text}\n")
            f.write("\n")


def embed_subtitles(video_path, srt_files):
    """Embeds all subtitle tracks into the video container using FFmpeg."""
    if not srt_files:
        return

    dir_name = os.path.dirname(video_path)
    file_name = os.path.basename(video_path)
    name_no_ext, ext = os.path.splitext(file_name)
    output_path = os.path.join(dir_name, f"{name_no_ext}_multilang{ext}")

    cmd = [FFMPEG_CMD, "-y", "-i", video_path]

    # Add each subtitle file with explicit UTF-8 encoding
    for srt, _, _ in srt_files:
        cmd.extend(["-sub_charenc", "UTF-8", "-i", srt])

    cmd.extend(["-map", "0:v", "-map", "0:a"])
    for i in range(len(srt_files)):
        cmd.extend(["-map", f"{i+1}"])

    cmd.extend(["-c:v", "copy", "-c:a", "copy", "-c:s",
                "mov_text" if ext in [".mp4", ".m4v", ".mov"] else "srt"])

    for i, (_, lang, label) in enumerate(srt_files):
        cmd.extend([f"-metadata:s:s:{i}", f"language={lang}", f"-metadata:s:s:{i}", f"title={label}"])

    cmd.extend(["-loglevel", "error", output_path])

    subprocess.run(cmd)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Application entry point with interactive and CLI modes."""
    setup_signal_handlers()

    if len(sys.argv) == 1:
        # Interactive mode
        init_ai_engine()

        # Load Config (Apply overrides before printing status)
        load_config()

        print("\n" + "="*60)
        print("   AUTO SUBTITLE GENERATOR (OPTIMIZED) - v1.0.0")
        print(f"   Running on: {platform.system()} {platform.release()}")
        print("="*60)

        print("\n[HARDWARE DETECTED]")
        print(f"   CPU : {OPTIMIZER.cpu_cores} Logical Cores")
        print(f"   GPU : {OPTIMIZER.gpu_name} ({OPTIMIZER.vram_gb} GB VRAM)")

        print(f"\n[AUTO-TUNED SETTINGS -> Profile: {OPTIMIZER.profile}]")
        print(f"   AI Compute  : {OPTIMIZER.config['whisper_compute']}")
        print(f"   Beam Size   : {OPTIMIZER.config['whisper_beam']}")
        print(f"   Batch Size  : {OPTIMIZER.config['nllb_batch']}")
        print(f"   Threads     : {OPTIMIZER.config['ffmpeg_threads']}")
        print("\n" + "-"*60)
        print(" [SETTINGS]")
        if INITIAL_PROMPT:
            if len(INITIAL_PROMPT) > 60:
                truncated = INITIAL_PROMPT[:57] + "..."
                print(f" Prompt: {truncated}")
            else:
                print(f" Prompt: {INITIAL_PROMPT}")
        else:
            print(" Prompt: Disabled (Raw Audio Context)")
        print(f" Languages: {len(TARGET_LANGUAGES)} loaded from config.yaml")

        print("\n" + "-"*60)
        print(" [HOW TO USE]")
        print(" 1. Drag and Drop a video file (or folder) here.")
        print(" 2. Or paste the file path below.")
        print("-"*60 + "\n")

        try:
            target = input(">> Drag & Drop Video Path: ").strip('"')
            if target.startswith("& '") and target.endswith("'"):
                target = target[3:-1]
        except KeyboardInterrupt:
            sys.exit(0)

    else:
        target = sys.argv[1]

    target = os.path.abspath(target)

    if os.path.exists(target):
        init_ai_engine()
        load_config()

        model_manager = ModelManager()

        if os.path.isdir(target):
            files = [
                os.path.join(target, f) for f in os.listdir(target)
                if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS
            ]
            for f in files:
                if "_multilang" not in f:
                    process_video(f, model_manager, initial_prompt=INITIAL_PROMPT)
        else:
            process_video(target, model_manager, initial_prompt=INITIAL_PROMPT)

        print("\nAll tasks completed.")
        time.sleep(3)
    else:
        print(f"Invalid path: {target}")
        time.sleep(3)


if __name__ == "__main__":
    main()
