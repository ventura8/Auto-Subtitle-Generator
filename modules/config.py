"""
Configuration module for Auto Subtitle Generator.
Handles loading settings from config.yaml and prompts.yaml.
"""
import os
import yaml
from typing import Dict, Any


# =============================================================================
# CONSTANTS & DEFAULTS
# =============================================================================

LOG_FILE = "subtitle_gen.log"
WHISPER_MODEL_SIZE = "large-v3"


# Optimized prompt (Music start prevents early hallucinations)
INITIAL_PROMPT = "Transcribe the following audio file."
USE_VOCAL_SEPARATION = True
FORCED_LANGUAGE = None
PROMPT_USE_CUSTOM_PRIORITY = False  # If True, custom_prompt overrides everything
DEBUG_LOGGING = False  # Controls detailed console output

# Anti-hallucination thresholds (Ultra-relaxed for debugging gap)
HALLUCINATION_SILENCE_THRESHOLD = 0.9   # Discard if >90% no-speech prob
HALLUCINATION_REPETITION_THRESHOLD = 15  # Flag if same segment repeats 15+ times

# Known hallucination phrases that Whisper outputs on unintelligible audio
HALLUCINATION_PHRASES = [
    # Romanian
    "nu uitați să dați like", "nu uitati sa dati like",
    "să lăsați un comentariu", "sa lasati un comentariu",
    "să distribuiți", "sa distribuiti",
    "abonați-vă la canal", "abonati-va la canal",
    "nu uitați să vă abonați", "nu uitati sa va abonati",
    "pentru a nu rata videoclipurile noastre",
    "nu uitați să dați like, să lăsați un comentariu și să distribuiți "
    "acest material video pe alte rețele sociale",
    "nu uitati sa dati like, sa lasati un comentariu si sa distribuiti "
    "acest material video pe alte retele sociale",
    "nu uitați să vă abonați la canal, să vă mulțumim și la rețeta următoare",
    "abonati-va la canal, sa va multumim si la reteta urmatoare",
    "vă mulțumim pentru vizionare", "va multumim pentru vizionare",
    "nu uitați să apăsați butonul de like",

    # English
    "thank you for watching", "thanks for watching",
    "subscribe to my channel", "please subscribe",
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

VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".m4v", ".ts", ".mts"
}

# AI Model settings (Defaults)
NLLB_MODEL_ID = "facebook/nllb-200-3.3B"
NLLB_NUM_BEAMS = 5
NLLB_LENGTH_PENALTY = 1.0
NLLB_REPETITION_PENALTY = 1.0
NLLB_NO_REPEAT_NGRAM_SIZE = 0
AUDIO_SEPARATOR_MODEL_ID = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
VAD_MIN_SILENCE_MS = 500


# NLLB language codes mapped by ISO 639-1
TARGET_LANGUAGES: dict[str, dict[str, str]] = {}

# ISO 639-1 to NLLB Code Mapping (Static fallback)
ISO_TO_NLLB = {
    # Major European
    "en": "eng_Latn", "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn",
    "it": "ita_Latn", "pt": "por_Latn", "ru": "rus_Cyrl", "zh": "zho_Hans",
    "ja": "jpn_Jpan", "ko": "kor_Hang", "hi": "hin_Deva", "ar": "arb_Arab",

    # Eastern European
    "ro": "ron_Latn", "bg": "bul_Cyrl", "cs": "ces_Latn", "pl": "pol_Latn",
    "hu": "hun_Latn", "uk": "ukr_Cyrl", "sk": "slk_Latn", "sl": "slv_Latn",
    "sr": "srp_Cyrl", "hr": "hrv_Latn", "el": "ell_Grek", "tr": "tur_Latn",

    # Northern European
    "nl": "nld_Latn", "sv": "swe_Latn", "da": "dan_Latn", "fi": "fin_Latn",
    "no": "nob_Latn", "et": "est_Latn", "lv": "lav_Latn", "lt": "lit_Latn",

    # Asian
    "th": "tha_Thai", "vi": "vie_Latn", "id": "ind_Latn", "ms": "zsm_Latn",
    "he": "heb_Hebr", "sd": "snd_Arab", "gu": "guj_Gujr", "mr": "mar_Deva",
    "bn": "ben_Beng", "pa": "pan_Guru", "ta": "tam_Tamil", "te": "tel_Telu",
    "kn": "kan_Knda", "ml": "mal_Mlym",

    # African
    "sw": "swh_Latn", "am": "amh_Ethi", "yo": "yor_Latn", "ig": "ibo_Latn",
    "ha": "hau_Latn", "zu": "zul_Latn", "xh": "xho_Latn", "af": "afr_Latn",
    "so": "som_Latn", "lg": "lug_Latn", "sn": "sna_Latn", "ny": "nya_Latn",
    "rw": "kin_Latn", "mg": "plt_Latn",

    # Others
    "hy": "hye_Armn", "ka": "kat_Geor", "az": "azj_Latn", "be": "bel_Cyrl"
}


# =============================================================================
# LOADING LOGIC
# =============================================================================

def get_nllb_code(iso_code):
    """Returns the NLLB code for a given ISO 639-1 code (default: English)."""
    # First check target languages config (user overrides)
    if iso_code in TARGET_LANGUAGES:
        return TARGET_LANGUAGES[iso_code]["code"]

    # Then check static map
    return ISO_TO_NLLB.get(iso_code, "eng_Latn")


def _load_whisper_language(w_conf, logger_func):
    global FORCED_LANGUAGE
    if "language" in w_conf:
        val = w_conf["language"]
        # Handle YAML 'false' boolean or empty string
        if val is False or val == "False" or not val:
            FORCED_LANGUAGE = None
        else:
            FORCED_LANGUAGE = str(val)
        logger_func(f"[Config] Forced Language: {FORCED_LANGUAGE}")


def _load_whisper_prompt(w_conf, logger_func):
    global INITIAL_PROMPT, PROMPT_USE_CUSTOM_PRIORITY
    PROMPT_USE_CUSTOM_PRIORITY = w_conf.get("custom_prompt_priority", False)
    if w_conf.get("use_prompt", True):
        custom = w_conf.get("custom_prompt", "")
        if custom:
            INITIAL_PROMPT = custom
            mode = "PRIORITY" if PROMPT_USE_CUSTOM_PRIORITY else "Base"
            bias = "disabled" if PROMPT_USE_CUSTOM_PRIORITY else "enabled"
            logger_func(
                f"[Config] Using Custom Prompt ({mode} Mode). "
                f"Auto-bias {bias}."
            )
        else:
            logger_func("[Config] Using Default Prompt (Enabled in config).")
    else:
        INITIAL_PROMPT = None
        logger_func("[Config] Prompt Disabled in config.")


def _load_whisper_config(w_conf, logger_func):
    global WHISPER_MODEL_SIZE, USE_VOCAL_SEPARATION

    if "model_size" in w_conf:
        WHISPER_MODEL_SIZE = w_conf["model_size"]
        logger_func(f"[Config] Whisper Model: {WHISPER_MODEL_SIZE}")

    _load_whisper_language(w_conf, logger_func)

    USE_VOCAL_SEPARATION = w_conf.get("use_vocal_separation", True)
    status = "ENABLED" if USE_VOCAL_SEPARATION else "DISABLED"
    logger_func(f"[Config] Vocal Separation: {status}")

    _load_whisper_prompt(w_conf, logger_func)


def _load_hallucination_config(h_conf, logger_func):
    global HALLUCINATION_SILENCE_THRESHOLD, HALLUCINATION_REPETITION_THRESHOLD
    global HALLUCINATION_PHRASES
    if "silence_threshold" in h_conf:
        HALLUCINATION_SILENCE_THRESHOLD = float(h_conf["silence_threshold"])
    if "repetition_threshold" in h_conf:
        HALLUCINATION_REPETITION_THRESHOLD = int(h_conf["repetition_threshold"])
    if "known_phrases" in h_conf and isinstance(h_conf["known_phrases"], list):
        HALLUCINATION_PHRASES = h_conf["known_phrases"]
    logger_func(
        f"[Config] Loaded Hallucination Filters "
        f"(Silence: {HALLUCINATION_SILENCE_THRESHOLD}, "
        f"Rep: {HALLUCINATION_REPETITION_THRESHOLD})"
    )


def _load_performance_overrides(p_conf: Dict[str, Any], optimizer: Any, logger_func: Any) -> None:
    if not p_conf:
        return
    updated_keys = []
    if p_conf.get("whisper_beam") is not None:
        optimizer.config["whisper_beam"] = int(p_conf["whisper_beam"])
        optimizer.config["whisper_beam_overridden"] = True
        updated_keys.append("whisper_beam")
    if p_conf.get("nllb_batch"):
        optimizer.config["nllb_batch"] = int(p_conf["nllb_batch"])
        updated_keys.append("nllb_batch")
    if p_conf.get("whisper_workers"):
        optimizer.config["whisper_workers"] = int(p_conf["whisper_workers"])
        updated_keys.append("whisper_workers")
    if p_conf.get("ffmpeg_threads"):
        optimizer.config["ffmpeg_threads"] = int(p_conf["ffmpeg_threads"])
        updated_keys.append("ffmpeg_threads")

    if updated_keys:
        logger_func(
            f"[Config] Performance Overrides: "
            f"{', '.join(updated_keys)}"
        )


def _load_base_config_snippet(config: Dict[str, Any], logger_func: Any) -> None:
    global DEBUG_LOGGING
    if "debug_logging" in config:
        DEBUG_LOGGING = config["debug_logging"]

    if "target_languages" in config:
        TARGET_LANGUAGES.update(config["target_languages"])
        logger_func(
            f"[Config] Loaded {len(TARGET_LANGUAGES)} languages from config."
        )

    if "whisper" in config:
        _load_whisper_config(config["whisper"], logger_func)

    if "hallucinations" in config:
        _load_hallucination_config(config["hallucinations"], logger_func)


def _load_nllb_config(n_conf: Dict[str, Any], logger_func: Any) -> None:
    global NLLB_NUM_BEAMS, NLLB_LENGTH_PENALTY, NLLB_REPETITION_PENALTY, NLLB_NO_REPEAT_NGRAM_SIZE
    if "num_beams" in n_conf:
        NLLB_NUM_BEAMS = int(n_conf["num_beams"])
    if "length_penalty" in n_conf:
        NLLB_LENGTH_PENALTY = float(n_conf["length_penalty"])
    if "repetition_penalty" in n_conf:
        NLLB_REPETITION_PENALTY = float(n_conf["repetition_penalty"])
    if "no_repeat_ngram_size" in n_conf:
        NLLB_NO_REPEAT_NGRAM_SIZE = int(n_conf["no_repeat_ngram_size"])

    logger_func(
        f"[Config] NLLB Quality: Beams={NLLB_NUM_BEAMS}, "
        f"LenPen={NLLB_LENGTH_PENALTY}, "
        f"RepPen={NLLB_REPETITION_PENALTY}, "
        f"NgramBlock={NLLB_NO_REPEAT_NGRAM_SIZE}"
    )


def _load_type_and_model_config(config: Dict[str, Any], logger_func: Any) -> None:
    global VIDEO_EXTENSIONS, NLLB_MODEL_ID, AUDIO_SEPARATOR_MODEL_ID

    if "file_types" in config and "extensions" in config["file_types"]:
        exts = config["file_types"]["extensions"]
        if exts:
            VIDEO_EXTENSIONS = set(exts)
            logger_func(
                f"[Config] Loaded {len(VIDEO_EXTENSIONS)} video extensions."
            )

    if "models" in config:
        m_conf = config["models"]
        if "nllb" in m_conf:
            NLLB_MODEL_ID = m_conf["nllb"]
        if "audio_separator" in m_conf:
            AUDIO_SEPARATOR_MODEL_ID = m_conf["audio_separator"]
        logger_func(
            f"[Config] Models: NLLB={NLLB_MODEL_ID}, "
            f"Separator={AUDIO_SEPARATOR_MODEL_ID}"
        )


def load_config(optimizer: Any, logger_func: Any) -> bool:
    """Loads configuration from config.yaml."""
    global VAD_MIN_SILENCE_MS

    config_path = "config.yaml"
    if not os.path.exists(config_path):
        logger_func(
            "[Config] config.yaml not found. Using internal defaults.",
            "WARNING"
        )
        TARGET_LANGUAGES.update({
            "en": {"code": "eng_Latn", "label": "English"},
            "es": {"code": "spa_Latn", "label": "Spanish"},
            "fr": {"code": "fra_Latn", "label": "French"},
        })
        return True

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        _load_base_config_snippet(config, logger_func)
        _load_type_and_model_config(config, logger_func)

        if "nllb" in config and isinstance(config["nllb"], dict):
            _load_nllb_config(config["nllb"], logger_func)

        if "vad" in config and "min_silence_duration_ms" in config["vad"]:
            VAD_MIN_SILENCE_MS = int(config["vad"]["min_silence_duration_ms"])
            logger_func(f"[Config] VAD Min Silence: {VAD_MIN_SILENCE_MS}ms")

        if "performance" in config:
            _load_performance_overrides(
                config["performance"], optimizer, logger_func
            )

        return True
    except Exception as e:
        logger_func(f"[Config] Error loading config.yaml: {e}", "ERROR")
        return False
