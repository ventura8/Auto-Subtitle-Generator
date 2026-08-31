"""
Configuration module for Auto Subtitle Generator.
Handles loading settings from config.yaml and prompts.yaml.
"""

import importlib
import os
import sys
import webbrowser
from typing import Any, Dict

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
HALLUCINATION_SILENCE_THRESHOLD = 0.9  # Discard if >90% no-speech prob
HALLUCINATION_REPETITION_THRESHOLD = 15  # Flag if same segment repeats 15+ times

# Known hallucination phrases that Whisper outputs on unintelligible audio
HALLUCINATION_PHRASES = [
    # Romanian
    "nu uitați să dați like",
    "nu uitati sa dati like",
    "să lăsați un comentariu",
    "sa lasati un comentariu",
    "să distribuiți",
    "sa distribuiti",
    "abonați-vă la canal",
    "abonati-va la canal",
    "nu uitați să vă abonați",
    "nu uitati sa va abonati",
    "pentru a nu rata videoclipurile noastre",
    "nu uitați să dați like, să lăsați un comentariu și să distribuiți acest material video pe alte rețele sociale",
    "nu uitati sa dati like, sa lasati un comentariu si sa distribuiti acest material video pe alte retele sociale",
    "nu uitați să vă abonați la canal, să vă mulțumim și la rețeta următoare",
    "abonati-va la canal, sa va multumim si la reteta urmatoare",
    "vă mulțumim pentru vizionare",
    "va multumim pentru vizionare",
    "nu uitați să apăsați butonul de like",
    "Să vă mulțumesc frumos pentru vizionare",
    # English
    "thank you for watching",
    "thanks for watching",
    "subscribe to my channel",
    "please subscribe",
    "like and subscribe",
    "hit the like button",
    "leave a comment",
    "share this video",
    "see you in the next",
    "bye bye",
    # French
    "merci d'avoir regardé",
    "n'oubliez pas de vous abonner",
    "laissez un commentaire",
    "à bientôt",
    # German
    "danke fürs zuschauen",
    "vergisst nicht zu abonnieren",
    # Spanish
    "gracias por ver",
    "no olvides suscribirte",
    # Italian
    "grazie per aver guardato",
    "non dimenticare di iscriverti",
]

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".m4v", ".ts", ".mts"}

# AI Model settings (Defaults)
TRANSLATOR_ENGINE = "nllb"
TRANSLATEGEMMA_MODEL_ID = "google/translategemma-12b-it"
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
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "hi": "hin_Deva",
    "ar": "arb_Arab",
    "fa": "pes_Arab",
    # Eastern European
    "ro": "ron_Latn",
    "bg": "bul_Cyrl",
    "cs": "ces_Latn",
    "pl": "pol_Latn",
    "hu": "hun_Latn",
    "uk": "ukr_Cyrl",
    "sk": "slk_Latn",
    "sl": "slv_Latn",
    "sr": "srp_Cyrl",
    "hr": "hrv_Latn",
    "el": "ell_Grek",
    "tr": "tur_Latn",
    # Northern European
    "nl": "nld_Latn",
    "sv": "swe_Latn",
    "da": "dan_Latn",
    "fi": "fin_Latn",
    "no": "nob_Latn",
    "et": "est_Latn",
    "lv": "lvs_Latn",
    "lt": "lit_Latn",
    # Asian
    "th": "tha_Thai",
    "vi": "vie_Latn",
    "id": "ind_Latn",
    "ms": "zsm_Latn",
    "he": "heb_Hebr",
    "sd": "snd_Arab",
    "gu": "guj_Gujr",
    "mr": "mar_Deva",
    "bn": "ben_Beng",
    "pa": "pan_Guru",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    # African
    "sw": "swh_Latn",
    "am": "amh_Ethi",
    "yo": "yor_Latn",
    "ig": "ibo_Latn",
    "ha": "hau_Latn",
    "zu": "zul_Latn",
    "xh": "xho_Latn",
    "af": "afr_Latn",
    "so": "som_Latn",
    "lg": "lug_Latn",
    "sn": "sna_Latn",
    "ny": "nya_Latn",
    "rw": "kin_Latn",
    "mg": "plt_Latn",
    # Others
    "hy": "hye_Armn",
    "ka": "kat_Geor",
    "az": "azj_Latn",
    "be": "bel_Cyrl",
}

NLLB_PREFIX_TO_ISO = {
    "eng": "en",
    "spa": "es",
    "fra": "fr",
    "deu": "de",
    "ita": "it",
    "por": "pt",
    "rus": "ru",
    "zho": "zh",
    "jpn": "ja",
    "kor": "ko",
    "hin": "hi",
    "arb": "ar",
    "ron": "ro",
    "bul": "bg",
    "ces": "cs",
    "pol": "pl",
    "hun": "hu",
    "ukr": "uk",
    "slk": "sk",
    "slv": "sl",
    "srp": "sr",
    "hrv": "hr",
    "ell": "el",
    "tur": "tr",
    "nld": "nl",
    "swe": "sv",
    "dan": "da",
    "fin": "fi",
    "nob": "no",
    "est": "et",
    "lav": "lv",
    "lvs": "lv",
    "lit": "lt",
    "tha": "th",
    "vie": "vi",
    "ind": "id",
    "zsm": "ms",
    "heb": "he",
    "snd": "sd",
    "guj": "gu",
    "mar": "mr",
    "ben": "bn",
    "pan": "pa",
    "tam": "ta",
    "tel": "te",
    "kan": "kn",
    "mal": "ml",
    "swh": "sw",
    "amh": "am",
    "yor": "yo",
    "ibo": "ig",
    "hau": "ha",
    "zul": "zu",
    "xho": "xh",
    "afr": "af",
    "som": "so",
    "lug": "lg",
    "sna": "sn",
    "nya": "ny",
    "kin": "rw",
    "plt": "mg",
    "hye": "hy",
    "kat": "ka",
    "azj": "az",
    "bel": "be",
    "pes": "fa",
}

# Mapping from ISO 639-1 / 2-letter codes to standard ISO 639-2 (3-letter) codes for container metadata
ISO_639_1_TO_639_2 = {
    "ar": "ara",
    "fa": "fas",
    "ms": "msa",
    "mg": "mlg",
    "sw": "swa",
    "az": "aze",
    "lv": "lav",
    "no": "nor",
    "nb": "nob",
    "nn": "nno",
    "zh": "zho",
    "ja": "jpn",
    "ko": "kor",
    "de": "deu",
    "fr": "fra",
    "es": "spa",
    "it": "ita",
    "pt": "por",
    "ru": "rus",
    "hi": "hin",
    "ro": "ron",
    "tr": "tur",
    "vi": "vie",
    "pl": "pol",
    "nl": "nld",
    "id": "ind",
    "uk": "ukr",
    "th": "tha",
    "cs": "ces",
    "hu": "hun",
    "sv": "swe",
    "el": "ell",
    "da": "dan",
    "fi": "fin",
    "bg": "bul",
    "hr": "hrv",
    "sr": "srp",
    "sk": "slk",
    "sl": "slv",
    "lt": "lit",
    "et": "est",
    "he": "heb",
    "en": "eng",
}

# Specific NLLB 3-letter prefixes mapped to standard ISO 639-2 codes
NLLB_PREFIX_TO_ISO639_2 = {
    "arb": "ara",
    "pes": "fas",
    "zsm": "msa",
    "plt": "mlg",
    "swh": "swa",
    "azj": "aze",
    "lvs": "lav",
    "nob": "nob",
}


def to_mux_language_code(lang: str | None) -> str:
    """Convert language code to a standard ISO 639-2 3-letter container metadata code."""
    if not lang:
        return "und"

    iso_639_2 = ISO_639_1_TO_639_2.get(lang)
    if iso_639_2:
        return iso_639_2

    prefix = _get_mux_language_prefix(lang)
    return NLLB_PREFIX_TO_ISO639_2.get(prefix, prefix)


def _get_mux_language_prefix(lang: str) -> str:
    """Return the NLLB language prefix used for container metadata mapping."""
    lang_info = TARGET_LANGUAGES.get(lang)
    if isinstance(lang_info, dict) and lang_info.get("code"):
        nllb_code = lang_info["code"]
    else:
        nllb_code = ISO_TO_NLLB.get(lang, lang)
    return nllb_code.split("_", maxsplit=1)[0] if nllb_code else lang


# =============================================================================
# LOADING LOGIC
# =============================================================================


def _get_yaml_module():
    """Return the optional PyYAML module used to parse config files."""
    try:
        return importlib.import_module("yaml")
    except ImportError:
        return None


def get_nllb_code(iso_code):
    """Returns the NLLB code for a given ISO 639-1 code (default: English)."""
    # First check target languages config (user overrides)
    if iso_code in TARGET_LANGUAGES:
        return TARGET_LANGUAGES[iso_code]["code"]

    # Then check static map
    return ISO_TO_NLLB.get(iso_code, "eng_Latn")


def nllb_to_iso(code):
    """Maps NLLB language code back to ISO 639-1."""
    if not code:
        return "en"
    target_language_iso = _map_iso_from_target_languages(code)
    if target_language_iso:
        return target_language_iso

    static_iso = _map_iso_from_static_lookup(code)
    if static_iso:
        return static_iso

    return _map_iso_from_prefix(code)


def _map_iso_from_target_languages(code: str) -> str | None:
    """Resolve ISO code using loaded target language configuration."""
    for iso, info in TARGET_LANGUAGES.items():
        if info.get("code") == code:
            return iso
    return None


def _map_iso_from_static_lookup(code: str) -> str | None:
    """Resolve ISO code using static ISO<->NLLB mapping."""
    for iso, nllb in ISO_TO_NLLB.items():
        if nllb == code:
            return iso
    return None


def _map_iso_from_prefix(code: str) -> str:
    """Resolve ISO code from NLLB language prefix fallback."""
    if "_" not in code:
        return code
    prefix = code.split("_")[0]
    return NLLB_PREFIX_TO_ISO.get(prefix, prefix[:2])


def _load_whisper_language(w_conf, logger_func):
    if "language" in w_conf:
        val = w_conf["language"]
        # Handle YAML 'false' boolean or empty string
        if val is False or val == "False" or not val:
            globals()["FORCED_LANGUAGE"] = None
        else:
            globals()["FORCED_LANGUAGE"] = str(val)
        logger_func(f"[Config] Forced Language: {globals().get('FORCED_LANGUAGE')}")


def _load_whisper_prompt(w_conf, logger_func):
    globals()["PROMPT_USE_CUSTOM_PRIORITY"] = w_conf.get("custom_prompt_priority", False)
    if w_conf.get("use_prompt", True):
        custom = w_conf.get("custom_prompt", "")
        if custom:
            globals()["INITIAL_PROMPT"] = custom
            mode = "PRIORITY" if globals().get("PROMPT_USE_CUSTOM_PRIORITY") else "Base"
            bias = "disabled" if globals().get("PROMPT_USE_CUSTOM_PRIORITY") else "enabled"
            logger_func(f"[Config] Using Custom Prompt ({mode} Mode). Auto-bias {bias}.")
        else:
            logger_func("[Config] Using Default Prompt (Enabled in config).")
    else:
        globals()["INITIAL_PROMPT"] = None
        logger_func("[Config] Prompt Disabled in config.")


def _load_whisper_config(w_conf, logger_func):
    if "model_size" in w_conf:
        globals()["WHISPER_MODEL_SIZE"] = w_conf["model_size"]
        logger_func(f"[Config] Whisper Model: {globals().get('WHISPER_MODEL_SIZE')}")

    _load_whisper_language(w_conf, logger_func)

    globals()["USE_VOCAL_SEPARATION"] = w_conf.get("use_vocal_separation", True)
    status = "ENABLED" if globals().get("USE_VOCAL_SEPARATION") else "DISABLED"
    logger_func(f"[Config] Vocal Separation: {status}")

    _load_whisper_prompt(w_conf, logger_func)


def _load_hallucination_config(h_conf, logger_func):
    if "silence_threshold" in h_conf:
        globals()["HALLUCINATION_SILENCE_THRESHOLD"] = float(h_conf["silence_threshold"])
    if "repetition_threshold" in h_conf:
        globals()["HALLUCINATION_REPETITION_THRESHOLD"] = int(h_conf["repetition_threshold"])
    if "known_phrases" in h_conf and isinstance(h_conf["known_phrases"], list):
        globals()["HALLUCINATION_PHRASES"] = h_conf["known_phrases"]
    logger_func(
        f"[Config] Loaded Hallucination Filters (Silence: {globals().get('HALLUCINATION_SILENCE_THRESHOLD')}, "
        f"Rep: {globals().get('HALLUCINATION_REPETITION_THRESHOLD')})"
    )


def _load_performance_overrides(p_conf: Dict[str, Any], optimizer: Any, logger_func: Any) -> None:
    if not p_conf:
        return

    updated_keys: list[str] = []
    _apply_whisper_beam_override(p_conf, optimizer, updated_keys)
    _apply_numeric_override(p_conf, optimizer, "nllb_batch", updated_keys)
    _apply_numeric_override(p_conf, optimizer, "translategemma_batch", updated_keys)
    _apply_numeric_override(p_conf, optimizer, "translategemma_max_new_tokens", updated_keys)
    _apply_numeric_override(p_conf, optimizer, "whisper_workers", updated_keys)
    _apply_numeric_override(p_conf, optimizer, "ffmpeg_threads", updated_keys)

    if updated_keys:
        logger_func(f"[Config] Performance Overrides: {', '.join(updated_keys)}")


def _apply_whisper_beam_override(p_conf: Dict[str, Any], optimizer: Any, updated_keys: list[str]) -> None:
    """Apply whisper beam override and mark profile override flag."""
    whisper_beam = p_conf.get("whisper_beam")
    if whisper_beam is None:
        return
    parsed_whisper_beam = int(whisper_beam)
    if parsed_whisper_beam <= 0:
        raise ValueError("performance.whisper_beam must be a positive integer")
    optimizer.config["whisper_beam"] = parsed_whisper_beam
    optimizer.config["whisper_beam_overridden"] = True
    updated_keys.append("whisper_beam")


def _apply_numeric_override(p_conf: Dict[str, Any], optimizer: Any, key: str, updated_keys: list[str]) -> None:
    """Apply integer performance override for one key when present."""
    value = p_conf.get(key)
    if value is None:
        return
    parsed_value = int(value)
    if parsed_value <= 0:
        raise ValueError(f"performance.{key} must be a positive integer")
    optimizer.config[key] = parsed_value
    updated_keys.append(key)


def _load_base_config_snippet(config: Dict[str, Any], logger_func: Any) -> None:
    if "debug_logging" in config:
        globals()["DEBUG_LOGGING"] = config["debug_logging"]

    _load_translation_engine(config.get("translation"), logger_func)
    _load_target_languages(config.get("target_languages"), logger_func)
    _load_optional_mapping_section(config.get("whisper"), _load_whisper_config, logger_func, "whisper")

    hallucination_config = config.get("hallucinations")
    if hallucination_config:
        _load_hallucination_config(hallucination_config, logger_func)


def _load_optional_mapping_section(section_value: Any, loader: Any, logger_func: Any, section_name: str) -> None:
    """Load an optional config section only when it is a mapping."""
    if section_value and not isinstance(section_value, dict):
        raise ValueError(f"{section_name} config section must be a mapping/object")
    if section_value:
        loader(section_value, logger_func)


def _load_translation_engine(translation_config: Any, logger_func: Any) -> None:
    """Load translation engine selection from config."""
    if not isinstance(translation_config, dict):
        return
    if "engine" not in translation_config:
        return
    globals()["TRANSLATOR_ENGINE"] = str(translation_config["engine"]).lower()
    logger_func(f"[Config] Translation Engine: {globals().get('TRANSLATOR_ENGINE')}")


def _load_target_languages(target_languages: Any, logger_func: Any) -> None:
    """Load normalized target language mapping from config."""
    if target_languages is None:
        TARGET_LANGUAGES.clear()
        return

    if not isinstance(target_languages, dict):
        raise ValueError("target_languages config section must be a mapping/object")

    normalized_languages = _normalize_target_languages(target_languages, logger_func)
    if not normalized_languages:
        raise ValueError("target_languages must contain at least one valid language entry with required key 'code'")

    TARGET_LANGUAGES.clear()
    TARGET_LANGUAGES.update(normalized_languages)
    logger_func(f"[Config] Loaded {len(TARGET_LANGUAGES)} languages from config.")


def _normalize_target_languages(raw_languages: Dict[Any, Any], logger_func: Any) -> Dict[str, Dict[str, str]]:
    """Normalize target language keys loaded from YAML to stable ISO strings."""
    normalized: Dict[str, Dict[str, str]] = {}
    if not isinstance(raw_languages, dict):
        return normalized

    for raw_key, lang_info in raw_languages.items():
        if not isinstance(lang_info, dict):
            continue
        if "code" not in lang_info:
            logger_func(f"[Config] Skipping target language '{raw_key}' because it is missing required key 'code'.", "WARNING")
            continue

        lang_key = _normalize_language_key(raw_key, logger_func)
        normalized[lang_key] = lang_info

    return normalized


def _normalize_language_key(raw_key: Any, logger_func: Any) -> str:
    """Normalize YAML language keys to deterministic lowercase-like strings."""
    if isinstance(raw_key, bool):
        lang_key = "no" if raw_key is False else "yes"
        logger_func(f"[Config] Normalized boolean language key '{raw_key}' to '{lang_key}'.", "WARNING")
        return lang_key
    if isinstance(raw_key, str):
        return raw_key
    return str(raw_key)


def _load_nllb_config(n_conf: Dict[str, Any], logger_func: Any) -> None:
    if "num_beams" in n_conf:
        globals()["NLLB_NUM_BEAMS"] = int(n_conf["num_beams"])
    if "length_penalty" in n_conf:
        globals()["NLLB_LENGTH_PENALTY"] = float(n_conf["length_penalty"])
    if "repetition_penalty" in n_conf:
        globals()["NLLB_REPETITION_PENALTY"] = float(n_conf["repetition_penalty"])
    if "no_repeat_ngram_size" in n_conf:
        globals()["NLLB_NO_REPEAT_NGRAM_SIZE"] = int(n_conf["no_repeat_ngram_size"])

    logger_func(
        f"[Config] NLLB Quality: Beams={globals().get('NLLB_NUM_BEAMS')}, "
        f"LenPen={globals().get('NLLB_LENGTH_PENALTY')}, "
        f"RepPen={globals().get('NLLB_REPETITION_PENALTY')}, "
        f"NgramBlock={globals().get('NLLB_NO_REPEAT_NGRAM_SIZE')}"
    )


def _load_type_and_model_config(config: Dict[str, Any], logger_func: Any) -> None:
    file_types = config.get("file_types")
    if isinstance(file_types, dict):
        _load_file_type_extensions(file_types, logger_func)

    models_config = config.get("models")
    if isinstance(models_config, dict):
        _load_model_identifiers(models_config)
        active_model = _get_active_translator_model_id()
        logger_func(f"[Config] Models: Translator ({TRANSLATOR_ENGINE.upper()})={active_model}, Separator={AUDIO_SEPARATOR_MODEL_ID}")


def _load_file_type_extensions(file_types_config: Dict[str, Any], logger_func: Any) -> None:
    """Load supported video extensions from config when provided."""
    exts = file_types_config.get("extensions")
    if isinstance(exts, list):
        globals()["VIDEO_EXTENSIONS"] = set(exts)
        logger_func(f"[Config] Loaded {len(exts)} video extensions.")


def _load_model_identifiers(models_config: Dict[str, Any]) -> None:
    """Apply model identifier overrides from config."""
    if "translategemma" in models_config:
        globals()["TRANSLATEGEMMA_MODEL_ID"] = models_config["translategemma"]
    if "nllb" in models_config:
        globals()["NLLB_MODEL_ID"] = models_config["nllb"]
    if "audio_separator" in models_config:
        globals()["AUDIO_SEPARATOR_MODEL_ID"] = models_config["audio_separator"]


def _get_active_translator_model_id() -> str:
    """Return currently active translator model id based on selected engine."""
    if globals().get("TRANSLATOR_ENGINE") == "translategemma":
        return str(globals().get("TRANSLATEGEMMA_MODEL_ID"))
    return str(globals().get("NLLB_MODEL_ID"))


def _reset_config_defaults() -> None:
    """Reset mutable runtime configuration state to module defaults."""
    globals()["WHISPER_MODEL_SIZE"] = "large-v3"
    globals()["INITIAL_PROMPT"] = "Transcribe the following audio file."
    globals()["USE_VOCAL_SEPARATION"] = True
    globals()["FORCED_LANGUAGE"] = None
    globals()["PROMPT_USE_CUSTOM_PRIORITY"] = False
    globals()["DEBUG_LOGGING"] = False
    globals()["HALLUCINATION_SILENCE_THRESHOLD"] = 0.9
    globals()["HALLUCINATION_REPETITION_THRESHOLD"] = 15
    globals()["HALLUCINATION_PHRASES"] = [
        "nu uitați să dați like",
        "nu uitati sa dati like",
        "să lăsați un comentariu",
        "sa lasati un comentariu",
        "să distribuiți",
        "sa distribuiti",
        "abonați-vă la canal",
        "abonati-va la canal",
        "nu uitați să vă abonați",
        "nu uitati sa va abonati",
        "pentru a nu rata videoclipurile noastre",
        "nu uitați să dați like, să lăsați un comentariu și să distribuiți acest material video pe alte rețele sociale",
        "nu uitati sa dati like, sa lasati un comentariu si sa distribuiti acest material video pe alte retele sociale",
        "nu uitați să vă abonați la canal, să vă mulțumim și la rețeta următoare",
        "abonati-va la canal, sa va multumim si la reteta urmatoare",
        "vă mulțumim pentru vizionare",
        "va multumim pentru vizionare",
        "nu uitați să apăsați butonul de like",
        "thank you for watching",
        "thanks for watching",
        "subscribe to my channel",
        "please subscribe",
        "like and subscribe",
        "hit the like button",
        "leave a comment",
        "share this video",
        "see you in the next",
        "bye bye",
        "merci d'avoir regardé",
        "n'oubliez pas de vous abonner",
        "laissez un commentaire",
        "à bientôt",
        "danke fürs zuschauen",
        "vergisst nicht zu abonnieren",
        "gracias por ver",
        "no olvides suscribirte",
        "grazie per aver guardato",
        "non dimenticare di iscriverti",
    ]
    globals()["VIDEO_EXTENSIONS"] = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".m4v", ".ts", ".mts"}
    globals()["TRANSLATOR_ENGINE"] = "nllb"
    globals()["TRANSLATEGEMMA_MODEL_ID"] = "google/translategemma-12b-it"
    globals()["NLLB_MODEL_ID"] = "facebook/nllb-200-3.3B"
    globals()["NLLB_NUM_BEAMS"] = 5
    globals()["NLLB_LENGTH_PENALTY"] = 1.0
    globals()["NLLB_REPETITION_PENALTY"] = 1.0
    globals()["NLLB_NO_REPEAT_NGRAM_SIZE"] = 0
    globals()["AUDIO_SEPARATOR_MODEL_ID"] = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
    globals()["VAD_MIN_SILENCE_MS"] = 500
    TARGET_LANGUAGES.clear()


def _handle_hf_token_prompt():
    """Handles prompting the user for their Hugging Face token and opening the license agreement."""
    # Only prompt in the main interactive process (not in worker/subprocesses)
    if not sys.stdin.isatty() or os.environ.get("IS_SUBPROCESS"):
        return

    _print_hf_token_prompt()
    _open_url_safely("https://huggingface.co/settings/tokens")

    token = input(">> Please paste your Hugging Face Access Token: ").strip()
    if not token:
        return

    os.environ["HF_TOKEN"] = token
    _print_hf_license_prompt()
    _open_url_safely(f"https://huggingface.co/{TRANSLATEGEMMA_MODEL_ID}")


def _open_url_safely(url: str) -> None:
    """Open URL in browser and print a warning instead of failing hard."""
    try:
        webbrowser.open(url)
    except OSError as e:
        print(f"[Warning] Could not open browser: {e}")


def _print_hf_token_prompt() -> None:
    """Print interactive prompt instructions for obtaining HF token."""
    print("=" * 80)
    print("[Hugging Face Token Required]")
    print("TranslateGemma is a gated model. To access it, you need a Hugging Face token.")
    print("We are automatically opening your default web browser to the token creation page:")
    print("  https://huggingface.co/settings/tokens")
    print("=" * 80)


def _print_hf_license_prompt() -> None:
    """Print instructions for accepting TranslateGemma license."""
    print("=" * 80)
    print("[Accepting Model Licensing Agreement]")
    print("We are opening the licensing agreement page for this model in your default browser:")
    print(f"  https://huggingface.co/{TRANSLATEGEMMA_MODEL_ID}")
    print("Please make sure to click 'Accept' on Hugging Face to unlock model downloads.")
    print("=" * 80)


def load_config(optimizer: Any, logger_func: Any) -> bool:
    """Loads configuration from config.yaml."""
    _reset_config_defaults()
    if hasattr(optimizer, "reset"):
        optimizer.reset()

    config_path = "config.yaml"
    if not os.path.exists(config_path):
        _load_default_languages(logger_func)
        return True

    yaml_module = _get_yaml_module()
    if yaml_module is None:
        logger_func("[Config] Missing dependency: PyYAML is required to load config.yaml.", "ERROR")
        return False

    return _load_config_from_file(config_path, yaml_module, optimizer, logger_func)


def _load_config_from_file(config_path: str, yaml_module: Any, optimizer: Any, logger_func: Any) -> bool:
    """Load and apply config values from an existing YAML file path."""
    try:
        config_data = _read_config_yaml(config_path, yaml_module)
        if isinstance(config_data, dict):
            _apply_loaded_config(config_data, optimizer, logger_func)
            return True
        logger_func("[Config] Error loading config.yaml: root document must be a mapping/object.", "ERROR")
        return False
    except (OSError, TypeError, ValueError, _yaml_error_type(yaml_module)) as e:
        logger_func(f"[Config] Error loading config.yaml: {e}", "ERROR")
        return False


def _load_default_languages(logger_func: Any) -> None:
    """Load minimal language defaults when config.yaml is absent."""
    logger_func("[Config] config.yaml not found. Using internal defaults.", "WARNING")
    TARGET_LANGUAGES.clear()
    TARGET_LANGUAGES.update(
        {
            "en": {"code": "eng_Latn", "label": "English"},
            "es": {"code": "spa_Latn", "label": "Spanish"},
            "fr": {"code": "fra_Latn", "label": "French"},
        }
    )


def _yaml_error_type(yaml_module: Any) -> type[Exception]:
    """Return YAML parse exception type for the loaded YAML module."""
    yaml_error = getattr(yaml_module, "YAMLError", ValueError)
    if isinstance(yaml_error, type) and issubclass(yaml_error, Exception):
        return yaml_error
    return ValueError


def _read_config_yaml(config_path: str, yaml_module: Any) -> dict[str, Any]:
    """Read and parse config.yaml into a dictionary."""
    with open(config_path, "r", encoding="utf-8") as file_handle:
        return yaml_module.safe_load(file_handle)


def _apply_loaded_config(config_data: dict[str, Any], optimizer: Any, logger_func: Any) -> None:
    """Apply parsed config values to runtime globals and optimizer settings."""
    _load_base_config_snippet(config_data, logger_func)
    _load_type_and_model_config(config_data, logger_func)
    _configure_hf_runtime(config_data)
    _load_optional_engine_tuning(config_data, optimizer, logger_func)


def _configure_hf_runtime(config_data: dict[str, Any]) -> None:
    """Configure Hugging Face token behavior for selected translation engine."""
    if config_data.get("hf_token"):
        os.environ["HF_TOKEN"] = str(config_data["hf_token"])

    if TRANSLATOR_ENGINE == "translategemma" and not os.environ.get("HF_TOKEN"):
        _handle_hf_token_prompt()


def _load_optional_engine_tuning(config_data: dict[str, Any], optimizer: Any, logger_func: Any) -> None:
    """Load optional NLLB/VAD/performance tuning values."""
    nllb_config = config_data.get("nllb")
    if TRANSLATOR_ENGINE == "nllb" and isinstance(nllb_config, dict):
        _load_nllb_config(nllb_config, logger_func)

    _apply_vad_settings(config_data.get("vad"), logger_func)
    _apply_performance_tuning(config_data.get("performance"), optimizer, logger_func)


def _apply_vad_settings(vad_config: Any, logger_func: Any) -> None:
    """Apply VAD silence configuration when provided."""
    if not isinstance(vad_config, dict) or "min_silence_duration_ms" not in vad_config:
        return
    globals()["VAD_MIN_SILENCE_MS"] = int(vad_config["min_silence_duration_ms"])
    logger_func(f"[Config] VAD Min Silence: {globals().get('VAD_MIN_SILENCE_MS')}ms")


def _apply_performance_tuning(performance_config: Any, optimizer: Any, logger_func: Any) -> None:
    """Apply performance overrides when present in config."""
    if performance_config and not isinstance(performance_config, dict):
        raise ValueError("performance config section must be a mapping/object")
    if performance_config:
        _load_performance_overrides(performance_config, optimizer, logger_func)
