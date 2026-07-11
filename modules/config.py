"""
Configuration module for Auto Subtitle Generator.
Handles loading settings from config.yaml and prompts.yaml.
"""

import os
import sys
import re
import importlib
from typing import Dict, Any
import warnings
import webbrowser


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
    "lv": "lav_Latn",
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
    "ta": "tam_Tamil",
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
    # First check TARGET_LANGUAGES configuration
    for iso, info in TARGET_LANGUAGES.items():
        if info.get("code") == code:
            return iso
    # Then check static mapping
    for iso, nllb in ISO_TO_NLLB.items():
        if nllb == code:
            return iso
    # Handshake map for 3-letter prefixes to 2-letter ISO codes
    if "_" in code:
        prefix = code.split("_")[0]
        mapping = {
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
        return mapping.get(prefix, prefix[:2])
    return code


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
        logger_func(f"[Config] Performance Overrides: {', '.join(updated_keys)}")


def _load_base_config_snippet(config: Dict[str, Any], logger_func: Any) -> None:
    if "debug_logging" in config:
        globals()["DEBUG_LOGGING"] = config["debug_logging"]

    if "translation" in config and isinstance(config["translation"], dict):
        t_conf = config["translation"]
        if "engine" in t_conf:
            globals()["TRANSLATOR_ENGINE"] = str(t_conf["engine"]).lower()
            logger_func(f"[Config] Translation Engine: {globals().get('TRANSLATOR_ENGINE')}")

    if "target_languages" in config:
        TARGET_LANGUAGES.update(_normalize_target_languages(config["target_languages"], logger_func))
        logger_func(f"[Config] Loaded {len(TARGET_LANGUAGES)} languages from config.")

    if "whisper" in config:
        _load_whisper_config(config["whisper"], logger_func)

    if "hallucinations" in config:
        _load_hallucination_config(config["hallucinations"], logger_func)


def _normalize_target_languages(raw_languages: Dict[Any, Any], logger_func: Any) -> Dict[str, Dict[str, str]]:
    """Normalize target language keys loaded from YAML to stable ISO strings."""
    normalized: Dict[str, Dict[str, str]] = {}
    if not isinstance(raw_languages, dict):
        return normalized

    for raw_key, lang_info in raw_languages.items():
        if not isinstance(lang_info, dict):
            continue

        lang_key = raw_key
        if isinstance(raw_key, bool):
            lang_key = "no" if raw_key is False else "yes"
            logger_func(f"[Config] Normalized boolean language key '{raw_key}' to '{lang_key}'.", "WARNING")
        elif not isinstance(raw_key, str):
            lang_key = str(raw_key)

        normalized[str(lang_key)] = lang_info

    return normalized


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

    if "file_types" in config and "extensions" in config["file_types"]:
        exts = config["file_types"]["extensions"]
        if exts:
            globals()["VIDEO_EXTENSIONS"] = set(exts)
            logger_func(f"[Config] Loaded {len(globals().get('VIDEO_EXTENSIONS'))} video extensions.")

    if "models" in config:
        m_conf = config["models"]
        if "translategemma" in m_conf:
            globals()["TRANSLATEGEMMA_MODEL_ID"] = m_conf["translategemma"]
        if "nllb" in m_conf:
            globals()["NLLB_MODEL_ID"] = m_conf["nllb"]
        if "audio_separator" in m_conf:
            globals()["AUDIO_SEPARATOR_MODEL_ID"] = m_conf["audio_separator"]
        active_model = (
            globals().get("TRANSLATEGEMMA_MODEL_ID")
            if globals().get("TRANSLATOR_ENGINE") == "translategemma"
            else globals().get("NLLB_MODEL_ID")
        )
        logger_func(f"[Config] Models: Translator ({TRANSLATOR_ENGINE.upper()})={active_model}, Separator={AUDIO_SEPARATOR_MODEL_ID}")


def _save_token_to_config(token):
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        return

    yaml_module = _get_yaml_module()
    if yaml_module is None:
        print("[Warning] PyYAML is not available; could not persist token to config.yaml.")
        return

    try:
        with open(config_path, "r", encoding="utf-8") as file_handle:
            config_text = file_handle.read()

        escaped_token = str(token).replace("\\", "\\\\").replace('"', '\\"')
        token_line = f'hf_token: "{escaped_token}"'

        token_line_pattern = re.compile(r"^(\s*hf_token\s*:\s*)([^#\n]*)(\s*(?:#.*)?)$", re.MULTILINE)
        if token_line_pattern.search(config_text):
            updated_text = token_line_pattern.sub(
                lambda match: f'{match.group(1)}"{escaped_token}"{match.group(3)}',
                config_text,
                count=1,
            )
        else:
            separator = "\n" if config_text and not config_text.endswith("\n") else ""
            updated_text = f"{config_text}{separator}{token_line}\n"

        with open(config_path, "w", encoding="utf-8") as file_handle:
            file_handle.write(updated_text)

        print("[Config] Token successfully saved to config.yaml for future runs.")
    except (OSError, TypeError, ValueError, _yaml_error_type(yaml_module)) as e:
        print(f"[Warning] Failed to save token to config.yaml: {e}")


def _handle_hf_token_prompt():
    """Handles prompting the user for their Hugging Face token and opening the license agreement."""
    # Only prompt in the main interactive process (not in worker/subprocesses)
    if sys.stdin.isatty() and not os.environ.get("IS_SUBPROCESS"):
        print("=" * 80)
        print("[Hugging Face Token Required]")
        print("TranslateGemma is a gated model. To access it, you need a Hugging Face token.")
        print("We are automatically opening your default web browser to the token creation page:")
        print("  https://huggingface.co/settings/tokens")
        print("=" * 80)

        try:
            webbrowser.open("https://huggingface.co/settings/tokens")
        except OSError as e:
            print(f"[Warning] Could not open browser: {e}")

        token = input(">> Please paste your Hugging Face Access Token: ").strip()
        if token:
            os.environ["HF_TOKEN"] = token
            _save_token_to_config(token)

            print("=" * 80)
            print("[Accepting Model Licensing Agreement]")
            print("We are opening the licensing agreement page for this model in your default browser:")
            print(f"  https://huggingface.co/{TRANSLATEGEMMA_MODEL_ID}")
            print("Please make sure to click 'Accept' on Hugging Face to unlock model downloads.")
            print("=" * 80)

            try:
                webbrowser.open(f"https://huggingface.co/{TRANSLATEGEMMA_MODEL_ID}")
            except OSError as e:
                print(f"[Warning] Could not open browser: {e}")


def load_config(optimizer: Any, logger_func: Any) -> bool:
    """Loads configuration from config.yaml."""

    config_path = "config.yaml"
    if not os.path.exists(config_path):
        _load_default_languages(logger_func)
        return True

    yaml_module = _get_yaml_module()
    if yaml_module is None:
        logger_func("[Config] Missing dependency: PyYAML is required to load config.yaml.", "ERROR")
        return False

    try:
        config_data = _read_config_yaml(config_path, yaml_module)
        _apply_loaded_config(config_data, optimizer, logger_func)
        return True
    except (OSError, TypeError, ValueError, _yaml_error_type(yaml_module)) as e:
        logger_func(f"[Config] Error loading config.yaml: {e}", "ERROR")
        return False


def _load_default_languages(logger_func: Any) -> None:
    """Load minimal language defaults when config.yaml is absent."""
    logger_func("[Config] config.yaml not found. Using internal defaults.", "WARNING")
    TARGET_LANGUAGES.update(
        {
            "en": {"code": "eng_Latn", "label": "English"},
            "es": {"code": "spa_Latn", "label": "Spanish"},
            "fr": {"code": "fra_Latn", "label": "French"},
        }
    )


def _yaml_error_type(yaml_module: Any) -> type:
    """Return YAML parse exception type for the loaded YAML module."""
    return getattr(yaml_module, "YAMLError", ValueError)


def _read_config_yaml(config_path: str, yaml_module: Any) -> dict[str, Any]:
    """Read and parse config.yaml into a dictionary."""
    with open(config_path, "r", encoding="utf-8") as file_handle:
        return yaml_module.safe_load(file_handle)


def _apply_loaded_config(config_data: dict[str, Any], optimizer: Any, logger_func: Any) -> None:
    """Apply parsed config values to runtime globals and optimizer settings."""
    _load_base_config_snippet(config_data, logger_func)
    _load_type_and_model_config(config_data, logger_func)

    # Ignore Hugging Face Hub unauthenticated warning only for NLLB engine
    if TRANSLATOR_ENGINE == "nllb":
        warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")

    # Load Hugging Face token if present
    if "hf_token" in config_data and config_data["hf_token"]:
        os.environ["HF_TOKEN"] = str(config_data["hf_token"])

    # Check if HF_TOKEN is missing when TranslateGemma is selected
    if TRANSLATOR_ENGINE == "translategemma" and not os.environ.get("HF_TOKEN"):
        _handle_hf_token_prompt()

    if TRANSLATOR_ENGINE == "nllb" and "nllb" in config_data and isinstance(config_data["nllb"], dict):
        _load_nllb_config(config_data["nllb"], logger_func)

    if "vad" in config_data and "min_silence_duration_ms" in config_data["vad"]:
        globals()["VAD_MIN_SILENCE_MS"] = int(config_data["vad"]["min_silence_duration_ms"])
        logger_func(f"[Config] VAD Min Silence: {globals().get('VAD_MIN_SILENCE_MS')}ms")

    if "performance" in config_data:
        _load_performance_overrides(config_data["performance"], optimizer, logger_func)
