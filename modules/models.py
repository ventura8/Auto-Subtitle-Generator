"""Models module for Auto Subtitle Generator."""

import ctypes
import gc
import importlib
import multiprocessing
import os
import shutil
import sys
import warnings

from . import config
from .utils import log

# Reduce VRAM fragmentation for Windows stability
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*expandable_segments not supported.*",
)
warnings.filterwarnings("ignore", message=".*The following generation flags are not valid.*")

# Backward-compatible placeholders referenced by tests and startup checks.
AUTO_TOKENIZER = None
AUTO_PROCESSOR = None
AUTO_MODEL_FOR_SEQ2SEQ_LM = None
AUTO_MODEL_FOR_CAUSAL_LM = None
AUTO_MODEL_FOR_IMAGE_TEXT_TO_TEXT = None
NLLB_TOKENIZER = None
WHISPER_MODEL = None

_COMPAT_EXPORTS = {
    "AutoTokenizer": "AUTO_TOKENIZER",
    "AutoProcessor": "AUTO_PROCESSOR",
    "AutoModelForSeq2SeqLM": "AUTO_MODEL_FOR_SEQ2SEQ_LM",
    "AutoModelForCausalLM": "AUTO_MODEL_FOR_CAUSAL_LM",
    "AutoModelForImageTextToText": "AUTO_MODEL_FOR_IMAGE_TEXT_TO_TEXT",
    "NllbTokenizer": "NLLB_TOKENIZER",
    "WhisperModel": "WHISPER_MODEL",
}


def __getattr__(name):
    """Expose compatibility attribute names used by tests and startup checks."""
    compat_name = _COMPAT_EXPORTS.get(name)
    if compat_name is not None:
        return globals()[compat_name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _import_optional_module(module_name):
    """Return an imported module or ``None`` when the dependency is absent."""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def _get_torch_module():
    """Return the optional torch module when available."""
    return _import_optional_module("torch")


def _get_nllb_components():
    """Return the optional NLLB tokenizer and model classes."""
    transformers_module = _import_optional_module("transformers")
    if transformers_module is None:
        return None, None
    return (
        getattr(transformers_module, "NllbTokenizer", None),
        getattr(transformers_module, "AutoModelForSeq2SeqLM", None),
    )


def _get_translategemma_components():
    """Return the optional TranslateGemma processor and model classes."""
    transformers_module = _import_optional_module("transformers")
    if transformers_module is None:
        return None, None
    return (
        getattr(transformers_module, "AutoProcessor", None),
        getattr(transformers_module, "AutoModelForImageTextToText", None),
    )


def _get_faster_whisper_components():
    """Return the optional Faster Whisper model and batching classes."""
    faster_whisper_module = _import_optional_module("faster_whisper")
    if faster_whisper_module is None:
        return None, None
    return (
        getattr(faster_whisper_module, "WhisperModel", None),
        getattr(faster_whisper_module, "BatchedInferencePipeline", None),
    )


def _get_separator_class():
    """Return the optional audio separator class."""
    separator_module = _import_optional_module("audio_separator.separator")
    if separator_module is None:
        return None
    return getattr(separator_module, "Separator", None)


def _prepare_whisper_cuda13_runtime():
    """Ensure CUDA 13 BLAS DLL is discoverable before Faster-Whisper GPU init."""
    site_packages = os.path.join(sys.prefix, "Lib", "site-packages")
    candidate_paths = [
        os.path.join(site_packages, "torch", "lib"),
        os.path.join(site_packages, "nvidia", "cu13", "bin"),
        os.path.join(site_packages, "nvidia", "cublas", "bin"),
        os.path.join(site_packages, "nvidia", "cudnn", "bin"),
    ]

    raw_path = os.environ.get("PATH", "")
    known = {os.path.normcase(os.path.normpath(entry)) for entry in raw_path.split(os.pathsep) if entry}

    for path in candidate_paths:
        if not os.path.isdir(path):
            continue

        normalized = os.path.normcase(os.path.normpath(path))
        if normalized not in known:
            os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
            known.add(normalized)

        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(path)
            except OSError:
                pass

    _ensure_cuda13_cublas_compat_alias(candidate_paths)

    try:
        ctypes.CDLL("cublas64_13.dll")
    except OSError as e:
        raise RuntimeError(
            "CUDA 13 BLAS runtime missing for Faster-Whisper GPU mode: cublas64_13.dll not loadable. "
            "Run install_dependencies.ps1 to repair the CUDA 13 runtime stack."
        ) from e


def _ensure_cuda13_cublas_compat_alias(candidate_paths):
    """Create a local cublas64_12 alias to CUDA 13 BLAS for CT2 compatibility."""
    dll12 = "cublas64_12.dll"
    dll13 = "cublas64_13.dll"

    for path in candidate_paths:
        if os.path.exists(os.path.join(path, dll12)):
            return

    for path in candidate_paths:
        source = os.path.join(path, dll13)
        target = os.path.join(path, dll12)
        if not os.path.exists(source) or os.path.exists(target):
            continue

        try:
            shutil.copy2(source, target)
            log(
                "[Whisper] Added CUDA BLAS compatibility alias cublas64_12.dll -> cublas64_13.dll",
                "WARNING",
            )
            return
        except OSError:
            continue


def _disable_default_max_length(model):
    """Avoid max_length/max_new_tokens conflict from inherited generation config."""
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None and getattr(generation_config, "max_length", None) is not None:
        generation_config.max_length = None


def _sanitize_generation_kwargs(gen_params):
    """Remove generation args that conflict with dynamic max_new_tokens usage."""
    needs_copy = "max_length" in gen_params
    early_stopping = gen_params.get("early_stopping")
    num_beams = gen_params.get("num_beams", 1)

    # Transformers validates early_stopping only when beam search is active.
    if early_stopping is not None:
        try:
            if int(num_beams) <= 1:
                needs_copy = True
        except (TypeError, ValueError):
            pass

    if not needs_copy:
        return gen_params

    sanitized = dict(gen_params)
    sanitized.pop("max_length", None)

    try:
        if int(sanitized.get("num_beams", 1)) <= 1:
            sanitized.pop("early_stopping", None)
    except (TypeError, ValueError):
        pass

    return sanitized


def _normalize_translategemma_lang_code(lang_code):
    """Normalize language codes to variants expected by TranslateGemma templates."""
    if lang_code is False:
        return "nb"
    if not isinstance(lang_code, str):
        return "en"
    if lang_code == "no":
        return "nb"
    return lang_code


def _select_bf16_dtype(torch_module):
    """Return bfloat16 only when CUDA BF16 support is available."""
    if torch_module is None or not hasattr(torch_module, "cuda"):
        return None
    if not torch_module.cuda.is_available():
        return None
    if not hasattr(torch_module.cuda, "is_bf16_supported"):
        return None
    return torch_module.bfloat16 if torch_module.cuda.is_bf16_supported() else None


# NOTE: do not create module-level lazy names like `torch`/`NllbTokenizer`
# as they cause redefinition and naming-style lint failures. Heavy
# optional dependencies are imported locally where used.


class Segment:
    """Represents a subtitle segment with timing and text."""

    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text

    def duration(self):
        """Return the segment duration in seconds."""
        return self.end - self.start

    def as_tuple(self):
        """Return the segment as a tuple for callers that need a stable shape."""
        return self.start, self.end, self.text


class SystemOptimizer:
    """Auto-detects hardware and applies optimal settings for AI workloads."""

    def __init__(self):
        self.profile = "STANDARD"
        self.vram_gb = 0
        self.ram_gb = 0
        try:
            self.cpu_cores = int(multiprocessing.cpu_count())
        except (TypeError, ValueError):
            self.cpu_cores = 1

        self.gpu_name = "None"
        self.config = {
            "whisper_beam": 5,
            "whisper_compute": "float16",
            "whisper_workers": 1,
            "nllb_batch": 16,
            "ffmpeg_threads": max(1, self.cpu_cores - 2),
            "device": "cpu",
        }

    def detect_hardware(self, verbose=True):
        """Probes system for GPU/VRAM and assigns performance profile."""
        if verbose:
            log("[Auto-Detect] Scanning Hardware...")
            log(f"[Auto-Detect] CPU Cores: {self.cpu_cores}")

        self._detect_gpu(verbose=verbose)
        self._assign_profile(verbose=verbose)

        return self.config

    def _detect_gpu(self, verbose=True):
        """Internal helper to detect GPU and VRAM."""
        try:
            # We assume torch is imported/available by the time this runs
            torch = _get_torch_module()
            if torch is None:
                raise ImportError

            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                try:
                    mem = float(props.total_memory)
                except (TypeError, ValueError):
                    mem = 0.0
                self.vram_gb = round(mem / (1024**3), 2)
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

    def _assign_profile(self, verbose=True):
        """Assigns performance profile based on detected hardware."""
        if self.config["device"] != "cuda":
            self.set_profile("CPU_ONLY", verbose=verbose)
            return

        try:
            vram = float(self.vram_gb)
        except (TypeError, ValueError):
            vram = 0.0

        if vram >= 22:
            profile = "ULTRA"
        elif vram >= 15:
            profile = "HIGH"
        elif vram >= 10:
            profile = "MID"
        else:
            profile = "LOW"

        self.set_profile(profile, verbose=verbose)

    def _calculate_batch_sizes(self, profile_name):
        """Calculates dynamic batch sizes based on profile and VRAM."""
        # 1. Calculate Target VRAM (Proportional Scaling)
        # We usage Total VRAM - 4GB (Safety Buffer for Windows/Display)
        target_vram = max(4.0, self.vram_gb - 4.0)

        # 2. Dynamic NLLB Scaling
        nllb_overhead = 8.1
        nllb_per_item = 0.40 if config.NLLB_NUM_BEAMS <= 5 else 0.80
        dynamic_nllb_batch = max(1, int((target_vram - nllb_overhead) / nllb_per_item))

        profile_caps = {"ULTRA": 32, "HIGH": 16, "MID": 8, "LOW": 4, "CPU_ONLY": 1}
        max_limit = profile_caps.get(profile_name, 4)
        dynamic_nllb_batch = min(dynamic_nllb_batch, max_limit)

        # 3. Dynamic Whisper Scaling
        wh_overhead = 3.1
        wh_per_item = 0.6
        dynamic_whisper_batch = max(1, int((target_vram - wh_overhead) / wh_per_item))

        # 4. Worker Scaling
        if self.vram_gb >= 24:
            whisper_workers = 10 if profile_name == "ULTRA" else 5
        elif self.vram_gb >= 10:
            whisper_workers = 4
        else:
            whisper_workers = 1

        return dynamic_nllb_batch, dynamic_whisper_batch, whisper_workers

    def set_profile(self, profile_name, verbose=True):
        """Applies a named performance profile."""
        valid_profiles = ["ULTRA", "HIGH", "MID", "LOW", "CPU_ONLY"]
        if profile_name not in valid_profiles:
            if verbose:
                log(f"[Warning] Invalid profile '{profile_name}'. Defaulting to STANDARD.")
            profile_name = "STANDARD"

        self.profile = profile_name
        if verbose and profile_name != "STANDARD":
            log(f"[Optimization] Applied Profile: {profile_name}")

        dyn_nllb, dyn_whisper, wh_workers = self._calculate_batch_sizes(profile_name)

        profiles = {
            "ULTRA": {
                "whisper_beam": 5,
                "whisper_compute": "float16",
                "whisper_workers": wh_workers,
                "whisper_batch_size": 1,  # FORCED: Sequential for Max Accuracy
                "nllb_batch": dyn_nllb,
                "ffmpeg_threads": self.cpu_cores,
            },
            "HIGH": {
                "whisper_beam": 5,
                "whisper_compute": "float16",
                "whisper_workers": max(1, wh_workers // 2),
                "whisper_batch_size": max(1, dyn_whisper // 2),
                "nllb_batch": max(1, dyn_nllb // 2),
                "ffmpeg_threads": self.cpu_cores,
            },
            "MID": {"whisper_beam": 5, "whisper_workers": 1, "nllb_batch": dyn_nllb},
            "LOW": {"whisper_beam": 5, "nllb_batch": 1, "whisper_compute": "int8_float16"},
            "CPU_ONLY": {
                "whisper_beam": 5,
                "whisper_compute": "int8",
                "nllb_batch": 1,
                "ffmpeg_threads": max(1, self.cpu_cores - 2),
            },
        }
        if profile_name in profiles:
            # Apply profile defaults
            profile_cfg = profiles[profile_name]

            # CRITICAL: Allow user overrides from config.yaml to persist!
            # We only apply profile defaults if the key is not already in self.config
            # OR if it's the default value we want to'over-tune'.
            # Specifically for whisper_beam, we check if it was manually set.
            for k, v in profile_cfg.items():
                # If the user hasn't explicitly overridden this in config.yaml, use profile default
                # (Assuming 'None' or the init default means 'not overridden')
                if k == "whisper_beam" and self.config.get("whisper_beam_overridden"):
                    continue
                self.config[k] = v
            if verbose and profile_name in ["ULTRA", "HIGH", "MID"]:
                msg = f"[Optimization] Dynamic NLLB batch size: {self.config['nllb_batch']} (based on {self.vram_gb}GB VRAM)"
                log(msg)


# Create global optimizer instance
OPTIMIZER = SystemOptimizer()


class NLLBTranslator:
    """Wrapper for NLLB-200 translation model."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._load()

    def _load(self):
        # Local imports for optional heavy dependencies
        nllb_tokenizer_cls, auto_model_for_seq2seq_lm_cls = _get_nllb_components()
        torch = _get_torch_module()

        # Basic validation
        if nllb_tokenizer_cls is None or auto_model_for_seq2seq_lm_cls is None:
            raise RuntimeError("transformers NLLB components not available")

        # Configure tokenizer and delegate heavy loading to helper
        self.tokenizer = nllb_tokenizer_cls.from_pretrained(config.NLLB_MODEL_ID)
        if torch is not None:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            log(f"[Load] Initializing NLLB (Torch {torch.__version__})", level="DEBUG")

        # Delegate the model instantiation to reduce function complexity
        self._perform_nllb_load(auto_model_for_seq2seq_lm_cls, torch)

    def _perform_nllb_load(self, auto_model_cls, torch):
        """Helper: perform the actual NLLB model loading (extracted)."""
        gc.collect()
        if torch is not None:
            try:
                torch.cuda.empty_cache()
            except (AttributeError, RuntimeError):
                pass

        dtype = _select_bf16_dtype(torch)

        if OPTIMIZER.config["device"] == "cuda":
            target_device = "cuda:0"
            device_map = {"": 0}
        else:
            target_device = "cpu"
            device_map = None

        try:
            try:
                self.model = auto_model_cls.from_pretrained(
                    config.NLLB_MODEL_ID,
                    dtype=dtype,
                    low_cpu_mem_usage=True,
                    attn_implementation="eager",
                    tie_word_embeddings=True,
                    device_map=device_map,
                )
            except (OSError, ValueError, RuntimeError) as net_err:
                log(f"[Load] Network/Load error ({net_err}). Trying local_files_only...", "WARNING")
                self.model = auto_model_cls.from_pretrained(
                    config.NLLB_MODEL_ID,
                    dtype=dtype,
                    low_cpu_mem_usage=True,
                    attn_implementation="eager",
                    tie_word_embeddings=True,
                    device_map=device_map,
                    local_files_only=True,
                )

            if device_map is None:
                self.model.to(target_device)

            _disable_default_max_length(self.model)
            self.model.tie_weights()
            log(f"[Load] NLLB loaded in {self.model.dtype} (Native Weight Tying).", level="DEBUG")
        except (OSError, RuntimeError, ValueError) as e:
            log(f"[Load] CRITICAL LOAD ERROR: {e}")
            raise

        # Warm-up to allocate buffers
        if OPTIMIZER.config["device"] == "cuda" and torch is not None:
            log("[Load] Warming up NLLB...", level="DEBUG")
            dummy = self.tokenizer("Hello world", return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                self.model.generate(**dummy, max_new_tokens=1)

    def translate(self, texts, src_lang_code, tgt_lang_code, **gen_kwargs):
        """Translates a batch of texts using verified Native NLLB logic."""
        if not self.model or not texts:
            return texts

        torch = _get_torch_module()
        if torch is None:
            raise RuntimeError("torch is required for NLLB translation")
        log(f"  [AI] Input[0] Repr: {repr(texts[0])}", level="DEBUG")

        # 1. Tokenize (Native)
        # Explicit NllbTokenizer automatically handles [src_lang, text, EOS]
        self.tokenizer.src_lang = src_lang_code
        self.tokenizer.tgt_lang = tgt_lang_code

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.model.device)

        input_len = inputs.input_ids.shape[1]
        log(f"  [AI] Native Tokens: {inputs.input_ids[0].tolist()[:10]}... (Len: {input_len})", level="DEBUG")

        # 2. High-Quality Generation Settings (Dynamic from config)
        tgt_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang_code)

        gen_params = {
            "num_beams": config.NLLB_NUM_BEAMS,
            "length_penalty": config.NLLB_LENGTH_PENALTY,
            "repetition_penalty": config.NLLB_REPETITION_PENALTY,
            "no_repeat_ngram_size": config.NLLB_NO_REPEAT_NGRAM_SIZE,
            "early_stopping": True,
            "do_sample": False,
            "use_cache": True,
        }
        gen_params.update(gen_kwargs)

        with torch.inference_mode():
            # Stop Rambling Hallucinations
            dynamic_max = min(512, int(input_len * 3) + 20)
            gen_params["forced_bos_token_id"] = tgt_token_id
            gen_params["max_new_tokens"] = dynamic_max
            gen_params = _sanitize_generation_kwargs(gen_params)
            gen_params = {**inputs, **gen_params}

            log(f"  [AI] Generation: Native-NLLB (Beam-5, Native-EO) | Max={dynamic_max}", level="DEBUG")

            generated_tokens = self.model.generate(**gen_params)

        return self.tokenizer.batch_decode(generated_tokens.cpu(), skip_special_tokens=True)

    def offload(self):
        """Moves model to CPU and clears cache."""
        if self.model:
            log("  [AI] Offloading NLLB to CPU...")
            self.model.to("cpu")
            torch = _get_torch_module()
            if torch is not None:
                torch.cuda.empty_cache()
            gc.collect()


class TranslateGemmaTranslator:
    """Wrapper for Google TranslateGemma translation model."""

    def __init__(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._load()

    def _load(self):
        # Local imports
        auto_processor_cls, auto_model_for_image_text_to_text_cls = _get_translategemma_components()
        torch = _get_torch_module()

        if auto_processor_cls is None or auto_model_for_image_text_to_text_cls is None:
            raise RuntimeError("transformers TranslateGemma components not available")

        # Disable TF32 when torch is present
        if torch is not None:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

        # GC/Cache clear
        gc.collect()
        if torch is not None:
            try:
                torch.cuda.empty_cache()
            except (AttributeError, RuntimeError):
                pass

        # Configure processor
        self.processor = auto_processor_cls.from_pretrained(config.TRANSLATEGEMMA_MODEL_ID)
        # Keep tokenizer alias for backward compatibility in tests/callers.
        self.tokenizer = self.processor
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"
            if self.processor.tokenizer.pad_token is None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        # Delegate heavy load to helper to reduce complexity
        self._perform_translategemma_load(auto_model_for_image_text_to_text_cls, torch)

        # Warm-up to allocate buffers
        if OPTIMIZER.config["device"] == "cuda" and torch is not None:
            log("[Load] Warming up TranslateGemma...", level="DEBUG")
            dummy = self._build_translate_gemma_inputs(["Hello"], "en", "es")
            with torch.no_grad():
                self.model.generate(**dummy, max_new_tokens=1)

    def translate(self, texts, src_lang_code, tgt_lang_code, **gen_kwargs):
        """Translates a batch of texts using TranslateGemma logic."""
        if not self.model or not texts:
            return texts

        torch = _get_torch_module()
        if torch is None:
            raise RuntimeError("torch is required for TranslateGemma translation")
        log(f"  [AI] Input[0] Repr: {repr(texts[0])}", level="DEBUG")

        # Map NLLB codes back to ISO 639-1 if they are NLLB codes
        src_iso = _normalize_translategemma_lang_code(config.nllb_to_iso(src_lang_code))
        tgt_iso = _normalize_translategemma_lang_code(config.nllb_to_iso(tgt_lang_code))

        inputs = self._build_translate_gemma_inputs(texts, src_iso, tgt_iso)

        input_len = inputs.input_ids.shape[1]
        log(f"  [AI] Native Tokens: {inputs.input_ids[0].tolist()[:10]}... (Len: {input_len})", level="DEBUG")

        gen_params = self._build_translategemma_gen_params(gen_kwargs)

        with torch.inference_mode():
            # Stop Rambling Hallucinations
            dynamic_max = min(512, int(input_len * 3) + 20)
            gen_params["max_new_tokens"] = dynamic_max
            gen_params = _sanitize_generation_kwargs(gen_params)
            gen_params = {**inputs, **gen_params}

            log(f"  [AI] Generation: TranslateGemma | Max={dynamic_max}", level="DEBUG")

            response_tokens = self.model.generate(**gen_params)[:, input_len:]

        processor = self._get_translategemma_processor()
        return processor.batch_decode(response_tokens.cpu(), skip_special_tokens=True)

    def _get_translategemma_processor(self):
        """Return the active processor for TranslateGemma prompt/token handling."""
        return self.processor if self.processor is not None else self.tokenizer

    def _build_translategemma_gen_params(self, gen_kwargs):
        """Build generation kwargs for TranslateGemma decoding."""
        processor = self._get_translategemma_processor()
        tokenizer = getattr(processor, "tokenizer", processor)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)

        gen_params = {
            "early_stopping": True,
            "do_sample": False,
            "use_cache": True,
            "pad_token_id": pad_token_id,
        }
        gen_params.update(gen_kwargs)
        return gen_params

    def _build_translate_gemma_inputs(self, texts, src_iso, tgt_iso):
        """Build prompts and tokenized inputs for TranslateGemma generation."""
        processor = self._get_translategemma_processor()
        prompts = [self._build_translategemma_prompt(text, src_iso, tgt_iso) for text in texts]

        if hasattr(processor, "tokenizer"):
            processor.tokenizer.padding_side = "left"
            if processor.tokenizer.pad_token is None:
                processor.tokenizer.pad_token = processor.tokenizer.eos_token

        return processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(self.model.device)

    def _build_translategemma_prompt(self, text, src_iso, tgt_iso):
        """Build one TranslateGemma prompt with a safe fallback path."""
        processor = self._get_translategemma_processor()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": src_iso,
                        "target_lang_code": tgt_iso,
                        "text": text,
                    }
                ],
            }
        ]
        try:
            prompt = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception as template_err:
            raise ValueError(f"TranslateGemma prompt build failed for src={src_iso}, tgt={tgt_iso}: {template_err}") from template_err

        if prompt is None:
            prompt = f"Translate from {src_iso} to {tgt_iso}: {text}"

        return prompt

    def offload(self):
        """Moves model to CPU and clears cache."""
        if self.model:
            log("  [AI] Offloading TranslateGemma to CPU...")
            self.model.to("cpu")
            torch = _get_torch_module()
            if torch is not None:
                torch.cuda.empty_cache()
            gc.collect()

    def _perform_translategemma_load(self, auto_model_cls, torch):
        """Helper: perform the heavy TranslateGemma model loading."""
        try:
            dtype = _select_bf16_dtype(torch)
            log(f"[Load] Loading TranslateGemma (12B) in {dtype}...", level="DEBUG")

            if OPTIMIZER.config["device"] == "cuda":
                target_device = "cuda:0"
                device_map = {"": 0}
            else:
                target_device = "cpu"
                device_map = None

            try:
                self.model = auto_model_cls.from_pretrained(
                    config.TRANSLATEGEMMA_MODEL_ID,
                    dtype=dtype,
                    low_cpu_mem_usage=True,
                    device_map=device_map,
                )
            except (OSError, ValueError, RuntimeError) as net_err:
                log(f"[Load] Network/Load error ({net_err}). Trying local_files_only...", "WARNING")
                self.model = auto_model_cls.from_pretrained(
                    config.TRANSLATEGEMMA_MODEL_ID,
                    dtype=dtype,
                    low_cpu_mem_usage=True,
                    device_map=device_map,
                    local_files_only=True,
                )

            if device_map is None:
                self.model.to(target_device)

            _disable_default_max_length(self.model)
            log(f"[Load] TranslateGemma loaded in {self.model.dtype}.", level="DEBUG")

        except (OSError, RuntimeError, ValueError) as e:
            err_msg = str(e)
            lower = err_msg.lower()
            if "gated repo" in lower or "401 client error" in lower or "restricted" in lower or "unauthorized" in lower:
                log("=" * 80, "ERROR")
                log("[GATED MODEL ERROR] TranslateGemma is gated on Hugging Face.", "ERROR")
                log(f"Model ID: {config.TRANSLATEGEMMA_MODEL_ID}", "ERROR")
                log("To resolve this, either:", "ERROR")
                log(
                    f"1) Visit https://huggingface.co/{config.TRANSLATEGEMMA_MODEL_ID} and accept the license",
                    "ERROR",
                )
                log(
                    "2) Authenticate via `huggingface-cli login` or set HF_TOKEN env var",
                    "ERROR",
                )
                log(
                    'Or set `translation.engine: "nllb"` in config.yaml to use NLLB.',
                    "ERROR",
                )
                log("=" * 80, "ERROR")
            log(f"[Load] CRITICAL LOAD ERROR: {e}")
            raise


class ModelManager:
    """Lazy loader and manager for AI models."""

    def __init__(self):
        self._whisper = None
        self._nllb = None
        self._separator = None
        self._whisper_base = None

    def get_whisper(self):
        """Return the lazily loaded Whisper model or batching pipeline."""
        if self._whisper is None:
            log("[AI] Loading Whisper")
            whisper_model_cls, batched_pipeline_cls = _get_faster_whisper_components()
            if whisper_model_cls is None or batched_pipeline_cls is None:
                raise RuntimeError("faster_whisper is not installed")
            if OPTIMIZER.config.get("device") == "cuda":
                _prepare_whisper_cuda13_runtime()
            model = whisper_model_cls(
                config.WHISPER_MODEL_SIZE,
                device=OPTIMIZER.config["device"],
                compute_type=OPTIMIZER.config["whisper_compute"],
                num_workers=OPTIMIZER.config["whisper_workers"],
            )

            # Wrap in batching pipeline if configured
            batch_size = OPTIMIZER.config.get("whisper_batch_size", 1)
            if batch_size > 1:
                log(f"[AI] Whisper Batching Enabled (Batch Size: {batch_size})")
                self._whisper = batched_pipeline_cls(model)
                self._whisper_base = model  # Keep reference for offloading
            else:
                self._whisper = model
                self._whisper_base = model

        return self._whisper

    def get_nllb(self):
        """Return the configured translator instance."""
        return self.get_translator()

    def get_translator(self):
        """Return the lazily loaded translation backend."""
        if self._nllb is None:
            # PROACTIVE: Clear all other models from VRAM before loading translator
            log(f"[AI] Clearing memory for {config.TRANSLATOR_ENGINE} (High-Perf Profiling)...", level="DEBUG")
            self.offload_whisper()
            self.offload_separator()

            if config.TRANSLATOR_ENGINE == "translategemma":
                log("[AI] Loading TranslateGemma...", level="DEBUG")
                self._nllb = TranslateGemmaTranslator()
            else:
                log("[AI] Loading NLLB...", level="DEBUG")
                self._nllb = NLLBTranslator()
        return self._nllb

    def get_separator(self, output_dir=None):
        """Return the lazily loaded audio separator backend."""
        if self._separator is None:
            log("[AI] Loading Audio Separator...")
            separator_cls = _get_separator_class()
            if separator_cls is None:
                raise RuntimeError("audio_separator is not installed")
            self._separator = separator_cls(
                model_file_dir=os.path.join(os.getcwd(), "models"),
                output_dir=output_dir if output_dir else os.getcwd(),
                output_single_stem="Vocals",
            )
            self._separator.load_model(model_filename=config.AUDIO_SEPARATOR_MODEL_ID)
        else:
            if output_dir:
                self._separator.output_dir = output_dir
            self._separator.output_single_stem = "Vocals"
        return self._separator

    def offload_whisper(self):
        """Frees Whisper VRAM."""
        if self._whisper:
            log("  [AI] Offloading Whisper to CPU...")
            # faster-whisper doesn't have a simple .to('cpu')
            # like torch, but we can delete and GC
            del self._whisper
            if hasattr(self, "_whisper_base"):
                del self._whisper_base
            self._whisper = None
            self._whisper_base = None
            torch = _get_torch_module()
            if torch is not None:
                torch.cuda.empty_cache()
            gc.collect()

    def offload_separator(self):
        """Frees Audio Separator VRAM."""
        if self._separator:
            log("  [AI] Offloading Audio Separator...")
            # Audio Separator handles its own cleanup usually,
            # but let's be safe
            del self._separator
            self._separator = None
            torch = _get_torch_module()
            if torch is not None:
                torch.cuda.empty_cache()
            gc.collect()

    def offload_nllb(self):
        """Frees NLLB VRAM."""
        if self._nllb:
            self._nllb.offload()

    def preload_nllb(self):
        """Optional preloading for checking OOM early."""
        self.get_nllb()
