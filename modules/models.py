"""
Models module for Auto Subtitle Generator.
Contains various classes for optimization and model management.
"""
import multiprocessing
import os
from . import config
from .utils import log

# Reduce VRAM fragmentation for Windows stability
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Lazy Handles
torch = None
AutoTokenizer = None
AutoModelForSeq2SeqLM = None
NllbTokenizer = None
WhisperModel = None


class Segment:

    """Represents a subtitle segment with timing and text."""

    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text


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
            "device": "cpu"
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
            import torch
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
                    log(
                        f"[Auto-Detect] GPU Detected: {props.name} "
                        f"({self.vram_gb} GB VRAM)"
                    )
            else:
                self.config["device"] = "cpu"
                if verbose:
                    log(
                        "[Auto-Detect] No CUDA GPU found. Falling back to CPU."
                    )
        except ImportError:
            if verbose:
                log(
                    "[Auto-Detect] Torch not loaded yet, assuming CPU for now."
                )

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
        dynamic_nllb_batch = max(
            1, int((target_vram - nllb_overhead) / nllb_per_item)
        )

        profile_caps = {
            "ULTRA": 32, "HIGH": 16, "MID": 8, "LOW": 4, "CPU_ONLY": 1
        }
        max_limit = profile_caps.get(profile_name, 4)
        dynamic_nllb_batch = min(dynamic_nllb_batch, max_limit)

        # 3. Dynamic Whisper Scaling
        wh_overhead = 3.1
        wh_per_item = 0.6
        dynamic_whisper_batch = max(
            1, int((target_vram - wh_overhead) / wh_per_item)
        )

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

        dyn_nllb, dyn_whisper, wh_workers = self._calculate_batch_sizes(
            profile_name
        )

        profiles = {
            "ULTRA": {
                "whisper_beam": 5,
                "whisper_compute": "float16",
                "whisper_workers": wh_workers,
                "whisper_batch_size": 1,  # FORCED: Sequential for Max Accuracy
                "nllb_batch": dyn_nllb,
                "ffmpeg_threads": self.cpu_cores
            },
            "HIGH": {
                "whisper_beam": 5,
                "whisper_compute": "float16",
                "whisper_workers": max(1, wh_workers // 2),
                "whisper_batch_size": max(1, dyn_whisper // 2),
                "nllb_batch": max(1, dyn_nllb // 2),
                "ffmpeg_threads": self.cpu_cores
            },
            "MID": {
                "whisper_beam": 5,
                "whisper_workers": 1,
                "nllb_batch": dyn_nllb
            },
            "LOW": {
                "whisper_beam": 5,
                "nllb_batch": 1,
                "whisper_compute": "int8_float16"
            },
            "CPU_ONLY": {
                "whisper_beam": 5,
                "whisper_compute": "int8",
                "nllb_batch": 1,
                "ffmpeg_threads": 4
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
                msg = (
                    f"[Optimization] Dynamic NLLB batch size: "
                    f"{self.config['nllb_batch']} "
                    f"(based on {self.vram_gb}GB VRAM)"
                )
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
        from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
        import torch
        # Disable TF32 to ensure maximum precision
        global torch, NllbTokenizer, AutoModelForSeq2SeqLM
        if torch is None:
            import torch
        if NllbTokenizer is None:
            from transformers import NllbTokenizer
        if AutoModelForSeq2SeqLM is None:
            from transformers import AutoModelForSeq2SeqLM

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.tokenizer = NllbTokenizer.from_pretrained(config.NLLB_MODEL_ID)

        log(f"[Load] Initializing NLLB Model Environment... (Torch: {torch.__version__})", level="DEBUG")

        # Check available VRAM
        # bfloat16 Model takes ~7.5GB VRAM.

        # AGGRESSIVE: Clear all memory before loading NLLB (prevents 37GB peak)
        # GC/Cache clear
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        # SUCCESSFUL CONFIG: Use bfloat16 + NllbTokenizer
        try:
            dtype = torch.bfloat16
            log(
                f"[Load] Loading NLLB-200 (3.3B) in {dtype} "
                "(Verified Native Mode)...",
                level="DEBUG"
            )

            # STRICT VRAM ENFORCEMENT
            # 'auto' allows offloading to CPU/Disk which causes shared RAM spikes.
            # We force 'cuda:0' (or primary) to ensure it stays in VRAM or errors out (OOM)
            # rather than slowing down system with shared memory.
            if OPTIMIZER.config["device"] == "cuda":
                target_device = "cuda:0"
                device_map = {"": 0}  # Force entire model onto GPU 0
            else:
                target_device = "cpu"
                device_map = None

            try:
                # First attempt: Try standard load (checks online if needed)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    config.NLLB_MODEL_ID,
                    dtype=dtype,  # Fixed deprecation
                    low_cpu_mem_usage=True,
                    attn_implementation="eager",
                    tie_word_embeddings=True,
                    device_map=device_map
                )
            except (OSError, ValueError, RuntimeError) as net_err:
                log(f"[Load] Network/Load error ({net_err}). Trying local_files_only...", "WARNING")
                # Fallback: Local only
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    config.NLLB_MODEL_ID,
                    dtype=dtype,
                    low_cpu_mem_usage=True,
                    attn_implementation="eager",
                    tie_word_embeddings=True,
                    device_map=device_map,
                    local_files_only=True
                )

            # Ensure model is on the right device if device_map was None
            if device_map is None:
                self.model.to(target_device)

            # Official Weight Tying
            self.model.tie_weights()
            log(
                f"[Load] NLLB loaded in {self.model.dtype} "
                "(Native Weight Tying).",
                level="DEBUG"
            )
        except Exception as e:
            log(f"[Load] CRITICAL LOAD ERROR: {e}")
            raise e

        # Warm-up to allocate buffers
        if OPTIMIZER.config["device"] == "cuda":
            log("[Load] Warming up NLLB...", level="DEBUG")
            dummy = self.tokenizer(
                "Hello world", return_tensors="pt"
            ).to(self.model.device)
            with torch.no_grad():
                self.model.generate(**dummy, max_new_tokens=1)

    def translate(self, texts, src_lang_code, tgt_lang_code, **gen_kwargs):
        """Translates a batch of texts using verified Native NLLB logic."""
        if not self.model or not texts:
            return texts

        import torch
        log(f"  [AI] Input[0] Repr: {repr(texts[0])}", level="DEBUG")

        # 1. Tokenize (Native)
        # Explicit NllbTokenizer automatically handles [src_lang, text, EOS]
        self.tokenizer.src_lang = src_lang_code
        self.tokenizer.tgt_lang = tgt_lang_code

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.model.device)

        input_len = inputs.input_ids.shape[1]
        log(
            f"  [AI] Native Tokens: {inputs.input_ids[0].tolist()[:10]}... "
            f"(Len: {input_len})",
            level="DEBUG"
        )

        # 2. High-Quality Generation Settings (Dynamic from config)
        tgt_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang_code)

        gen_params = {
            "num_beams": config.NLLB_NUM_BEAMS,
            "length_penalty": config.NLLB_LENGTH_PENALTY,
            "repetition_penalty": config.NLLB_REPETITION_PENALTY,
            "no_repeat_ngram_size": config.NLLB_NO_REPEAT_NGRAM_SIZE,
            "early_stopping": True,
            "do_sample": False,
            "use_cache": True
        }
        gen_params.update(gen_kwargs)

        with torch.inference_mode():
            # Stop Rambling Hallucinations
            dynamic_max = min(512, int(input_len * 3) + 20)

            log(
                f"  [AI] Generation: Native-NLLB (Beam-5, Native-EO) | "
                f"Max={dynamic_max}",
                level="DEBUG"
            )

            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=tgt_token_id,
                max_new_tokens=dynamic_max,
                **gen_params
            )

        return self.tokenizer.batch_decode(
            generated_tokens.cpu(), skip_special_tokens=True
        )

    def offload(self):
        """Moves model to CPU and clears cache."""
        if self.model:
            log("  [AI] Offloading NLLB to CPU...")
            self.model.to("cpu")
            import torch
            torch.cuda.empty_cache()
            import gc
            gc.collect()


class ModelManager:
    """Lazy loader and manager for AI models."""

    def __init__(self):
        self._whisper = None
        self._nllb = None
        self._separator = None

    def get_whisper(self):
        if self._whisper is None:
            log("[AI] Loading Whisper")
            from faster_whisper import WhisperModel, BatchedInferencePipeline
            model = WhisperModel(
                config.WHISPER_MODEL_SIZE,
                device=OPTIMIZER.config["device"],
                compute_type=OPTIMIZER.config["whisper_compute"],
                num_workers=OPTIMIZER.config["whisper_workers"]
            )

            # Wrap in batching pipeline if configured
            batch_size = OPTIMIZER.config.get("whisper_batch_size", 1)
            if batch_size > 1:
                log(
                    f"[AI] Whisper Batching Enabled (Batch Size: {batch_size})"
                )
                self._whisper = BatchedInferencePipeline(model)
                self._whisper_base = model  # Keep reference for offloading
            else:
                self._whisper = model
                self._whisper_base = model

        return self._whisper

    def get_nllb(self):
        if self._nllb is None:
            # PROACTIVE: Clear all other models from VRAM before loading NLLB
            log(
                "[AI] Clearing memory for NLLB (High-Perf Profiling)...",
                level="DEBUG"
            )
            self.offload_whisper()
            self.offload_separator()

            log("[AI] Loading NLLB...", level="DEBUG")
            self._nllb = NLLBTranslator()
        return self._nllb

    def get_separator(self):
        if self._separator is None:
            log("[AI] Loading Audio Separator...")
            from audio_separator.separator import Separator
            self._separator = Separator(
                model_file_dir=os.path.join(os.getcwd(), "models"),
                output_dir=os.getcwd()
            )
            self._separator.load_model(
                model_filename=config.AUDIO_SEPARATOR_MODEL_ID
            )
        return self._separator

    def offload_whisper(self):
        """Frees Whisper VRAM."""
        if self._whisper:
            log("  [AI] Offloading Whisper to CPU...")
            # faster-whisper doesn't have a simple .to('cpu')
            # like torch, but we can delete and GC
            del self._whisper
            if hasattr(self, '_whisper_base'):
                del self._whisper_base
            self._whisper = None
            self._whisper_base = None
            import torch
            torch.cuda.empty_cache()
            import gc
            gc.collect()

    def offload_separator(self):
        """Frees Audio Separator VRAM."""
        if self._separator:
            log("  [AI] Offloading Audio Separator...")
            # Audio Separator handles its own cleanup usually,
            # but let's be safe
            del self._separator
            self._separator = None
            import torch
            torch.cuda.empty_cache()
            import gc
            gc.collect()

    def offload_nllb(self):
        """Frees NLLB VRAM."""
        if self._nllb:
            self._nllb.offload()

    def preload_nllb(self):
        """Optional preloading for checking OOM early."""
        self.get_nllb()
