"""VRAM-aware model selection and batch sizing.

The coarse hardware profile (ULTRA/HIGH/MID/LOW/CPU_ONLY) picks sensible caps,
but two decisions have to be made against the *actual* card:

* **Which translation model fits.** Weights that take more than about 60 % of
  VRAM leave no room for beam-search activations. On an RTX 3080 Laptop
  (8 GB) NLLB-200-3.3B (6.3 GB fp16) hit 130 CUDA out-of-memory retries while
  loading, and the loader's CPU fallback then ran the whole model on the
  laptop CPU: 160 MiB on the GPU, 850 % CPU, 0.4x realtime, over a hundred
  times slower than on a 32 GB card. Picking the largest NLLB whose weights
  fit the budget keeps the model on the GPU with the next size down.
* **How large a batch the free memory allows.** Activation memory per item is
  measured per model size (see ``NLLB_PER_ITEM_GB``) and the batch is sized
  from the VRAM left *after* the model has loaded, capped by the profile.

All figures are from calibration on an RTX 5090 with the production generate
settings, 10 beams (``docs/hardware_optimization.md``): fp16 weights measured
as 6.27 / 2.59 / 1.15 GB and per-item activations as 0.154 / 0.093 / 0.063 GB
for 3.3B / 1.3B / 600M, rounded up here. They are conservative estimates,
not exact accounting; the worker's OOM bisection remains the safety net.
"""

# fp16 weight footprint in GB, largest first. Only models the pipeline ships with.
NLLB_CANDIDATES = (
    ("facebook/nllb-200-3.3B", 6.3),
    ("facebook/nllb-200-distilled-1.3B", 2.6),
    ("facebook/nllb-200-distilled-600M", 1.2),
)
NLLB_WEIGHT_GB = dict(NLLB_CANDIDATES)
NLLB_WEIGHT_GB["facebook/nllb-200-1.3B"] = 2.6

# Peak activation memory per batch item at 10 beams (GB); scales roughly with beams.
NLLB_PER_ITEM_GB = {
    "facebook/nllb-200-3.3B": 0.16,
    "facebook/nllb-200-distilled-1.3B": 0.10,
    "facebook/nllb-200-1.3B": 0.10,
    "facebook/nllb-200-distilled-600M": 0.07,
}
NLLB_PER_ITEM_DEFAULT_GB = 0.16
CALIBRATION_BEAMS = 10

# Weights above this share of VRAM cannot load (3080 8 GB at 82 %: OOM, silent CPU fallback).
# 0.60 keeps a 10 GB card on the distilled 1.3B rather than squeezing the 3.3B in at 63 %.
WEIGHT_BUDGET_FRACTION = 0.60
# Share of the free VRAM after loading that may go to activations.
ACTIVATION_BUDGET_FRACTION = 0.80

AUTO = "auto"


def select_nllb_model(configured_id, vram_gb):
    """Return ``(model_id, note)`` for the card: the configured id, or the largest that fits on ``auto``.

    ``note`` is a human-readable reason, or a warning when an explicitly
    configured model will not fit comfortably.
    """
    if configured_id and configured_id != AUTO:
        return configured_id, _fit_warning(configured_id, vram_gb)
    if vram_gb <= 0:
        model_id = NLLB_CANDIDATES[-1][0]
        return model_id, f"auto: no GPU memory reported, using {model_id}"
    return _largest_fitting_model(vram_gb)


def _largest_fitting_model(vram_gb):
    """Return the largest candidate whose weights fit the VRAM budget, with the reason."""
    budget = vram_gb * WEIGHT_BUDGET_FRACTION
    for model_id, weight_gb in NLLB_CANDIDATES:
        if weight_gb <= budget:
            return model_id, f"auto: {weight_gb:.1f} GB weights fit the {budget:.1f} GB budget of a {vram_gb} GB card"
    model_id, weight_gb = NLLB_CANDIDATES[-1]
    return model_id, f"auto: {vram_gb} GB is below every budget; using the smallest model ({weight_gb:.1f} GB)"


def _fit_warning(model_id, vram_gb):
    """Return a warning when a configured model's weights exceed the VRAM budget, else None."""
    weight_gb = NLLB_WEIGHT_GB.get(model_id)
    if weight_gb is None or vram_gb <= 0:
        return None
    budget = vram_gb * WEIGHT_BUDGET_FRACTION
    if weight_gb <= budget:
        return None
    return (
        f"{model_id} needs {weight_gb:.1f} GB for weights, above the {budget:.1f} GB budget of a {vram_gb} GB card; "
        "expect an out-of-memory fallback to the CPU. Set models.nllb to 'auto'."
    )


def dynamic_batch_size(free_gb, model_id, num_beams, cap, floor=1):
    """Size a translation batch from the VRAM free after the model loaded.

    ``cap`` is the profile's maximum; the result never exceeds it and never
    drops below ``floor``. Per-item cost is scaled by ``num_beams`` relative
    to the calibration beams.
    """
    per_item = NLLB_PER_ITEM_GB.get(model_id, NLLB_PER_ITEM_DEFAULT_GB)
    per_item *= max(1, num_beams) / CALIBRATION_BEAMS
    usable = max(0.0, free_gb) * ACTIVATION_BUDGET_FRACTION
    fitted = int(usable / per_item) if per_item > 0 else cap
    return max(floor, min(cap, fitted))


def whisper_compute_type(vram_gb):
    """Pick Faster-Whisper's CUDA compute type: fp16 with room to spare, int8 weights on small cards."""
    return "float16" if vram_gb >= 6 else "int8_float16"


def describe_budget(vram_gb, free_gb, model_id, batch, cap):
    """One log line summarising the dynamic decision."""
    weight_gb = NLLB_WEIGHT_GB.get(model_id)
    weights = f"{weight_gb:.1f} GB weights" if weight_gb else "unknown weights"
    return f"[VRAM] {vram_gb} GB card, {free_gb:.1f} GB free after load ({weights}) -> batch {batch} (profile cap {cap})"
