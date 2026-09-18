"""Early runtime environment bootstrapping."""

import os
import sys

# ``CUDA_VISIBLE_DEVICES=""`` is not a reliable opt-out: torch 2.14/cu132 still
# reported the GPU as available and used it. A negative id hides every device.
CPU_ONLY_CUDA_DEVICES = "-1"


def force_cpu_only_env() -> None:
    """Hide every CUDA device so all engines fall back to the CPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = CPU_ONLY_CUDA_DEVICES


def bootstrap_cpu_env(argv: list[str] | None = None) -> None:
    """Hide CUDA devices when the --cpu flag is present, before heavy imports."""
    cli_args = sys.argv[1:] if argv is None else argv
    if "--cpu" in cli_args:
        force_cpu_only_env()
