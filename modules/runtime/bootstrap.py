"""Early runtime environment bootstrapping."""

import os
import sys


def bootstrap_cpu_env(argv: list[str] | None = None) -> None:
    """Set CUDA_VISIBLE_DEVICES to empty if --cpu flag is present before heavy imports."""
    cli_args = sys.argv[1:] if argv is None else argv
    if "--cpu" in cli_args:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
