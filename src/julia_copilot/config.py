"""Shared paths, defaults and logging configuration.

Every path is resolved relative to the current working directory, which is
assumed to be the repository root (that is how the original scripts behaved).
Each of them can be overridden from the CLI.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

DATA_DIR = Path("data")
REPOS_DIR = DATA_DIR / "repos"
REPOS_ZIP = DATA_DIR / "repos.zip"
JULIA_CSV = DATA_DIR / "julia.csv.gz"
COMBINED_PROJECTS_JSON = DATA_DIR / "combined_projects.json"
ENCODED_DATA_ROOT = DATA_DIR / "encoded_data"
CHECKPOINTS_DIR = Path("checkpoints")

BENCHMARK_DIR = Path("benchmark")
BENCHMARK_FILE = BENCHMARK_DIR / "humaneval-jl-reworded.jsonl"

# --------------------------------------------------------------------------- #
# Model / tokenisation defaults
# --------------------------------------------------------------------------- #

DEFAULT_MODEL = "HuggingFaceTB/SmolLM-135M"
CONTEXT_LENGTH = 2048

# --------------------------------------------------------------------------- #
# Parallelism
# --------------------------------------------------------------------------- #

MAX_WORKERS = 50
"""Upper bound on worker processes; the shared GPU box throttles beyond this."""

NUM_WORKERS = max(1, min(MAX_WORKERS, multiprocessing.cpu_count() - 1))

# --------------------------------------------------------------------------- #
# Repositories excluded from cloning
# --------------------------------------------------------------------------- #

EXCLUDED_REPOS: frozenset[str] = frozenset(
    {
        "lanl-ansi/MINLPLib.jl",
        "Mehrnoom/Cryptocurrency-Pump-Dump",
        "Cheng-Lin-Li/Market-Trend-Prediction",
    }
)
"""Repositories that are pathologically large or contain no usable Julia code."""

EXCLUDED_PROJECTS: frozenset[str] = frozenset({"HumanEval.jl"})
"""Projects dropped from training data because they leak into the benchmark."""


def init_pandarallel(*, progress_bar: bool = False, verbose: bool = False) -> None:
    """Initialise ``pandarallel`` lazily.

    The original modules called ``pandarallel.initialize()`` at import time,
    which forked worker processes even when the module was merely imported (for
    instance by the test suite). Calling it from here keeps the side effect
    inside the entry points that actually need it.
    """
    from pandarallel import pandarallel

    pandarallel.initialize(
        nb_workers=NUM_WORKERS,
        progress_bar=progress_bar,
        verbose=2 if verbose else 0,
    )


def setup_logging(verbose: bool = False, logfile: Path | str | None = None) -> None:
    """Configure the root logger once, for CLI entry points only."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if logfile is not None:
        handlers.append(logging.FileHandler(logfile, mode="a", encoding="utf-8"))

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )


def default_cuda_device() -> str:
    """Pin ``CUDA_VISIBLE_DEVICES`` to the lab's shared GPU unless already set."""
    return os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
