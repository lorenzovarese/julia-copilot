"""Fine-tuning and benchmarking of small language models for Julia code completion.

The package is organised as a pipeline, each stage exposed both as a Python
function and as a ``julia-copilot`` subcommand:

``clone``      -> download and strip GitHub repositories whose primary language is Julia
``parse``      -> extract functions (regex or tree-sitter backend) into a JSON corpus
``filter``     -> split the corpus by documentation quality
``encode``     -> tokenise the corpus into a HuggingFace dataset
``train``      -> fine-tune a causal LM on the encoded dataset
``query``      -> generate completions from a checkpoint
``extract``    -> flatten MultiPL-E benchmark results into a single JSON file
``mcnemar``    -> compare two benchmark runs with McNemar's test
"""

from julia_copilot.config import (
    BENCHMARK_FILE,
    COMBINED_PROJECTS_JSON,
    CONTEXT_LENGTH,
    DEFAULT_MODEL,
    NUM_WORKERS,
    REPOS_DIR,
)

__version__ = "0.2.0"

__all__ = [
    "BENCHMARK_FILE",
    "COMBINED_PROJECTS_JSON",
    "CONTEXT_LENGTH",
    "DEFAULT_MODEL",
    "NUM_WORKERS",
    "REPOS_DIR",
    "__version__",
]
