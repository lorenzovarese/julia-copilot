"""Turn the extracted function corpus into a tokenised HuggingFace dataset.

Pipeline: drop leaked/benchmark projects, deduplicate, keep only documented
"basic" functions, strip the signature back out of the docstring, optionally
truncate the docstring to its first sentence, assemble the training string, and
tokenise to a fixed width.

Two bugs from the original ``src/encode_data.py`` are fixed here:

* **Benchmark leakage.** The HumanEval exclusion tested
  ``name not in functions_in_benchmark`` where the right-hand side was a
  ``pandas.Series``. Membership on a Series checks its *index*, not its values,
  so the test was always true and no benchmark function was ever removed. The
  lookup now goes through a ``set``.
* **Separator stripping.** ``clean_doc`` called ``re.sub(r"---+", ...)`` and
  discarded the result, so horizontal rules survived into the training data.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from julia_copilot.config import (
    BENCHMARK_FILE,
    CONTEXT_LENGTH,
    DEFAULT_MODEL,
    ENCODED_DATA_ROOT,
    EXCLUDED_PROJECTS,
    NUM_WORKERS,
    init_pandarallel,
)

logger = logging.getLogger(__name__)

MIN_DOC_TOKENS = 10
"""Docstrings shorter than this carry no signal ("TODO", "helper", ...)."""

DOCSTRING_PATTERN = re.compile(r'""".*?"""', re.DOTALL)
SIGNATURE_NAME_PATTERN = re.compile(r"function\s+([^(]+)\(")
SEPARATOR_PATTERN = re.compile(r"---+")


def setup_tokenizer(model_name: str = DEFAULT_MODEL):
    """Load the tokenizer and the effective context length for ``model_name``.

    The context width is the smaller of the model's positional limit and
    :data:`julia_copilot.config.CONTEXT_LENGTH`. The original code filtered
    against the constant but padded to the model's own limit; when the two
    disagree that produces examples longer than the filter allowed.

    Returns:
        ``(tokenizer, context_length)``.
    """
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    context_length = min(getattr(config, "max_position_embeddings", CONTEXT_LENGTH), CONTEXT_LENGTH)
    return tokenizer, context_length


def benchmark_function_names(benchmark_file: Path = BENCHMARK_FILE) -> set[str]:
    """Names of every function appearing in the MultiPL-E HumanEval-jl prompts.

    Anything with a matching name is dropped from training so the benchmark
    stays out of sample.
    """
    benchmark_file = Path(benchmark_file)
    if not benchmark_file.exists():
        logger.warning("Benchmark file %s not found; leakage filter disabled.", benchmark_file)
        return set()

    prompts = pd.read_json(benchmark_file, lines=True).prompt

    names: set[str] = set()
    for prompt in prompts:
        match = SIGNATURE_NAME_PATTERN.search(DOCSTRING_PATTERN.sub("", prompt))
        if match:
            names.add(match.group(1).strip())
    return names


def remove_signature_from_doc(docstring: str, signature: str) -> str:
    """Strip a leading restatement of the signature from a docstring.

    Julia convention repeats ``name(args)`` at the top of the docstring; keeping
    it would let the model copy the signature instead of predicting it.
    """
    match = re.match(r"^function\s+([^(]+)", signature)
    if match is None:
        return docstring.strip("\n")

    name = match.group(1)
    pattern = rf"^(function )?\s*{re.escape(name)}\s*\(.*?\)\s*"
    while re.match(pattern, docstring, flags=re.DOTALL):
        docstring = re.sub(pattern, "", docstring, flags=re.DOTALL)
    return docstring.strip("\n")


def first_sentence(docstring: str) -> str:
    """Truncate to the first sentence, or the first paragraph, whichever ends sooner."""
    dot = docstring.find(".")
    para = docstring.find("\n\n")
    candidates = [pos for pos in (dot, para) if pos >= 0]
    return docstring if not candidates else docstring[: min(candidates) + 1]


def clean_doc(docstring: str) -> str:
    """Remove horizontal rules and wrap the docstring in triple quotes."""
    docstring = SEPARATOR_PATTERN.sub("", docstring)
    return f'"""{docstring}\n"""'


def _encoded_path(
    root: Path, model_name: str, frac_of_data: float, simple_input: bool, first_line: bool
) -> Path:
    """Cache location, keyed by every option that changes the encoded output."""
    suffix = f"{frac_of_data * 100:03.0f}"
    if simple_input:
        suffix += "_simple"
    if first_line:
        suffix += "_first_line"
    return Path(root) / model_name.replace("/", "_") / suffix


def encode_data(
    encoded_data_root: Path = ENCODED_DATA_ROOT,
    data_path: Path = Path("data/combined_projects.zip"),
    model_name: str = DEFAULT_MODEL,
    *,
    simple_input: bool = False,
    first_line_of_doc: bool = False,
    frac_of_data: float = 1.0,
    force: bool = False,
    benchmark_file: Path = BENCHMARK_FILE,
):
    """Build (or load from cache) the tokenised training dataset.

    Args:
        encoded_data_root: Root of the on-disk dataset cache.
        data_path: Corpus of extracted projects (JSON, optionally zipped).
        model_name: Model whose tokenizer and context width are used.
        simple_input: Encode docstring + signature as input and the body as the
            label, rather than the whole function as both.
        first_line_of_doc: Keep only the first sentence of each docstring.
        frac_of_data: Fraction of the corpus to sample, in ``(0, 1]``.
        force: Re-encode even when a cached dataset exists.
        benchmark_file: HumanEval-jl prompts used for the leakage filter.

    Returns:
        A :class:`datasets.Dataset` ready for :mod:`julia_copilot.train`.
    """
    import datasets

    tokenizer, context_length = setup_tokenizer(model_name)
    encoded_path = _encoded_path(
        encoded_data_root, model_name, frac_of_data, simple_input, first_line_of_doc
    )

    if not force and encoded_path.exists():
        logger.info("Loading cached encoded data from %s", encoded_path)
        return datasets.load_from_disk(str(encoded_path))

    init_pandarallel()
    logger.info("Loading corpus from %s", data_path)
    projects = pd.read_json(data_path)

    logger.info("Removing projects that overlap the benchmark...")
    projects = projects[~projects["project"].isin(EXCLUDED_PROJECTS)]

    logger.info("Flattening projects into functions...")
    functions = pd.DataFrame([func for funcs in projects.functions for func in funcs])

    before = len(functions)
    functions = functions.drop_duplicates(subset=["documentation", "signature", "body"])
    logger.info("Dropped %d duplicate functions.", before - len(functions))

    logger.info("Keeping documented, non-benchmark, basic functions...")
    in_benchmark = benchmark_function_names(benchmark_file)

    def keep(function) -> bool:
        match = SIGNATURE_NAME_PATTERN.search(function["signature"])
        if match is None:
            return False
        return (
            function["type"] == "basic"
            and len(function["documentation"]) > 0
            and match.group(1).strip() not in in_benchmark
        )

    before = len(functions)
    functions = functions[functions.apply(keep, axis=1)]
    logger.info(
        "Filtered out %d functions (%d remaining).", before - len(functions), len(functions)
    )

    logger.info("Stripping repeated signatures from docstrings...")
    functions["documentation"] = functions.parallel_apply(
        lambda row: remove_signature_from_doc(row["documentation"], row["signature"]), axis=1
    )

    if first_line_of_doc:
        logger.info("Truncating docstrings to their first sentence...")
        functions["documentation"] = functions["documentation"].map(first_sentence)

    if frac_of_data < 1:
        logger.info("Sampling %.0f%% of the data...", frac_of_data * 100)
        functions = functions.sample(frac=frac_of_data, random_state=100)
        logger.info("%d instances after sampling.", len(functions))

    functions["documentation"] = functions["documentation"].map(clean_doc)

    logger.info("Assembling training strings...")
    if simple_input:
        functions["simple_input"] = (
            tokenizer.bos_token + functions["documentation"] + "\n" + functions["signature"] + "\n"
        )

    functions["complete_function"] = (
        tokenizer.bos_token
        + functions["documentation"]
        + "\n"
        + functions["signature"]
        + "\n"
        + functions["body"]
        + "\nend"
        + tokenizer.eos_token
    )

    logger.info("Dropping over-long, under-documented and test functions...")

    def fits(function) -> bool:
        return (
            len(tokenizer(function["documentation"])["input_ids"]) >= MIN_DOC_TOKENS
            and len(tokenizer(function["complete_function"])["input_ids"]) <= context_length
            and "@testset" not in function["body"]
            and "function test" not in function["signature"].lower()
        )

    before = len(functions)
    functions = functions[functions.parallel_apply(fits, axis=1)]
    logger.info(
        "Filtered out %d functions (%d remaining).", before - len(functions), len(functions)
    )

    logger.info("Tokenising...")

    def encode(batch):
        if simple_input:
            encoded = tokenizer(
                batch["simple_input"],
                truncation=True,
                max_length=context_length,
                padding="max_length",
            )
            labels = tokenizer(
                [body + "\nend" + tokenizer.eos_token for body in batch["body"]],
                truncation=True,
                max_length=context_length,
                padding="max_length",
            )
            encoded["labels"] = labels["input_ids"].copy()
        else:
            encoded = tokenizer(
                batch["complete_function"],
                truncation=True,
                max_length=context_length,
                padding="max_length",
            )
            encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    dataset = datasets.Dataset.from_pandas(functions)
    dataset = dataset.map(encode, batched=True, num_proc=NUM_WORKERS)

    logger.info("Saving encoded data to %s", encoded_path)
    encoded_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(encoded_path))
    return dataset
