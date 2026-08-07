"""Generate Julia completions from a fine-tuned checkpoint.

The original ``src/query.py`` did ``from encode_data import TOKENIZER,
CONTEXT_LENGTH, MODEL_NAME``, but ``encode_data`` never defined ``TOKENIZER`` or
``MODEL_NAME`` -- the module raised :class:`ImportError` on every invocation.
The tokenizer is now loaded explicitly from the checkpoint (falling back to the
base model when the checkpoint is adapter-only).
"""

from __future__ import annotations

import logging
from pathlib import Path

from julia_copilot.config import CONTEXT_LENGTH, DEFAULT_MODEL

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(checkpoint: Path | str, tokenizer_name: str | None = None):
    """Load a checkpoint onto the best available device.

    Args:
        checkpoint: Directory produced by :mod:`julia_copilot.train`.
        tokenizer_name: Where to read the tokenizer from. Defaults to the
            checkpoint itself, then to :data:`~julia_copilot.config.DEFAULT_MODEL`
            when the checkpoint carries no tokenizer (LoRA-only saves).

    Returns:
        ``(model, tokenizer, device)``.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(str(checkpoint)).to(device)

    source = tokenizer_name or str(checkpoint)
    try:
        tokenizer = AutoTokenizer.from_pretrained(source)
    except OSError:
        logger.warning("No tokenizer in %s, falling back to %s.", source, DEFAULT_MODEL)
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, device


def generate_completions(
    model,
    tokenizer,
    query: str,
    device: str,
    *,
    top_n: int = 1,
    max_length: int = CONTEXT_LENGTH,
) -> list[str]:
    """Generate ``top_n`` completions for ``query`` with beam search.

    Args:
        model: A causal language model.
        tokenizer: Its tokenizer.
        query: Prompt, typically a docstring plus a function signature.
        device: ``"cpu"`` or ``"cuda"``.
        top_n: Number of completions (also the beam width).
        max_length: Hard cap on the generated sequence length.

    Returns:
        The decoded completions, best first.
    """
    inputs = tokenizer(query, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=top_n,
        num_beams=max(top_n, 1),
        no_repeat_ngram_size=2,
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
