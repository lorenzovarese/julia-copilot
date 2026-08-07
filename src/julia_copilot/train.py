"""Fine-tune a causal language model on the encoded Julia corpus.

Two paths are available:

* full fine-tuning through :class:`transformers.Trainer`;
* 4-bit NF4 quantisation plus LoRA adapters through :class:`trl.SFTTrainer`,
  which is what makes the larger checkpoints fit on a single shared GPU.

Both accept the same dataset and write to the same checkpoint layout.
"""

from __future__ import annotations

import logging
from pathlib import Path

from julia_copilot.config import CHECKPOINTS_DIR, DEFAULT_MODEL
from julia_copilot.encode import setup_tokenizer

logger = logging.getLogger(__name__)

LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
LORA_RANK = 8


def _training_arguments(output_dir: Path, batch_size: int, config_cls=None, **overrides):
    """Build the shared training configuration.

    ``config_cls`` lets the LoRA path pass :class:`trl.SFTConfig`, which
    subclasses ``TrainingArguments`` and adds the SFT-specific fields that
    ``SFTTrainer`` expects.

    ``overwrite_output_dir`` was dropped from ``TrainingArguments`` in
    transformers 5, so it is no longer passed; checkpoints are written per
    epoch into a run-specific directory anyway.
    """
    if config_cls is None:
        from transformers import TrainingArguments

        config_cls = TrainingArguments

    return config_cls(
        output_dir=str(output_dir),
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=batch_size,
        num_train_epochs=NUM_EPOCHS,
        fp16=True,
        **overrides,
    )


def trainer_for_quantized_model(
    model_name: str, dataset, batch_size: int = 2, output_dir: Path = CHECKPOINTS_DIR
):
    """Build an :class:`~trl.SFTTrainer` over a 4-bit model with LoRA adapters."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    tokenizer, _ = setup_tokenizer(model_name)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", quantization_config=quantization_config
    )

    lora_config = LoraConfig(task_type="CAUSAL_LM", r=LORA_RANK, lora_alpha=2 * LORA_RANK)
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return SFTTrainer(
        model=model,
        args=_training_arguments(
            output_dir, batch_size, config_cls=SFTConfig, optim="adamw_bnb_8bit"
        ),
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )


def trainer_for_model(
    model_name: str, dataset, batch_size: int = 2, output_dir: Path = CHECKPOINTS_DIR
):
    """Build a plain :class:`~transformers.Trainer` for full fine-tuning."""
    from transformers import AutoModelForCausalLM, Trainer

    tokenizer, _ = setup_tokenizer(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    return Trainer(
        model=model,
        args=_training_arguments(output_dir, batch_size),
        train_dataset=dataset,
        processing_class=tokenizer,
    )


def checkpoint_dir(
    model_name: str = DEFAULT_MODEL,
    frac_of_data: float = 1.0,
    *,
    simple_input: bool = False,
    first_line: bool = False,
    root: Path = CHECKPOINTS_DIR,
) -> Path:
    """Mirror the encoded-data layout so checkpoints stay traceable to their data."""
    suffix = f"{frac_of_data * 100:03.0f}"
    if simple_input:
        suffix += "_simple"
    if first_line:
        suffix += "_first_line"
    return Path(root) / model_name.replace("/", "_") / suffix
