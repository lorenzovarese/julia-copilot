import argparse, os
import torch

from encode_data import encode_data, setup_tokenizer

from transformers import AutoModelForCausalLM
from transformers import TrainingArguments, Trainer

from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def trainer_for_quantized_model(model_name, dataset, batch_size=2, output_dir="checkpoints"):
    tokenizer, _ = setup_tokenizer(model_name)

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16 # P.S. use torch.float16 on gym J
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=nf4_config)
    r = 8
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=r,
        lora_alpha=2*r
    )

    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    args = TrainingArguments(
        save_strategy="epoch",
        output_dir=output_dir,
        overwrite_output_dir=True,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        num_train_epochs=5,
        fp16=True,
        optim="adamw_8bit",
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        peft_config=lora_config,
        tokenizer=tokenizer
    )
    return trainer

def trainer_for_model(model_name, dataset, batch_size=2, output_dir="checkpoints"):
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer, _ = setup_tokenizer(model_name)

    args = TrainingArguments(
        save_strategy="epoch",
        output_dir=output_dir,
        overwrite_output_dir=True,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        num_train_epochs=5,
        fp16=True,
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    return trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="HuggingFaceTB/SmolLM-135M", help="Model to use for training. Default is 'HuggingFaceTB/SmolLM-135M'.")
    parser.add_argument("-q", "--quantized", action="store_true", help="Quantize the model using BitsAndBytes.")
    parser.add_argument("--first-line", action="store_true", help="Extract only the first line of the documentation.")
    parser.add_argument("--frac-of-data", type=float, default=1, help="Fraction of data to use for training. Default is 1. Use a smaller value (between 0 and 1) for testing.")
    parser.add_argument("-b", "--batch-size", type=int, default=2, help="Batch size for training. Default is 2. Increase this if you have a larger GPU.")
    parser.add_argument("--simple-input", action="store_true", help="Save the input only the docstring and signature of the function, and setas the expected output the body of the function. If you don't give this flag, then the entire function (docstring + signature + body) is given as both input and expected output")
    parser.add_argument("--encoded-data-root", type=str, default=os.path.join("data", "encoded_data"), help="Path to the encoded dataset. Default is 'data/encoded_data'.")

    args = parser.parse_args()

    encoded_dataset = encode_data(
        encoded_data_root=args.encoded_data_root,
        model_name=args.model,
        frac_of_data=args.frac_of_data,
        simple_input=args.simple_input,
        first_line_of_doc=args.first_line,
        verbose=True,
    )
    print(f"Training on {len(encoded_dataset)} instances")

    output_dir = os.path.join("checkpoints", args.model.replace("/", "_"))
    output_dir = os.path.join(output_dir, f"{args.frac_of_data*100:03.0f}")
    if args.simple_input:
        output_dir += "_simple"
    if args.first_line:
        output_dir += "_first_line"

    print(f"Saving checkpoints to '{output_dir}'")
    if args.quantized:
        trainer = trainer_for_quantized_model(args.model, encoded_dataset, batch_size=args.batch_size, output_dir=output_dir)
    else:
        trainer = trainer_for_model(args.model, encoded_dataset, batch_size=args.batch_size, output_dir=output_dir)

    # for param in model.model.layers[:20].parameters():
    #     param.requires_grad = False
    # print(f"num params:", model.num_parameters())
    # print(f"num trainable params:", model.num_parameters(only_trainable=True))
    trainer.train()
