import argparse, os
import torch

from encode_data import encode_data, setup_tokenizer

from transformers import AutoModelForCausalLM
from transformers import TrainingArguments, Trainer


if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def trainer_for_model(model_name, dataset, output_dir="checkpoints"):
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer, _ = setup_tokenizer(model_name)

    batch_size = 2
    if model_name == "HuggingFaceTB/SmolLM-360M": # can't fit more than 1 batch
        batch_size = 1

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
    parser.add_argument("--first-line", action="store_true", help="Extract only the first line of the documentation.")
    parser.add_argument("--frac-of-data", type=float, default=1, help="Fraction of data to use for training. Default is 1. Use a smaller value (between 0 and 1) for testing.")
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
    trainer = trainer_for_model(args.model, encoded_dataset, output_dir=output_dir)

    # for param in model.model.layers[:20].parameters():
    #     param.requires_grad = False
    # print(f"num params:", model.num_parameters())
    # print(f"num trainable params:", model.num_parameters(only_trainable=True))
    trainer.train()
