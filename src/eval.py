import argparse, os
import torch
from encode_data import encode_data, TOKENIZER, MODEL_NAME

from transformers import AutoModelForCausalLM
from transformers import TrainingArguments, Trainer


def trainer_for_model(model, dataset, output_dir=os.path.join("data", "checkpoints")):
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    args = TrainingArguments(
        save_strategy="epoch",
        # eval_strategy="epoch",
        output_dir=output_dir,
        overwrite_output_dir=False,
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        num_train_epochs=5,
        fp16=True,
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=dataset,
        tokenizer=TOKENIZER,
        # compute_metrics=compute_metrics,
    )

    return trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train-model", action="store_true", help="Force the training of the model.")
    parser.add_argument("--frac-of-data", type=float, default=1, help="Fraction of data to use for training. Default is 1. Use a smaller value (between 0 and 1) for testing.")
    parser.add_argument("--simple-input", action="store_true", help="Save the input only the docstring and signature of the function, and setas the expected output the body of the function. If you don't give this flag, then the entire function (docstring + signature + body) is given as both input and expected output")
    parser.add_argument("--encoded-data-path", type=str, default=os.path.join("data", "encoded_data"), help="Path to the encoded dataset. Default is 'data/encoded_data'. Note: The path is then extended with the fraction of the data, together with whether it is only recent issues or not.")

    args = parser.parse_args()

    encoded_dataset = encode_data(
        encoded_data_path=args.encoded_data_path,
        frac_of_data=args.frac_of_data,
        simple_input=args.simple_input,
        verbose=True,
    )
    print(f"Training on {len(encoded_dataset)} instances")
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train_model:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    else:
        if args.checkpoint:
            model = AutoModelForCausalLM.from_pretrained(args.checkpoint)
        else:
            raise ValueError("You need to provide a checkpoint path if you're not forcing training")

    trainer = trainer_for_model(model, encoded_dataset)

    # for param in model.model.layers[:20].parameters():
    #     param.requires_grad = False
    print(f"num params:", model.num_parameters())
    print(f"num trainable params:", model.num_parameters(only_trainable=True))
    if args.train_model:
        trainer.train()
        save_path = os.path.join("data", "best-model")
        trainer.model.save_pretrained(save_path)
        trainer.tokenizer.save_pretrained(save_path)
