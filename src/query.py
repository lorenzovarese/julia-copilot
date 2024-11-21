import torch
from transformers import AutoModelForCausalLM
from encode_data import TOKENIZER, CONTEXT_LENGTH, MODEL_NAME
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, required=True, help="Path to the checkpoint to use for training.")
    parser.add_argument("-q", "--query", type=str, required=True, help="Query to generate completions for.")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint).to(device)

    inputs = TOKENIZER(args.query, return_tensors="pt").to(device)
    # get top 1 suggestion
    print("Generating completions...")
    outputs = model.generate(**inputs, max_length=1024)
    completion = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    print(completion)

    # get top 5 suggestions
    # outputs = model.generate(**inputs, max_length=CONTEXT_LENGTH, num_return_sequences=5, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    # completions = [TOKENIZER.decode(output, skip_special_tokens=True) for output in outputs]
    # for i, completion in enumerate(completions):
    #     print(f"{i+1}. {completion}")
