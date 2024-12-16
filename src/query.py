import torch
from transformers import AutoModelForCausalLM
from encode_data import TOKENIZER, CONTEXT_LENGTH, MODEL_NAME
import argparse

def generate_completions(model, tokenizer, query, device, top_n=1, max_length=CONTEXT_LENGTH):
    """
    Generate completions for a given query using a language model.

    Parameters:
        model: Pretrained causal language model.
        tokenizer: Tokenizer associated with the language model.
        query (str): Query text to generate completions for.
        device (str): Device to run the model on ('cpu' or 'cuda').
        top_n (int): Number of completions to generate. Default is 1.
        max_length (int): Maximum length of the generated sequence. Default is CONTEXT_LENGTH.

    Returns:
        list: A list of generated completions.
    """
    inputs = tokenizer(query, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs, 
        max_length=max_length, 
        num_return_sequences=top_n, 
        num_beams=top_n, 
        no_repeat_ngram_size=2, 
        early_stopping=True
    )
    completions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return completions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, required=True, help="Path to the checkpoint to use for training.")
    parser.add_argument("-q", "--query", type=str, required=True, help="Query to generate completions for.")
    parser.add_argument("--top-n", type=int, default=1, help="Number of top completions to generate (default: 1).")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint).to(device)

    print("Generating completions...")
    completions = generate_completions(model, TOKENIZER, args.query, device, top_n=args.top_n)

    for i, completion in enumerate(completions):
        print(f"{i+1}. {completion}")
