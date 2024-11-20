from transformers import AutoTokenizer
import pandas as pd
import os, multiprocessing
import datasets
import argparse
from pandarallel import pandarallel  # For parallel processing
NUM_PROC = min(50, multiprocessing.cpu_count() - 1)

MODEL_NAME = "HuggingFaceTB/SmolLM-135M"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
TOKENIZER.pad_token = TOKENIZER.eos_token

CONTEXT_LENGTH = 2048

def encode_data(
        encoded_data_path=os.path.join("data", f"encoded_data"),
        data_path=os.path.join("data", "combined_projects.zip"), 
        simple_input=False,
        frac_of_data=1,
        force=False,
        verbose=False,
    ):
    encoded_data_path = encoded_data_path + f"_{frac_of_data*100:03.0f}"

    if simple_input:
        encoded_data_path = encoded_data_path + "_simple"

    if not force and os.path.exists(encoded_data_path):
        if verbose: print(f"Loading encoded data from '{encoded_data_path}'...")
        return datasets.load_from_disk(encoded_data_path)

    pandarallel.initialize(progress_bar=True, nb_workers=NUM_PROC)
    
    if verbose: 
        if force: print("Forcing re-encoding of data.")
        else: print(f"No cached data found at {encoded_data_path}.")
        print(f"Loading data from '{data_path}'...")

    projects = pd.read_json(data_path)

    if verbose: print("Extracting functions...")
    functions = pd.DataFrame([func for funcs in projects.functions for func in funcs])

    if verbose: print("Dropping duplicates...")
    functions = functions.drop_duplicates(subset=["documentation", "signature", "body"])

    if frac_of_data < 1:
        if verbose: print(f"Sampling {frac_of_data*100:0.0f}% of the data...")
        functions = functions.sample(frac=frac_of_data, random_state=100)

    if verbose: print("Creating the actual functions...")

    if simple_input:
        functions["simple_input"] = \
            functions["documentation"] + "\n" \
            + functions["signature"] + "\n" 

    functions["complete_function"] = \
        functions["documentation"] + "\n" \
        + functions["signature"] + "\n" \
        + functions["body"] + "\n" \
        + "end"

    if verbose: print("Filtering functions that are longer than the context length but have at least some documentation...")
    def filter_function(function):
        return (
            function['type'] == "basic"
            and len(TOKENIZER(function['documentation'])['input_ids']) > 10 
            and len(TOKENIZER(function["complete_function"])['input_ids']) <= CONTEXT_LENGTH
        )

    len_before = len(functions)
    functions = functions[functions.parallel_apply(filter_function, axis=1)]
    print(f"Filtered out {len_before - len(functions)} functions ({len(functions)} remaining).")

    if verbose: print("Encoding data...")
    def encode_function(function):
        if simple_input:
            tokenized_inputs = TOKENIZER(
                function["simple_input"],
                truncation=True,
                max_length=CONTEXT_LENGTH,
                padding="max_length"
            )
            tokenized_output = TOKENIZER(
                [body + "\nend" for body in function["body"]],
                truncation=True,
                max_length=CONTEXT_LENGTH,
                padding="max_length"
            )
            tokenized_inputs['labels'] = tokenized_output['input_ids'].copy()
        else:
            tokenized_inputs = TOKENIZER(
                function["complete_function"],
                truncation=True,
                max_length=CONTEXT_LENGTH,
                padding="max_length"
            )
            tokenized_inputs['labels'] = tokenized_inputs['input_ids'].copy()

        return tokenized_inputs
    dataset = datasets.Dataset.from_pandas(functions)
    dataset = dataset.map(encode_function, batched=True, num_proc=NUM_PROC)

    if verbose: print(f"Saving encoded data to '{encoded_data_path}'...")
    dataset.save_to_disk(encoded_data_path)
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode the dataset for training a model.")
    parser.add_argument("-f", "--force", action="store_true", help="Force re-encoding of data.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose output.")
    parser.add_argument("--encoded-data-path", type=str, default=os.path.join("data", "encoded_data"), help="Path to save the encoded dataset. Default is 'data/encoded_data'. Note: The path is then extended with the fraction of the data, together with whether it is only recent issues or not.")
    parser.add_argument("--simple-input", action="store_true", help="Save the input only the docstring and signature of the function, and setas the expected output the body of the function. If you don't give this flag, then the entire function (docstring + signature + body) is given as both input and expected output")
    parser.add_argument("--frac", type=float, default=1, help="Fraction of the data to use. Default is 1.")

    args = parser.parse_args()
    encode_data(
        encoded_data_path=args.encoded_data_path,
        force=args.force,
        verbose=args.verbose,
        simple_input=args.simple_input,
        frac_of_data=args.frac,
    )
