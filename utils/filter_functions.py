import json
import re

def count_code_lines(body):
    """Counts the lines of code in the function body excluding comments and empty lines."""
    lines = body.split("\n")
    code_lines = [line for line in lines if line.strip() and not line.strip().startswith("#")]
    return len(code_lines)

def is_type_annotated(signature):
    """Checks if a function signature contains Julia type annotations."""
    return "::" in signature

def remove_non_meaningful_comments(body):
    """Removes comments that do not contain at least 3 alphabetical characters."""
    lines = body.split("\n")
    cleaned_lines = []
    for line in lines:
        if line.strip().startswith("#"):
            if len(re.findall(r'[a-zA-Z]', line)) >= 3:
                cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

def normalize_function_body(body):
    """Normalizes function body by removing comments and extra whitespaces."""
    body = remove_non_meaningful_comments(body)
    lines = body.split("\n")
    normalized_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    return "\n".join(normalized_lines)

def extract_function_name(signature):
    """Extracts the function name from the signature."""
    match = re.match(r"function\s+([a-zA-Z_][a-zA-Z0-9_]*)\(", signature)
    if match:
        return match.group(1)
    return None

def has_good_documentation(func):
    """
    Determines if a function has good documentation.
    Criteria:
    - Documentation starts with the function name.
    - Documentation contains an example with '# Example' or '```jldoctest'.
    """
    documentation = func.get("documentation", "").strip()
    signature = func.get("signature", "").strip()
    function_name = extract_function_name(signature)

    if not documentation or not function_name:
        return False

    # Check if the first word in the documentation matches the function name
    if documentation.startswith(function_name):
        return True

    # Check for examples in the documentation
    return "# Example" in documentation or "```jldoctest" in documentation or "```julia-repl" in documentation

def filter_and_split_by_doc_quality(input_file, output_file, good_doc_file, bad_doc_file):
    """Processes the JSON file to filter functions and split based on documentation quality."""
    with open(input_file, "r") as f:
        data = json.load(f)

    total_functions = sum(len(entry["functions"]) for entry in data)
    non_standard_signature_count = 0
    non_fully_annotated_count = 0
    duplicate_functions_count = 0

    seen_bodies = set()
    filtered_data = []
    good_doc_data = []
    bad_doc_data = []

    total_good_functions = 0
    total_bad_functions = 0

    for entry in data:
        # Skip the project HumanEval.jl
        if entry["project"] == "HumanEval.jl":
            continue

        filtered_functions = []
        good_doc_functions = []
        bad_doc_functions = []

        for func in entry["functions"]:
            if func['type'] == 'basic':
                # Skip functions with "test" in the signature
                if "test" in func["signature"].lower():
                    continue
                
                func["body"] = remove_non_meaningful_comments(func["body"])
                normalized_body = normalize_function_body(func["body"])

                if normalized_body in seen_bodies:
                    duplicate_functions_count += 1
                    continue

                seen_bodies.add(normalized_body)

                if not func["signature"].strip().startswith("function"):
                    non_standard_signature_count += 1

                if not is_type_annotated(func["signature"]):
                    non_fully_annotated_count += 1
                else:
                    if count_code_lines(func["body"]) < 25:
                        filtered_functions.append(func)

                        # Separate into good and bad documentation
                        if has_good_documentation(func):
                            good_doc_functions.append(func)
                            total_good_functions += 1
                        else:
                            bad_doc_functions.append(func)
                            total_bad_functions += 1

        if filtered_functions:
            filtered_data.append({
                "author": entry["author"],
                "project": entry["project"],
                "functions": filtered_functions
            })

        if good_doc_functions:
            good_doc_data.append({
                "author": entry["author"],
                "project": entry["project"],
                "functions": good_doc_functions
            })

        if bad_doc_functions:
            bad_doc_data.append({
                "author": entry["author"],
                "project": entry["project"],
                "functions": bad_doc_functions
            })

    with open(output_file, "w") as f:
        json.dump(filtered_data, f, indent=4)
    print(f"Total functions after filtering: {sum(len(entry['functions']) for entry in filtered_data)}")

    with open(good_doc_file, "w") as f:
        json.dump(good_doc_data, f, indent=4)
    print(f"Number of functions with good documentation: {total_good_functions}")

    with open(bad_doc_file, "w") as f:
        json.dump(bad_doc_data, f, indent=4)
    print(f"Number of functions with bad documentation: {total_bad_functions}")

    print(f"Total functions before filtering: {total_functions}")
    print(f"Number of duplicate functions removed: {duplicate_functions_count}")
    print(f"Number of functions with non-standard signatures: {non_standard_signature_count}")
    print(f"Number of functions without type annotations: {non_fully_annotated_count}")

input_file = "data/combined_projects.json"  # Path to the input JSON file
output_file = "data/filtered_functions.json"  # Path to save all filtered functions
good_doc_file = "data/good_doc_functions.json"  # Path to save functions with good documentation
bad_doc_file = "data/bad_doc_functions.json"  # Path to save functions with bad documentation

filter_and_split_by_doc_quality(input_file, output_file, good_doc_file, bad_doc_file)
