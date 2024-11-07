import json
import zipfile
from pathlib import Path
from tree_sitter import Language, Parser
from clone import zipped_repos  # Assumes zipped_repos is in clone.py

# Load the Julia language grammar
JULIA_LANGUAGE = Language('build/my-languages.so', 'julia')

# Initialize a parser for Julia
parser = Parser()
parser.set_language(JULIA_LANGUAGE)
output_folder = Path("data/extracted_projects")
combined_data_path = Path("data/combined_projects.json")

# Function to extract function details from a node
def extract_function_details(node, code):
    code_segment = code[node.start_byte:node.end_byte]
    signature = code_segment.splitlines()[0] if code_segment.splitlines() else ""

    # Skip if 'test' appears in the function signature
    if "test" in signature.lower():
        return None

    doc_node = node.prev_sibling if node.prev_sibling and node.prev_sibling.type == 'string_literal' else None
    documentation = code[doc_node.start_byte:doc_node.end_byte] if doc_node else ""
    inline_comments = collect_inline_comments(node, code)
    body = get_cleaned_body(node, code)

    return {
        "signature": signature.strip(),
        "documentation": documentation.strip(),
        "inline_comments": inline_comments,
        "body": body
    }

# Function to collect inline comments recursively
def collect_inline_comments(node, code):
    inline_comments = []
    for child in node.children:
        if child.type == 'line_comment':
            inline_comments.append(code[child.start_byte:child.end_byte].strip())
        inline_comments.extend(collect_inline_comments(child, code))
    return inline_comments

# Function to extract the cleaned body of a function
def get_cleaned_body(node, code):
    body_lines = []
    for child in node.children:
        if child.type not in ('line_comment', 'block_comment'):
            line = code[child.start_byte:child.end_byte].strip()
            if line:
                body_lines.append(line)
    return "\n".join(body_lines)

# Recursive function to extract functions and handle nested modules
def extract_functions(node, code, module_name=""):
    functions = []
    for child in node.children:
        if child.type == "function_definition":
            func_details = extract_function_details(child, code)
            if func_details:  # Only add if func_details is not None
                if module_name:
                    func_details["module"] = module_name
                functions.append(func_details)
        elif child.type == "module":
            module_tokens = code[child.start_byte:child.end_byte].split()
            current_module_name = module_tokens[1] if len(module_tokens) > 1 else "unknown_module"
            functions.extend(extract_functions(child, code, module_name=current_module_name))
        else:
            functions.extend(extract_functions(child, code, module_name=module_name))
    return functions

# Process the zipped repositories and save each project’s data
def process_zipped_repos(zip_file):
    output_folder.mkdir(parents=True, exist_ok=True)
    project_data = []
    processed_developers = set()  # Track processed developers
    
    with zip_file as zf:
        for file_name in zf.namelist():
            if "test" in file_name.lower() or not file_name.endswith('.jl'):
                continue
            
            # Extract developer and project name from the path
            path_parts = file_name.split('/')
            if len(path_parts) < 3:
                continue
            
            developer = path_parts[0]
            project = path_parts[1]
            project_id = f"{developer}/{project}"
            
            # Limit processing to the first 10 unique developers
            if developer not in processed_developers:
                if len(processed_developers) >= 10:
                    break
                processed_developers.add(developer)

            # Only process files within this specific developer/project folder
            project_functions = []
            with zf.open(file_name) as file:
                code = file.read().decode('utf-8')
                try:
                    tree = parser.parse(bytes(code, "utf8"))
                    if tree.root_node.has_error:
                        print(f"Parsing error in {file_name}, skipping.")
                        continue
                    functions = extract_functions(tree.root_node, code)
                    project_functions.extend(functions)
                except ValueError as e:
                    print(f"Failed to parse {file_name}: {e}")
                    continue
            
            # Store functions and project information
            if project_functions:
                project_info = {
                    "project": project_id,
                    "functions": project_functions
                }
                
                # Save individual project JSON file
                project_file = output_folder / f"{developer}_{project}.json"
                with project_file.open("w") as json_file:
                    json.dump(project_info, json_file, indent=4)
                
                project_data.append(project_info)
    
    return project_data

# Save combined project data as a single JSON file
def save_combined_data(project_data):
    combined_data_path.parent.mkdir(parents=True, exist_ok=True)
    with combined_data_path.open("w") as json_file:
        json.dump(project_data, json_file, indent=4)

# Main function to execute the processing
def main():
    zip_file = zipped_repos()
    project_data = process_zipped_repos(zip_file)
    save_combined_data(project_data)

if __name__ == "__main__":
    main()
