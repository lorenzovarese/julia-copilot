import json
import re
from pathlib import Path
from tree_sitter import Language, Parser
from clone import zipped_repos

# Load the Julia language grammar
JULIA_LANGUAGE = Language('build/julia.so', 'julia')

# Initialize a parser for Julia
parser = Parser()
parser.set_language(JULIA_LANGUAGE)
output_folder = Path("data/extracted_projects")
combined_data_path = Path("data/combined_projects.json")

# Function to extract function details from a node
def extract_function_details(node, code, file_path, function_type="basic"):
    code_segment = code[node.start_byte:node.end_byte]
    
    # Define the signature and body based on function type
    if function_type == "basic":
        # Standard function with `function ... end` syntax
        signature_match = re.search(r'^\s*function\s+(\w+\(.*?\))', code_segment, re.MULTILINE)
        signature = signature_match.group(0).strip() if signature_match else ""
        body = get_cleaned_body(node, code, signature)
    elif function_type == "single-line":
        # Single-line function with `name(args) = expression` syntax
        signature_match = re.match(r'^\s*(\w+\(.*?\))\s*=', code_segment)
        signature = signature_match.group(1).strip() if signature_match else ""
        body = code_segment.split("=", 1)[1].strip() if "=" in code_segment else ""
    else:
        signature = ""
        body = ""

    # Skip if 'test' appears in the function signature
    if "test" in signature.lower():
        return None

    # Documentation extraction
    doc_node = node.prev_sibling if node.prev_sibling and node.prev_sibling.type == 'string_literal' else None
    documentation = code[doc_node.start_byte:doc_node.end_byte] if doc_node else ""
    
    # Extract inline comments
    inline_comments = collect_inline_comments(node, code)
    
    # Calculate the line number based on byte position
    line_number = code[:node.start_byte].count('\n') + 1

    return {
        "signature": signature,
        "documentation": documentation.strip(),
        "inline_comments": inline_comments,
        "body": body,
        "file_path": file_path,
        "line_number": line_number,
        "type": function_type
    }

# Function to extract the cleaned body of a function without the signature
def get_cleaned_body(node, code, signature):
    body_lines = []
    signature_found = False
    for line in code[node.start_byte:node.end_byte].splitlines():  # Removed .decode("utf-8")
        if signature in line:
            signature_found = True
            continue
        if signature_found:
            body_lines.append(line.strip())  # Only add lines after the signature line
    return "\n".join(body_lines)

# Function to collect inline comments recursively
def collect_inline_comments(node, code):
    inline_comments = []
    for child in node.children:
        if child.type == 'line_comment':
            inline_comments.append(code[child.start_byte:child.end_byte].strip())
        inline_comments.extend(collect_inline_comments(child, code))
    return inline_comments

# Recursive function to extract functions and handle nested modules
def extract_functions(node, code, file_path, module_name=""):
    functions = []
    for child in node.children:
        if child.type == "function_definition":
            func_details = extract_function_details(child, code, file_path, function_type="basic")
            if func_details:
                if module_name:
                    func_details["module"] = module_name
                functions.append(func_details)
        elif child.type == "assignment":
            # Check for anonymous or single-line functions
            code_segment = code[child.start_byte:child.end_byte]
            if "->" in code_segment:
                # Anonymous function in assignment
                func_details = extract_anonymous_function(child, code)
                if func_details:
                    func_details["file_path"] = file_path
                    func_details["type"] = "anonymous"
                    if module_name:
                        func_details["module"] = module_name
                    functions.append(func_details)
            elif re.match(r'^\s*\w+\(.*?\)\s*=', code_segment):
                # Single-line function with `name(args) = expression` syntax
                func_details = extract_function_details(child, code, file_path, function_type="single-line")
                if func_details:
                    if module_name:
                        func_details["module"] = module_name
                    functions.append(func_details)
        elif child.type == "module":
            module_tokens = code[child.start_byte:child.end_byte].split()
            current_module_name = module_tokens[1] if len(module_tokens) > 1 else "unknown_module"
            functions.extend(extract_functions(child, code, file_path, module_name=current_module_name))
        else:
            functions.extend(extract_functions(child, code, file_path, module_name=module_name))
    return functions

# Function to extract details for anonymous functions
def extract_anonymous_function(node, code):
    code_segment = code[node.start_byte:node.end_byte]
    if "->" not in code_segment:
        return None
    
    signature = code_segment.splitlines()[0].strip()
    body = code_segment.split("->", 1)[1].strip()  # Extract body after `->`
    inline_comments = collect_inline_comments(node, code)

    return {
        "signature": signature,
        "documentation": "",  # Anonymous functions typically lack documentation
        "inline_comments": inline_comments,
        "body": body,
        "type": "anonymous"
    }

# Process the zipped repositories and save each project’s data
def process_zipped_repos(zip_file):
    output_folder.mkdir(parents=True, exist_ok=True)
    project_data = {}
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
            project_key = f"{developer}/{project}"
            
            # Limit processing to the first 10 unique developers
            if developer not in processed_developers:
                if len(processed_developers) >= 10:
                    break
                processed_developers.add(developer)

            # Only process files within this specific developer/project folder
            with zf.open(file_name) as file:
                code = file.read().decode('utf-8')
                try:
                    tree = parser.parse(bytes(code, "utf8"))
                    if tree.root_node.has_error:
                        print(f"Parsing error in {file_name}, skipping.")
                        continue
                    # Extract functions
                    functions = extract_functions(tree.root_node, code, file_path=file_name)
                    
                    # Add functions to the existing project entry if it exists, or create a new one
                    if project_key not in project_data:
                        project_data[project_key] = {
                            "author": developer,
                            "project": project,
                            "functions": []
                        }
                    project_data[project_key]["functions"].extend(functions)
                except ValueError as e:
                    print(f"Failed to parse {file_name}: {e}")
                    continue
    
    return list(project_data.values())

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
