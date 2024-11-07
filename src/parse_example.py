from tree_sitter import Language, Parser
import json

# Load the Julia language (ignoring deprecation warning)
JULIA_LANGUAGE = Language('build/my-languages.so', 'julia')

# Initialize a parser for Julia
parser = Parser()
parser.set_language(JULIA_LANGUAGE)

# Example Julia source code
julia_code = """
function add(a, b)
    # This is an inline comment
    if 1 == 1:
     # This is an indented comment
     return #end of line comment
    end
    # Ciao 2
    return a + b
end

\"\"\"
 This is the doc
 
 Example:
 >> hi.jl
    Hi!
\"\"\"
function multiply(a, b)
    return a * b
end
"""

# Parse the source code
tree = parser.parse(bytes(julia_code, "utf8"))
root_node = tree.root_node

# Function to recursively print the syntax tree for debugging
def print_tree(node, code, indent=""):
    print(f"{indent}{node.type} [{node.start_point} - {node.end_point}]")
    for child in node.children:
        print_tree(child, code, indent + "  ")

print("Parsed Syntax Tree:")
print_tree(root_node, julia_code)

# Helper function to clean and extract the body of the function, excluding comments
def get_cleaned_body(node, code):
    body_lines = []
    for child in node.children:
        if child.type not in ('line_comment', 'block_comment'):
            line = code[child.start_byte:child.end_byte].strip()
            if line:
                body_lines.append(line)
    return "\n".join(body_lines)

# Helper function to recursively collect all inline comments within a function body
def collect_inline_comments(node, code):
    inline_comments = []
    for child in node.children:
        if child.type == 'line_comment':
            inline_comments.append(code[child.start_byte:child.end_byte].strip())
        # Recursively collect comments from nested structures
        inline_comments.extend(collect_inline_comments(child, code))
    return inline_comments

# Helper function to extract documentation, inline comments, signature, and cleaned function body
def extract_function_details(node, code):
    # Extract the function signature
    signature = code[node.start_byte:node.end_byte].splitlines()[0]
    
    # Check for documentation (multi-line strings just before the function)
    doc_node = node.prev_sibling if node.prev_sibling and node.prev_sibling.type == 'string_literal' else None
    documentation = code[doc_node.start_byte:doc_node.end_byte] if doc_node else ""

    # Collect all inline comments within the function body, including indented ones
    inline_comments = collect_inline_comments(node, code)

    # Get cleaned body without comments
    body = get_cleaned_body(node, code)

    return {
        "signature": signature.strip(),
        "documentation": documentation.strip(),
        "inline_comments": inline_comments,
        "body": body
    }

# Extract all functions from the parsed tree
def extract_functions(node, code):
    functions = []
    for child in node.children:
        if child.type == "function_definition":
            func_details = extract_function_details(child, code)
            functions.append(func_details)
        # Recursively search for functions in children nodes
        functions.extend(extract_functions(child, code))
    return functions

# Generate project data
project_data = {
    "project": "unique_project_qualifier_name",
    "functions": extract_functions(root_node, julia_code)
}

# Print JSON output
print("\nJSON Output:")
print(json.dumps([project_data], indent=4))

# Optionally save to file
with open("extracted_functions.json", "w") as json_file:
    json.dump([project_data], json_file, indent=4)
