import re
from tree_sitter import Language, Parser

# Load the Julia language grammar
JULIA_LANGUAGE = Language('build/my-languages.so', 'julia')

# Initialize a parser for Julia
parser = Parser()
parser.set_language(JULIA_LANGUAGE)

# Example Julia code snippet
code = """
const ε = 1e-19  

function mysqrt(a)
  x = 1.0
  while true
    y = (x + a / x) / 2.0
    if abs(x-y) < ε
      break
    end
    x = y
  end
  x
end

function pad(str, n)
  str = string(str)
  len = length(str)
  str = str * " " ^ (n-len)
end
"""

# Parse the code
tree = parser.parse(bytes(code, "utf8"))

# Function to print the syntax tree
def print_tree(node, code, indent=""):
    print(f"{indent}{node.type} [{node.start_byte}, {node.end_byte}]")
    for child in node.children:
        print_tree(child, code, indent + "  ")

# Function to extract function details from a node
def extract_function_details(node, code):
    code_segment = code[node.start_byte:node.end_byte].decode('utf8')
    
    # Use regex to capture Julia function definition
    signature_match = re.search(r'^\s*function\s+(\w+\(.*?\))', code_segment, re.MULTILINE)
    signature = signature_match.group(0).strip() if signature_match else ""
    
    # Documentation extraction
    doc_node = node.prev_sibling if node.prev_sibling and node.prev_sibling.type == 'string_literal' else None
    documentation = code[doc_node.start_byte:doc_node.end_byte].decode('utf8') if doc_node else ""
    
    # Extract the body of the function
    body = code_segment
    
    return {
        "signature": signature,
        "documentation": documentation.strip(),
        "body": body,
    }

# Function to extract and print function information
def extract_and_print_functions(node, code):
    functions = []
    for child in node.children:
        if child.type == "function_definition":
            func_details = extract_function_details(child, code)
            if func_details:
                functions.append(func_details)
    for func in functions:
        print("\nExtracted Function Details:")
        print(f"Signature: {func['signature']}")
        print(f"Documentation: {func['documentation']}")
        print(f"Body: {func['body']}")

# Print the syntax tree
print("Syntax Tree:")
print_tree(tree.root_node, code)

# Extract and print function details
extract_and_print_functions(tree.root_node, bytes(code, "utf8"))
