# parse_julia_functions.py

from tree_sitter import Language, Parser
import os

# Define paths
BUILD_DIR = 'build'
LIB_NAME = 'julia.so'  # Change to 'julia.dll' on Windows or 'julia.dylib' on macOS if needed
LIB_PATH = os.path.join(BUILD_DIR, LIB_NAME)

# Ensure the language library exists
if not os.path.exists(LIB_PATH):
    raise FileNotFoundError(
        f"Language library not found at {LIB_PATH}. Please run build_languages.py first."
    )

# Load the Julia language
JULIA_LANGUAGE = Language(LIB_PATH, 'julia')

# Initialize the parser with Julia language
parser = Parser()
parser.set_language(JULIA_LANGUAGE)

# Define the Tree-sitter query directly within the script
# This query captures:
# 1. Docstrings (triple-quoted string literals)
# 2. Function signatures (name and parameters)
# 3. Function bodies
QUERY_TEXT = """
;; Capture function documentation as docstring
(
  (string_literal) @docstring
  (#match? @docstring "^\"\"\"")
)

;; Capture function signature
(
  (function_definition
    name: (identifier) @function_name
    parameters: (parameters) @parameters)
  @signature
)

;; Capture function body
(
  (function_definition
    body: (block) @body)
  @body
)
"""

# Compile the query
query = JULIA_LANGUAGE.query(QUERY_TEXT)

# Sample Julia code to parse
julia_code = """
\"\"\"
    greet(name)

Prints a greeting to the specified name.
\"\"\"
function greet(name)
    println("Hello, " * name * "!")
end

\"\"\"
    add(a, b)

Adds two numbers.
\"\"\"
function add(a, b)
    return a + b
end

# Function without documentation
function subtract(a, b)
    return a - b
end

# Single-line function
multiply(a, b) = a * b

\"\"\"
    compute_total!(data::DataFrame, verbose::Bool=true)

Computes the total with optional verbosity.
\"\"\"
compute_total!(data::DataFrame, verbose::Bool=true) = begin
    total = sum(data.values)
    println("Total computed.") if verbose
    return total
end
"""

# Encode the code to bytes, as Tree-sitter expects byte strings
code_bytes = julia_code.encode('utf8')

# Parse the code
tree = parser.parse(code_bytes)

# Execute the query on the syntax tree
captures = query.captures(tree.root_node)

# Organize captures
functions = []
current_docstring = None

# Sort captures based on node start_byte to maintain order
sorted_captures = sorted(captures, key=lambda c: c[0].start_byte)

for capture in sorted_captures:
    node, capture_name = capture
    text = code_bytes[node.start_byte:node.end_byte].decode('utf8').strip()

    if capture_name == 'docstring':
        # Store the current docstring
        current_docstring = text.strip('"""').strip()
    elif capture_name == 'signature':
        # Extract function name
        func_name_node = node.child_by_field_name('name')
        func_name = code_bytes[func_name_node.start_byte:func_name_node.end_byte].decode('utf8').strip() if func_name_node else "Unknown"

        # Extract parameters
        parameters_node = node.child_by_field_name('parameters')
        parameters = code_bytes[parameters_node.start_byte:parameters_node.end_byte].decode('utf8').strip() if parameters_node else "No parameters"

        # Initialize a new function entry
        current_function = {
            'function_name': func_name,
            'parameters': parameters,
            'documentation': current_docstring if current_docstring else 'No documentation'
        }

        # Reset current_docstring after associating it with a function
        current_docstring = None
    elif capture_name == 'body':
        # Extract function body
        body = text
        if 'function_name' in current_function:
            current_function['body'] = body
            functions.append(current_function)
            current_function = {}
        else:
            # Function body without signature (unlikely)
            functions.append({
                'function_name': 'Unknown',
                'parameters': 'Unknown',
                'documentation': 'No documentation',
                'body': body
            })

# Print the extracted information
for func in functions:
    print(f"Function Name: {func['function_name']}")
    print("Documentation:")
    print(func['documentation'])
    print("Signature:")
    print(f"function {func['function_name']}({func['parameters']})")
    print("Body:")
    print(func['body'])
    print("-" * 40)
