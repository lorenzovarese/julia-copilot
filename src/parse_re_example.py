import re

def extract_julia_functions(julia_code):
    functions = []
    lines = julia_code.split('\n')
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].strip()

        # Initialize documentation
        documentation = ""

        # Temporary storage for comments before a potential docstring
        temp_comments = []

        # Check for leading comments
        while i < n and lines[i].strip().startswith('#'):
            comment = lines[i].strip().strip('#').strip()
            temp_comments.append(comment)
            i += 1

        # After leading comments, check for docstring
        if i < n and lines[i].strip().startswith('"""'):
            docstring = []
            # Capture the entire docstring
            while i < n:
                doc_line = lines[i].strip()
                docstring.append(doc_line)
                if doc_line.endswith('"""') and len(docstring) > 1:
                    break
                i += 1
            # Extract docstring content without triple quotes and leading/trailing whitespace
            documentation = '\n'.join(docstring).strip('"""').strip()
            i += 1  # Move past the closing triple quotes
            # Ignore any comments after docstring and before function definition
        else:
            # No docstring, use the captured leading comments as documentation
            if temp_comments:
                documentation = '\n'.join(temp_comments)

        if i >= n:
            break

        # Now check for function definitions
        func_def = lines[i].strip()

        # Patterns for different function definitions
        patterns = {
            'standard': r'^function\s+(\w+)\s*\(([^)]*)\)',
            'single_line': r'^(\w+)\s*\(([^)]*)\)\s*=\s*(.*)',
            'anonymous': r'^(\w+)\s*=\s*\(([^)]*)\)\s*->\s*(.*)',
            'docstring_function': r'^function\s+(\w+)\s*\(([^)]*)\)',
        }

        match = None
        func_type = None
        for key, pattern in patterns.items():
            match = re.match(pattern, func_def)
            if match:
                func_type = key
                break

        if func_type == 'standard' or func_type == 'docstring_function':
            # Standard function with 'function' ... 'end'
            func_name = match.group(1)
            signature = lines[i].strip()
            i += 1
            body = []
            inline_comments = []
            while i < n:
                body_line = lines[i]
                stripped_line = body_line.strip()
                if stripped_line == 'end':
                    break
                body.append(body_line)
                # Check for inline comments
                comment_matches = re.findall(r'#(.*)', body_line)
                if comment_matches:
                    for comment in comment_matches:
                        inline_comments.append(comment.strip())
                i += 1
            body_text = '\n'.join(body).strip()
            functions.append({
                'name': func_name,
                'documentation': documentation if documentation else "None",
                'signature': signature,
                'body': body_text,
                'inline_comments': inline_comments if inline_comments else ["None"]
            })
            i += 1  # Skip 'end'
            documentation = ""  # Reset documentation
        elif func_type == 'single_line':
            # Single-line function definition
            func_name = match.group(1)
            params = match.group(2)
            expr = match.group(3)
            signature = lines[i].strip()
            # Extract inline comments from the expression
            body, inline_comments = split_code_comment(expr)
            functions.append({
                'name': func_name,
                'documentation': documentation if documentation else "None",
                'signature': signature,
                'body': body.strip(),
                'inline_comments': inline_comments if inline_comments else ["None"]
            })
            i += 1
            documentation = ""  # Reset documentation
        elif func_type == 'anonymous':
            # Anonymous function assigned to a variable
            var_name = match.group(1)
            params = match.group(2)
            expr = match.group(3)
            signature = lines[i].strip()
            # Extract inline comments from the expression
            body, inline_comments = split_code_comment(expr)
            functions.append({
                'name': var_name,
                'documentation': documentation if documentation else "None",
                'signature': signature,
                'body': body.strip(),
                'inline_comments': inline_comments if inline_comments else ["None"]
            })
            i += 1
            documentation = ""  # Reset documentation
        else:
            # Check for single-line functions without parameters, e.g., divide(a, b) = a / b  # comment
            single_line_pattern = r'^(\w+)\s*=\s*(.*)'
            match_single = re.match(single_line_pattern, func_def)
            if match_single:
                func_name = match_single.group(1)
                expr = match_single.group(2)
                signature = lines[i].strip()
                body, inline_comments = split_code_comment(expr)
                functions.append({
                    'name': func_name,
                    'documentation': documentation if documentation else "None",
                    'signature': signature,
                    'body': body.strip(),
                    'inline_comments': inline_comments if inline_comments else ["None"]
                })
                i += 1
                documentation = ""  # Reset documentation
            else:
                # Not a function definition
                i += 1

    return functions

def split_code_comment(line):
    """
    Splits a line into code and comment parts.
    Returns code, list of comments.
    """
    parts = line.split('#')
    code = parts[0].strip()
    comments = [part.strip() for part in parts[1:]] if len(parts) > 1 else []
    return code, comments

def main():
    julia_code = '''
# Function without documentation but with an attach comment that counts as documentation
function subtract(a, b)
    return a - b
end

# This should be counted
# as documentation too even if multiline
function test()
    return 1
end

# Another function with multiple comments
# serving as documentation
# for the multiply function
function multiply(a, b)
    # Inline comment inside function
    return a * b  # Another inline comment
end

# One-line function
divide(a, b) = a / b  # Inline comment at end

# Anonymous function assigned to a variable
add = (a, b) -> a + b  # Inline comment

# comment line (one or more) to be ignored
"""
    greet(name)

Prints a greeting to the specified name.
"""
function greet(name)
    println("Hello, " * name * "!")
end

# Function with docstring and multi-line body
"""
    compute_total!(data::DataFrame, verbose::Bool=true)

Computes the total with optional verbosity.
"""
compute_total!(data::DataFrame, verbose::Bool=true) = begin
    total = sum(data.values)
    println("Total computed.") if verbose
    return total
end

# Anonymous function with comment as documentation
anonymous_func = () -> println("Anonymous!")

# Function with no parameters and multiple inline comments
function hello_world()
    # Greet the world
    println("Hello, World!")  # Print statement
    # End of function
end

# Edge case: function with default parameter
function increment(x, step=1)
    return x + step
end

# Edge case: function with complex parameter types
function complex_function(a::Int, b::Float64, c::Vector{String}=["default"])
    # Processing
    return a + b
end
'''

    functions = extract_julia_functions(julia_code)

    # Print the extracted functions
    for idx, func in enumerate(functions, 1):
        print(f"Function {idx}: {func['name']}")
        print("Documentation:")
        print(func['documentation'] if func['documentation'] else "None")
        print("Signature:")
        print(func['signature'])
        print("Body:")
        print(func['body'])
        print("Inline Comments:")
        if func['inline_comments'] and func['inline_comments'] != ["None"]:
            for comment in func['inline_comments']:
                print(f"  - {comment}")
        else:
            print("None")
        print("-" * 40)

if __name__ == "__main__":
    main()

