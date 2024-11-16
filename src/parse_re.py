import json
import re
from pathlib import Path
from typing import List, Dict
import zipfile
import sys

# Define regex patterns

# Documentation: Captures only docstrings (""" ... """) preceding a function
DOC_PATTERN = re.compile(r'''
    (?P<doc>
        ^\s*\"\"\"(?:.|\n)*?\"\"\"\n+     # Docstring with triple quotes
    )
''', re.MULTILINE | re.VERBOSE)

# Multi-line function definition (function ... end)
MULTI_LINE_FUNC_PATTERN = re.compile(r'''
    ^\s*function\s+                             # function keyword
    (?P<name>[A-Za-z_][\w!?@#$%^&*+\-]*)\s*     # function name with special characters
    \(\s*(?P<params>[^\)]*)\s*\)\s*\n          # parameters within parentheses
    (?P<body>(?:.|\n)*?)                       # non-greedy capture of the body
    ^\s*end\b                                   # end keyword
''', re.MULTILINE | re.VERBOSE)

# One-line function definition
ONE_LINE_FUNC_PATTERN = re.compile(r'''
    ^\s*
    (?P<name>[A-Za-z_][\w!?@#$%^&*+\-]*)\s*     # function name with special characters
    \(\s*(?P<params>[^\)]*)\s*\)\s*=\s*         # parameters within parentheses followed by =
    (?P<body>.*)                                # function body
''', re.MULTILINE | re.VERBOSE)

# Anonymous function pattern (assignment to a variable)
ANONYMOUS_FUNC_PATTERN = re.compile(r'''
    ^\s*
    (?P<var>[A-Za-z_][\w!?@#$%^&*+\-]*)\s*=\s* # variable name with special characters and assignment
    \(\s*(?P<params>[^\)]*)\s*\)\s*->\s*       # parameters within parentheses followed by ->
    (?P<body>.*)                                # function body expression
''', re.MULTILINE | re.VERBOSE)

def extract_functions(code: str, file_path: str) -> List[Dict]:
    """
    Extract functions from Julia code using regular expressions.

    Args:
        code (str): The Julia code as a string.
        file_path (str): The path to the file being processed.

    Returns:
        List[Dict]: A list of dictionaries containing function details.
    """
    functions = []

    # Preprocess code to ensure consistent line endings
    code = code.replace('\r\n', '\n').replace('\r', '\n')

    # Split code into lines for processing
    lines = code.split('\n')
    total_lines = len(lines)
    index = 0

    while index < total_lines:
        # Initialize documentation
        documentation = ""

        # Temporary storage for comments before a potential docstring
        temp_comments = []

        # Check for leading comments
        while index < total_lines and lines[index].strip().startswith('#'):
            comment = lines[index].strip().strip('#').strip()
            temp_comments.append(comment)
            index += 1

        # After leading comments, check for docstring
        if index < total_lines and lines[index].strip().startswith('"""'):
            docstring = []
            # Capture the entire docstring
            while index < total_lines:
                doc_line = lines[index].strip()
                docstring.append(doc_line)
                if doc_line.endswith('"""') and len(docstring) > 1:
                    break
                index += 1
            # Extract docstring content without triple quotes and leading/trailing whitespace
            documentation = '\n'.join(docstring).strip('"""').strip()
            index += 1  # Move past the closing triple quotes
            # Ignore any comments after docstring and before function definition
        else:
            # No docstring, use the captured leading comments as documentation
            if temp_comments:
                documentation = '\n'.join(temp_comments)

        if index >= total_lines:
            break

        # Reconstruct the remaining code from the current index
        remaining_code = '\n'.join(lines[index:])

        # Attempt to match multi-line function
        multi_match = MULTI_LINE_FUNC_PATTERN.match(remaining_code)
        if multi_match:
            func_name = multi_match.group('name')
            params = multi_match.group('params').strip()
            body = multi_match.group('body').strip()

            # Extract inline comments within the body
            inline_comments = re.findall(r'#.*', body)

            functions.append({
                "documentation": documentation if documentation else "None",
                "signature": f"function {func_name}({params})",
                "body": body,
                "inline_comments": inline_comments if inline_comments else ["None"],
                "file_path": file_path,
                "line_number": index + 1,
                "type": "basic"
            })

            # Calculate the number of lines consumed by the function
            body_lines = body.split('\n')
            # +2 for 'function' and 'end' lines
            index += len(body_lines) + 2
            continue

        # Attempt to match one-line function
        one_line_match = ONE_LINE_FUNC_PATTERN.match(remaining_code)
        if one_line_match:
            func_name = one_line_match.group('name')
            params = one_line_match.group('params').strip()
            expr = one_line_match.group('body').strip()

            # Initialize inline_comments before using it
            inline_comments = []

            # Extract inline comments within the body
            # Check if the expression starts with 'begin'
            if expr.startswith('begin'):
                # Multi-line body
                body_lines = []
                # Remove 'begin' from the expression
                expr = expr[len('begin'):].strip()
                if expr:
                    body_lines.append(expr)
                index += 1  # Move past the current line

                while index < total_lines:
                    body_line = lines[index]
                    stripped_line = body_line.strip()
                    if stripped_line == 'end':
                        break
                    body_lines.append(body_line)
                    # Check for inline comments
                    comment_matches = re.findall(r'#(.*)', body_line)
                    if comment_matches:
                        for comment in comment_matches:
                            inline_comments.append(comment.strip())
                    index += 1
                body_text = '\n'.join(body_lines).strip()
                body = f"begin\n{body_text}\nend"
                index += 1  # Skip 'end'
                # inline_comments already populated

                functions.append({
                    "documentation": documentation if documentation else "None",
                    "signature": f"{func_name}({params}) = begin",
                    "body": body,
                    "inline_comments": inline_comments if inline_comments else ["None"],
                    "file_path": file_path,
                    "line_number": index + 1 - len(body_lines) - 2,  # Adjust line number
                    "type": "single-line"
                })
            else:
                # Single-line expression
                body, comments = split_code_comment(expr)
                inline_comments.extend(comments)

                functions.append({
                    "documentation": documentation if documentation else "None",
                    "signature": f"{func_name}({params}) =",
                    "body": body,
                    "inline_comments": inline_comments if inline_comments else ["None"],
                    "file_path": file_path,
                    "line_number": index + 1,
                    "type": "single-line"
                })
                index += 1
            continue

        # Attempt to match anonymous function
        anon_match = ANONYMOUS_FUNC_PATTERN.match(lines[index])
        if anon_match:
            var_name = anon_match.group('var')
            params = anon_match.group('params').strip()
            expr = anon_match.group('body').strip()

            # Initialize inline_comments before using it
            inline_comments = []

            # Extract inline comments within the body
            comment_matches = re.findall(r'#(.*)', expr)
            if comment_matches:
                for comment in comment_matches:
                    inline_comments.append(comment.strip())

            # Check if the expression starts with 'begin'
            if expr.startswith('begin'):
                # Multi-line body
                body_lines = []
                # Remove 'begin' from the expression
                expr = expr[len('begin'):].strip()
                if expr:
                    body_lines.append(expr)
                index += 1  # Move past the current line

                while index < total_lines:
                    body_line = lines[index]
                    stripped_line = body_line.strip()
                    if stripped_line == 'end':
                        break
                    body_lines.append(body_line)
                    # Check for inline comments
                    comment_matches = re.findall(r'#(.*)', body_line)
                    if comment_matches:
                        for comment in comment_matches:
                            inline_comments.append(comment.strip())
                    index += 1
                body_text = '\n'.join(body_lines).strip()
                body = f"begin\n{body_text}\nend"
                index += 1  # Skip 'end'
                # inline_comments already populated

                functions.append({
                    "documentation": documentation if documentation else "None",
                    "signature": f"Anonymous Function assigned to {var_name}({params}) = begin",
                    "body": body,
                    "inline_comments": inline_comments if inline_comments else ["None"],
                    "file_path": file_path,
                    "line_number": index + 1 - len(body_lines) - 2,  # Adjust line number
                    "type": "anonymous"
                })
            else:
                # Single-line expression
                body, comments = split_code_comment(expr)
                inline_comments.extend(comments)

                functions.append({
                    "documentation": documentation if documentation else "None",
                    "signature": f"Anonymous Function assigned to {var_name}({params}) =",
                    "body": body,
                    "inline_comments": inline_comments if inline_comments else ["None"],
                    "file_path": file_path,
                    "line_number": index + 1,
                    "type": "anonymous"
                })
                index += 1
            continue

        # No match found, move to the next line
        index += 1

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

def process_zipped_repos(zip_file: zipfile.ZipFile) -> List[Dict]:
    """
    Process a zipped repository containing Julia projects.

    Args:
        zip_file (zipfile.ZipFile): The zip file object to process.

    Returns:
        List[Dict]: A list of projects with their functions.
    """
    output_folder = Path("data/extracted_projects")
    output_folder.mkdir(parents=True, exist_ok=True)
    project_data = {}
    processed_developers = set()  # Track processed developers

    for file_name in zip_file.namelist():
        if "test" in file_name.lower() or not file_name.endswith('.jl'):
            continue

        # Extract developer and project name from the path
        path_parts = file_name.split('/')
        if len(path_parts) < 3:
            continue

        developer = path_parts[0].strip()
        project = path_parts[1].strip()
        project_key = f"{developer}/{project}"

        # Determine if this developer should be processed
        if developer not in processed_developers:
            if len(processed_developers) >= 100:
                # Already processed 100 developers, skip new developers
                continue
            processed_developers.add(developer)

        # Only process files within the first 100 developers
        if developer not in processed_developers:
            # This check is redundant due to the above, but kept for clarity
            continue

        # Now, process the file
        with zip_file.open(file_name) as file:
            try:
                code = file.read().decode('utf-8')
            except UnicodeDecodeError:
                print(f"UnicodeDecodeError in {file_name}, skipping.")
                continue

            # Extract functions using regex
            functions = extract_functions(code, file_path=file_name)

            # Add functions to the existing project entry if it exists, or create a new one
            if project_key not in project_data:
                project_data[project_key] = {
                    "author": developer,
                    "project": project,
                    "functions": []
                }
            project_data[project_key]["functions"].extend(functions)

    return list(project_data.values())

def save_combined_data(project_data: List[Dict], output_path: Path):
    """
    Save the combined project data to a JSON file.

    Args:
        project_data (List[Dict]): The project data to save.
        output_path (Path): The path to the output JSON file.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(project_data, f, indent=4, ensure_ascii=False)
        print(f"Data successfully saved to {output_path}")
    except Exception as e:
        print(f"Error saving data to JSON: {e}")

def compute_metrics(project_data: List[Dict]):
    """
    Compute and print metrics based on the extracted project data.

    Args:
        project_data (List[Dict]): The list of projects with their functions.
    """
    # Count unique authors
    authors = set()
    for project in project_data:
        authors.add(project['author'])
    num_authors = len(authors)

    # Number of projects
    num_projects = len(project_data)

    # Number of functions
    num_functions = sum(len(project['functions']) for project in project_data)

    # Initialize counters
    func_types = {
        "basic": {"count": 0, "documented": 0, "commented": 0},
        "single-line": {"count": 0, "documented": 0, "commented": 0},
        "anonymous": {"count": 0, "documented": 0, "commented": 0}
    }

    for project in project_data:
        for func in project['functions']:
            func_type = func['type']
            if func_type not in func_types:
                continue
            func_types[func_type]['count'] += 1
            if func['documentation'] and func['documentation'] != "None":
                func_types[func_type]['documented'] += 1
            if func['inline_comments'] and func['inline_comments'] != ["None"]:
                func_types[func_type]['commented'] += 1

    # Print overall metrics
    print("\n=== Overall Metrics ===")
    print(f"Number of Authors: {num_authors}")
    print(f"Number of Projects: {num_projects}")
    print(f"Number of Functions: {num_functions}")

    # Print function type metrics in tabular format
    print("\n=== Function Type Metrics ===")
    header = f"{'Function Type':<25} {'Count':<10} {'% Documented':<15} {'% Commented':<15}"
    print(header)
    print("-" * len(header))
    for f_type, data in func_types.items():
        count = data['count']
        if count == 0:
            pct_doc = pct_com = 0
        else:
            pct_doc = (data['documented'] / count) * 100
            pct_com = (data['commented'] / count) * 100
        type_name = {
            "basic": "Normal Function",
            "single-line": "One-Line Function",
            "anonymous": "Anonymous Function"
        }.get(f_type, f_type.capitalize())
        print(f"{type_name:<25} {count:<10} {pct_doc:<15.2f} {pct_com:<15.2f}")
    print()

def main():
    """
    Main function to process zipped Julia repositories and extract function data.
    """
    if len(sys.argv) != 2:
        print("Usage: python extract_julia_functions.py path_to_zip_file.zip")
        sys.exit(1)

    zip_path = sys.argv[1]
    zip_file = Path(zip_path)
    if not zip_file.is_file():
        print(f"The zip file {zip_path} does not exist.")
        sys.exit(1)

    try:
        with zipfile.ZipFile(zip_file, 'r') as zf:
            project_data = process_zipped_repos(zf)
    except zipfile.BadZipFile:
        print(f"The file {zip_path} is not a valid zip file.")
        sys.exit(1)

    # Define the output JSON file path
    combined_data_path = Path("data/combined_projects.json")

    # Save the combined project data as a single JSON file
    save_combined_data(project_data, combined_data_path)

    # Compute and print metrics
    compute_metrics(project_data)

if __name__ == "__main__":
    main()
