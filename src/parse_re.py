import json
import re
from pathlib import Path
from typing import List, Dict
import sys
import time
import pandas as pd  
from pandarallel import pandarallel  # For parallel processing
import logging
import multiprocessing

# Configure logging to redirect messages to parsing_re_errors.txt
logging.basicConfig(
    filename='parsing_re_errors.txt',
    filemode='a',  # Append mode
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO  # Set to INFO to capture both INFO and ERROR messages
)

# Initialize pandarallel with 8 workers and enable progress bar
pandarallel.initialize(progress_bar=True, nb_workers=min(50, multiprocessing.cpu_count()-1))

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
                "documentation": documentation if documentation else "",
                "signature": f"function {func_name}({params})",
                "body": body,
                "inline_comments": inline_comments if inline_comments else [],
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
                    "documentation": documentation if documentation else "",
                    "signature": f"{func_name}({params}) = begin",
                    "body": body,
                    "inline_comments": inline_comments if inline_comments else [],
                    "file_path": file_path,
                    "line_number": index + 1 - len(body_lines) - 2,  # Adjust line number
                    "type": "single-line"
                })
            else:
                # Single-line expression
                body, comments = split_code_comment(expr)
                inline_comments.extend(comments)

                functions.append({
                    "documentation": documentation if documentation else "",
                    "signature": f"{func_name}({params}) =",
                    "body": body,
                    "inline_comments": inline_comments if inline_comments else [],
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
                    "documentation": documentation if documentation else "",
                    "signature": f"Anonymous Function assigned to {var_name}({params}) = begin",
                    "body": body,
                    "inline_comments": inline_comments if inline_comments else [],
                    "file_path": file_path,
                    "line_number": index + 1 - len(body_lines) - 2,  # Adjust line number
                    "type": "anonymous"
                })
            else:
                # Single-line expression
                body, comments = split_code_comment(expr)
                inline_comments.extend(comments)

                functions.append({
                    "documentation": documentation if documentation else "",
                    "signature": f"Anonymous Function assigned to {var_name}({params}) =",
                    "body": body,
                    "inline_comments": inline_comments if inline_comments else [],
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


def is_file_too_long(file_path: Path, max_lines: int = 10000) -> bool:
    """
    Checks if a file exceeds the maximum number of lines.

    Args:
        file_path (Path): Path to the file.
        max_lines (int): Maximum allowed lines.

    Returns:
        bool: True if file has more than max_lines, False otherwise.
    """
    try:
        with file_path.open('r', encoding='utf-8') as f:
            for i, _ in enumerate(f, 1):
                if i > max_lines:
                    return True
        return False
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return True  # Treat unreadable files as too long to skip


def process_author(author_path: Path) -> List[Dict]:
    """
    Process a single author: extract functions from all projects and files.

    Args:
        author_path (Path): Path to the author's directory.

    Returns:
        List[Dict]: A list of project dictionaries with extracted functions.
    """
    developer = author_path.name
    start_time = time.time()  # Start timer for the author
    project_entries = []

    try:
        # Iterate through each project under the author
        for project_path in author_path.iterdir():
            if not project_path.is_dir():
                continue  # Skip non-directory files

            project = project_path.name
            project_functions = []

            # Iterate through each .jl file in the project
            for file_path in project_path.rglob('*.jl'):
                # Check if file exceeds 10000 lines
                if is_file_too_long(file_path, max_lines=10000):
                    logging.info(f"Skipping {file_path} as it exceeds 10000 lines.")
                    continue

                try:
                    code = file_path.read_text(encoding='utf-8')
                except UnicodeDecodeError:
                    logging.error(f"UnicodeDecodeError in {file_path}, skipping.")
                    continue
                except Exception as e:
                    logging.error(f"Error reading {file_path}: {e}, skipping.")
                    continue

                # Extract functions from the code
                functions = extract_functions(code, str(file_path))
                project_functions.extend(functions)

            if project_functions:
                # Create a single project entry with all functions from its .jl files
                project_entry = {
                    "author": developer,
                    "project": project,
                    "functions": project_functions
                }
                project_entries.append(project_entry)

    except Exception as e:
        logging.error(f"Error processing author '{developer}': {e}")

    # Measure elapsed time
    elapsed = time.time() - start_time
    if elapsed > 10:
        logging.info(f"Author '{developer}' took {elapsed:.2f} seconds to process.")

    return project_entries


def process_repos(repo_path: Path, selected_authors: List[Path]) -> List[Dict]:
    """
    Process the repository by extracting functions from selected authors in parallel.

    Args:
        repo_path (Path): Path to the 'repo' directory.
        selected_authors (List[Path]): List of Paths to author directories.

    Returns:
        List[Dict]: A list of projects with their functions.
    """
    # Create a DataFrame of selected authors
    authors_df = pd.DataFrame({'author_path': selected_authors})

    # Apply the process_author function in parallel with progress bar
    authors_df['projects'] = authors_df.parallel_apply(lambda row: process_author(row['author_path']), axis=1)

    # Aggregate all project entries into a single list
    project_data = []
    for projects in authors_df['projects']:
        project_data.extend(projects)

    return project_data


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
        logging.error(f"Error saving data to JSON: {e}")


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
            if func['documentation'] and func['documentation'] != "":
                func_types[func_type]['documented'] += 1
            if func['inline_comments'] and func['inline_comments'] != []:
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
    Main function to process the repository and extract function data.
    """
    if len(sys.argv) != 2:
        print("Usage: python extract_julia_functions.py path_to_repo_folder")
        sys.exit(1)

    repo_path = Path(sys.argv[1])
    if not repo_path.is_dir():
        print(f"The repository folder {repo_path} does not exist or is not a directory.")
        sys.exit(1)

    try:
        # Extract list of unique authors (first-level directories)
        all_authors = [author for author in repo_path.iterdir() if author.is_dir()]
        selected_authors = sorted(all_authors)[:]  # Adjust the number as needed [:N]

        print(f"Selected {len(selected_authors)} authors for processing.")

        # Process the selected authors in parallel
        project_data = process_repos(repo_path, selected_authors)

    except Exception as e:
        logging.error(f"An error occurred while processing the repository: {e}")
        sys.exit(1)

    # Define the output JSON file path
    combined_data_path = Path("data/combined_projects.json")

    # Save the combined project data as a single JSON file
    save_combined_data(project_data, combined_data_path)

    # Compute and print metrics
    compute_metrics(project_data)


if __name__ == "__main__":
    main()
