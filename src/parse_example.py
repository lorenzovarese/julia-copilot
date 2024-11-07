
from tree_sitter import Language, Parser
import json

# Load the Julia language grammar
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

module CDSAPI

using HTTP
using JSON
using Base64

\"\"\"
    retrieve(name, params, filename; max_sleep = 120.)

Retrieves data for `name` from the Climate Data Store
with the specified `params` and stores it in the current
directory as `filename`.

The client periodically requests the status of the retrieve request.
`max_sleep` is the maximum time (in seconds) between the status updates.
\"\"\"
function retrieve(name, params, filename; max_sleep = 120.)
    creds = Dict()
    open(joinpath(homedir(),".cdsapirc")) do f
        for line in readlines(f)
            key, val = strip.(split(line,':', limit=2))
            creds[key] = val
        end
    end

    apikey = string("Basic ", base64encode(creds["key"]))
    response = HTTP.request(
        "POST",
        creds["url"] * "/resources/$name",
        ["Authorization" => apikey],
        body=JSON.json(params),
        verbose=1)

    resp_dict = JSON.parse(String(response.body))
    data = Dict("state" => "queued")
    sleep_seconds = 1.

    while data["state"] != "completed"
        data = HTTP.request("GET", creds["url"] * "/tasks/" * string(resp_dict["request_id"]),  ["Authorization" => apikey])
        data = JSON.parse(String(data.body))
        println("request queue status ", data["state"])

        if data["state"] == "failed"
            error("Request to dataset $name failed. Check " *
                  "https://cds.climate.copernicus.eu/cdsapp#!/yourrequests " *
                  "for more information (after login).")
        end

        sleep_seconds = min(1.5 * sleep_seconds,max_sleep)
        if data["state"] != "completed"
            sleep(sleep_seconds)
        end
    end

    HTTP.download(data["location"], filename)
    return data
end

\"\"\"
    py2ju(dictstr)

Takes a Python dictionary as string and converts it into Julia's `Dict`

# Examples
```julia-repl
julia> str = \"""{
               'format': 'zip',
               'variable': 'surface_air_temperature',
               'product_type': 'climatology',
               'month': '08',
               'origin': 'era_interim',
           }\""";

julia> CDSAPI.py2ju(str)
Dict{String,Any} with 5 entries:
  "format"       => "zip"
  "month"        => "08"
  "product_type" => "climatology"
  "variable"     => "surface_air_temperature"
  "origin"       => "era_interim"

```
\"\"\"
function py2ju(dictstr)
    dictstr_cpy = replace(dictstr, "'" => "\"")
    lastcomma_pos = findlast(",", dictstr_cpy).start

    # if there's no pair after the last comma
    if findnext(":", dictstr_cpy, lastcomma_pos) == nothing
        # remove the comma
        dictstr_cpy = dictstr_cpy[firstindex(dictstr_cpy):(lastcomma_pos - 1)] * dictstr_cpy[(lastcomma_pos + 1):lastindex(dictstr_cpy)]
    end

    # removes trailing comma from a list
    rx = r",[ \n\r\t]*\]"
    dictstr_cpy = replace(dictstr_cpy, rx => "]")

    return JSON.parse(dictstr_cpy)
end

end # module
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

def extract_function_details(node, code):
    """
    Extracts details of a Julia function from the AST node.
    """
    # Safely extract the signature line, if available
    code_segment = code[node.start_byte:node.end_byte]
    signature = code_segment.splitlines()[0] if code_segment.splitlines() else ""

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

def collect_inline_comments(node, code):
    """
    Recursively collects all inline comments within a function body.
    """
    inline_comments = []
    for child in node.children:
        if child.type == 'line_comment':
            inline_comments.append(code[child.start_byte:child.end_byte].strip())
        inline_comments.extend(collect_inline_comments(child, code))
    return inline_comments

def get_cleaned_body(node, code):
    """
    Extracts the cleaned function body, excluding comments.
    """
    body_lines = []
    for child in node.children:
        if child.type not in ('line_comment', 'block_comment'):
            line = code[child.start_byte:child.end_byte].strip()
            if line:
                body_lines.append(line)
    return "\n".join(body_lines)

def extract_functions(node, code, module_name=""):
    """
    Recursively extracts all functions from the AST node, handling nested modules.
    """
    functions = []
    for child in node.children:
        if child.type == "function_definition":
            func_details = extract_function_details(child, code)
            if module_name:
                func_details["module"] = module_name
            functions.append(func_details)
        elif child.type == "module":
            # Process functions within this module
            module_tokens = code[child.start_byte:child.end_byte].split()
            current_module_name = module_tokens[1] if len(module_tokens) > 1 else "unknown_module"
            functions.extend(extract_functions(child, code, module_name=current_module_name))
        else:
            # Continue searching within other nodes for nested functions
            functions.extend(extract_functions(child, code, module_name=module_name))
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
