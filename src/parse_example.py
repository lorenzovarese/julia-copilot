import re
from tree_sitter import Language, Parser

# Load the Julia language grammar
JULIA_LANGUAGE = Language('build/my-languages.so', 'julia')

# Initialize a parser for Julia
parser = Parser()
parser.set_language(JULIA_LANGUAGE)

# Example Julia code snippet
code = """
function reinitialize_containers!(mesh::ParallelT8codeMesh, equations, dg::DGSEM, cache)
    @unpack elements, interfaces, boundaries, mortars, mpi_interfaces, mpi_mortars,
    mpi_cache = cache
    resize!(elements, ncells(mesh))
    init_elements!(elements, mesh, dg.basis)

    count_required_surfaces!(mesh)
    required = count_required_surfaces(mesh)

    resize!(interfaces, required.interfaces)

    resize!(boundaries, required.boundaries)

    resize!(mortars, required.mortars)

    resize!(mpi_interfaces, required.mpi_interfaces)

    resize!(mpi_mortars, required.mpi_mortars)

    mpi_mesh_info = (mpi_mortars = mpi_mortars,
                     mpi_interfaces = mpi_interfaces,

                     # Temporary arrays for updating `mpi_cache`.
                     global_mortar_ids = fill(UInt64(0), nmpimortars(mpi_mortars)),
                     global_interface_ids = fill(UInt64(0), nmpiinterfaces(mpi_interfaces)),
                     neighbor_ranks_mortar = Vector{Vector{Int}}(undef,
                                                                 nmpimortars(mpi_mortars)),
                     neighbor_ranks_interface = fill(-1, nmpiinterfaces(mpi_interfaces)))

    fill_mesh_info!(mesh, interfaces, mortars, boundaries,
                    mesh.boundary_names; mpi_mesh_info = mpi_mesh_info)

    init_mpi_cache!(mpi_cache, mesh, mpi_mesh_info, nvariables(equations), nnodes(dg),
                    eltype(elements))

    empty!(mpi_mesh_info.global_mortar_ids)
    empty!(mpi_mesh_info.global_interface_ids)
    empty!(mpi_mesh_info.neighbor_ranks_mortar)
    empty!(mpi_mesh_info.neighbor_ranks_interface)

    # Re-initialize and distribute normal directions of MPI mortars; requires
    # MPI communication, so the MPI cache must be re-initialized beforehand.
    init_normal_directions!(mpi_mortars, dg.basis, elements)
    exchange_normal_directions!(mpi_mortars, mpi_cache, mesh, nnodes(dg))

    return nothing
end

# Compatibility to `dgsem_p4est/containers.jl`.
function init_mpi_interfaces!(interfaces, mesh::ParallelT8codeMesh)
    # Do nothing.
    return nothing
end

# Compatibility to `dgsem_p4est/containers.jl`.
function init_mpi_mortars!(mortars, mesh::ParallelT8codeMesh)
    # Do nothing.
    return nothing
end

# Compatibility to `dgsem_p4est/containers_parallel.jl`.
function init_mpi_mortars!(mpi_mortars, mesh::ParallelT8codeMesh, basis, elements)
    # Do nothing.
    return nothing
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
def extract_function_details(node, code, function_type="basic"):
    code_segment = code[node.start_byte:node.end_byte]
    
    # Regular expression to capture Julia function definition and single-line functions
    if function_type == "basic":
        signature_match = re.search(r'^\s*function\s+(\w+\(.*?\))', code_segment, re.MULTILINE)
    elif function_type == "single-line":
        signature_match = re.search(r'^\s*(\w+\(.*?\))\s*=', code_segment)
    else:
        signature_match = None

    signature = signature_match.group(0).strip() if signature_match else ""

    # Extract the body of the function
    if function_type == "single-line":
        body = code_segment.split("=", 1)[1].strip() if "=" in code_segment else ""
    else:
        body = code_segment
    
    return {
        "signature": signature,
        "body": body,
        "type": function_type
    }

# Function to extract and print function information
def extract_and_print_functions(node, code):
    functions = []
    for child in node.children:
        if child.type == "function_definition":
            # Basic function with "function ... end" syntax
            func_details = extract_function_details(child, code, function_type="basic")
            if func_details:
                functions.append(func_details)
        elif child.type == "assignment":
            # Single-line function detection
            code_segment = code[child.start_byte:child.end_byte]
            if re.match(r'^\s*\w+\(.*?\)\s*=', code_segment):
                func_details = extract_function_details(child, code, function_type="single-line")
                if func_details:
                    functions.append(func_details)

    for func in functions:
        print("\nExtracted Function Details:")
        print(f"Signature: {func['signature']}")
        print(f"Body: {func['body']}")
        print(f"Type: {func['type']}")

# Print the syntax tree
print("Syntax Tree:")
print_tree(tree.root_node, code)

# Extract and print function details
extract_and_print_functions(tree.root_node, code)
