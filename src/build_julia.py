from tree_sitter import Language

# Define the path for the built library
BUILD_DIR = 'build'
LIB_NAME = 'julia.so'
LIB_PATH = f'{BUILD_DIR}/{LIB_NAME}'

# Ensure the build directory exists
import os
if not os.path.exists(BUILD_DIR):
    os.makedirs(BUILD_DIR)

# Build the language library
Language.build_library(
    # Store the library in the `build` directory
    LIB_PATH,

    # Include one or more languages
    [
        'tree-sitter-julia'
    ]
)

print(f"Built languages library at {LIB_PATH}")
