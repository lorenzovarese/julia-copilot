# This script is only run once to create the language library
from tree_sitter import Language

# Build the Julia language library
Language.build_library(
    'build/my-languages.so', 
    ['tree-sitter-julia']
)
