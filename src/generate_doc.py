import json
import random
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Load the JSON file
with open("data/combined_projects.json", 'r') as file:
    functions_data = json.load(file)

# Filter functions with type "basic"
basic_functions = [func for project in functions_data for func in project["functions"] if func["type"] == "basic"]


# Sample 10% of the basic functions
sample_size = 1 # max(1, int(len(basic_functions) * 0.05))  # Ensure at least one function is sampled
sampled_functions = random.sample(basic_functions, sample_size)

# Few-shot examples for prompting OpenAI API
FEW_SHOT_PROMPT = """
--------------------- Example 1 --------------------- 
\"\"\"prompt
 "\"\"\" Check if in given vector of numbers, are any two numbers closer to each other than\ngiven threshold.\n>>> has_close_elements([1.0, 2.0, 3.0], 0.5)\nfalse\n>>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\ntrue\"\"\"\nfunction has_close_elements(numbers::Vector{Float64}, threshold::Float64)::Bool \n"
```

```julia-function
\"\"\"
    has_close_elements(numbers::Vector{Float64}, threshold::Float64)::Bool

Check if in given list of `numbers`, are any two numbers closer to each other than
given `threshold`.

# Examples

```jldoctest
julia> has_close_elements([1.0, 2.0, 3.0], 0.5)
false

julia> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
true
```
\"\"\"
function has_close_elements(numbers::Vector{Float64}, threshold::Float64)::Bool
    for (i, x) in enumerate(numbers)
        for (j, y) in enumerate(numbers)
            if i != j
                distance = abs(x - y)
                if distance < threshold
                    return true
                end
            end
        end
    end

    return false
end```

--------------------- Example 2 --------------------- 
```prompt
"\"\"\"Given a positive integer n, you have to make a pile of n levels of stones.\nThe first level has n stones.\nThe number of stones in the next level is:\n    - the next odd number if n is odd.\n    - the next even number if n is even.\nReturn the number of stones in each level in a vector, where element at index\ni represents the number of stones in the level (i+1).\nExamples:\n>>> make_a_pile(3)\n[3, 5, 7]\"\"\"\nfunction make_a_pile(n::Int64)::Vector{Int64} \n"
```

```julia-function
\"\"\"
    separate_paren_groups(paren_string::String)::Vector{String}

Input to this function is a string containing multiple groups of nested
parentheses. Your goal is to separate those group into separate strings and
return the list of those. Separate groups are balanced (each open brace is
properly closed) and not nested within each other Ignore any spaces in the input
string.

# Examples

```jldoctest
julia> separate_paren_groups("( ) (( )) (( )( ))")
3-element Vector{String}:
 "()"
 "(())"
 "(()())"
```
\"\"\"
function separate_paren_groups(paren_string::String)::Vector{String}
    result = []
    current_string = []
    current_depth = 0

    for c in paren_string
        if c == '('
            current_depth += 1
            push!(current_string, c)
        elseif c == ')'
            current_depth -= 1
            push!(current_string, c)

            if current_depth == 0
                push!(result, join(current_string))
                empty!(current_string)
            end
        end
    end

    return result
end```
"""

# Configure quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True
)

# Load a quantized model and tokenizer
model_name = "bigscience/bloom-560m"  # You can use a different smaller model if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=quantization_config
)

# Generate documentation using the quantized model
def generate_documentation_with_model(function):
    """Generates Julia-style documentation using a quantized model."""
    prompt = FEW_SHOT_PROMPT + f"""
    --------------------- New Function --------------------- 
    \"\"\"{function['documentation']}\n{function['body']}\"\"\"
    function {function['signature']}"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        top_p=0.95
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Save the generated documentation to files
output_dir = "sampled_functions"
os.makedirs(output_dir, exist_ok=True)

for idx, func in enumerate(sampled_functions, 1):
    documentation = generate_documentation_with_model(func)
    file_path = os.path.join(output_dir, f"function_{idx}.jl")
    with open(file_path, 'w') as file:
        file.write(documentation)

print(f"Sampled {sample_size} functions. Documentation saved in the '{output_dir}' directory.")
