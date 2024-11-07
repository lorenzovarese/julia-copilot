import json
from llama_cpp import Llama

# Path to the downloaded GGUF model file
# Link: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf
model_path = "model/Phi-3-mini-4k-instruct-q4.gguf"

# Load the model with specific parameters
llm = Llama(
    model_path=model_path, # path to GGUF file
    n_ctx=2048,  # The max sequence length to use - note that longer sequence lengths require much more resources
    n_threads=8,      # Set based on the CPU capability; for MacBook Pro, 8 is suitable
    n_gpu_layers=0    # Set to 0 for CPU-only
)

def generate_function_description(function):
    """
    Uses Phi-3 Mini model to generate a concise function description.
    :param function: Dictionary containing 'signature', 'body', and 'inline_comments'.
    :return: Generated function description as a string.
    """
    # Format the prompt according to Phi-3 Mini's chat format
    prompt = f"""<|user|>
Generate a clear and concise description for the following function:

Signature:
{function["signature"]}

Inline Comments:
{" ".join(function["inline_comments"])}

Function Body:
{function["body"]}

<|end|>
<|assistant|>"""

    # Run the model with the prompt to generate a response
    response = llm(
        prompt,
        max_tokens=256,      # Limit the response length
        stop=["<|end|>"],    # Stop at end-of-assistant tag
        echo=False            # Whether to include the prompt in output
    )
    return response['choices'][0]['text'].strip()

# Example function data to test the documentation generation
function_data = [
    {
        "project": "sample_project",
        "functions": [
            {
                "signature": "function add(a, b)",
                "documentation": "",
                "generated_doc": "",
                "inline_comments": [
                    "# Adds two numbers and returns the result",
                    "# Example usage: add(5, 10)"
                ],
                "body": "function add(a, b)\n    return a + b\nend"
            }
        ]
    },
    {
    "project": "complex_project",
    "functions": [
        {
            "signature": "function transform_to_exponential_with_phase_shift(z::Vector{Complex{T}}, phase_shift::T) where T<:Real",
            "documentation": "",
            "generated_doc": "",
            "inline_comments": [
            ],
            "body": "function transform_to_exponential_with_phase_shift(z::Vector{Complex{T}}, phase_shift::T) where T<:Real\n    z_transformed = @. abs(z) * exp(im * (angle(z) + phase_shift))\n    return z_transformed\nend"
        }
    ]
}

]

# Generate documentation for each function
for project in function_data:
    for func in project["functions"]:
        description = generate_function_description(func)
        func["generated_doc"] = description  # Store the generated documentation

# Print the updated JSON with generated documentation
print(json.dumps(function_data, indent=4))

'''
Result:
[
    {
        "project": "sample_project",
        "functions": [
            {
                "signature": "function add(a, b)",
                "documentation": "",
                "generated_doc": "`add(a, b)` is a function designed to take two numeric arguments, `a` and `b`, and return their sum. It is straightforward, with an in-line comment indicating its purpose and an example usage to demonstrate the function call with `add(5, 10)`.",
                "inline_comments": [
                    "# Adds two numbers and returns the result",
                    "# Example usage: add(5, 10)"
                ],
                "body": "function add(a, b)\n    return a + b\nend"
            }
        ]
    },
    {
        "project": "complex_project",
        "functions": [
            {
                "signature": "function transform_to_exponential_with_phase_shift(z::Vector{Complex{T}}, phase_shift::T) where T<:Real",
                "documentation": "",
                "generated_doc": "This `transform_to_exponential_with_phase_shift` function converts a given vector of complex numbers (`z`) into a new vector representing an exponential function of the original complex numbers, with a specified phase shift. The function takes two arguments: a vector of complex numbers `z` and a real-valued `phase_shift`. The transformation applies the absolute value of `z` and then applies a complex exponential function, incorporating the phase shift, to each element of `z`. The result is a new vector where each complex number from the original `z` vector is transformed into its exponential form by scaling with its absolute value and adjusting the phase by the provided phase shift.",
                "inline_comments": [],
                "body": "function transform_to_exponential_with_phase_shift(z::Vector{Complex{T}}, phase_shift::T) where T<:Real\n    z_transformed = @. abs(z) * exp(im * (angle(z) + phase_shift))\n    return z_transformed\nend"
            }
        ]
    }
]
'''
