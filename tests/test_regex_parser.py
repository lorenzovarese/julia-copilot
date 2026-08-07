"""Tests for the regex-based Julia function extractor."""

from __future__ import annotations

from julia_copilot.parsing.regex_parser import (
    extract_functions,
    format_metrics,
    split_code_comment,
)

MULTI_LINE = '''"""
    add(x::Int, y::Int)

Add two numbers.
"""
function add(x::Int, y::Int)
    return x + y  # trivial
end
'''

ONE_LINE = "square(x::Int) = x * x\n"

ANONYMOUS = "double = (x) -> 2 * x\n"


def test_extracts_multi_line_function():
    (func,) = extract_functions(MULTI_LINE, "add.jl")

    assert func["type"] == "basic"
    assert func["signature"] == "function add(x::Int, y::Int)"
    assert "return x + y" in func["body"]
    assert "Add two numbers." in func["documentation"]
    assert func["file_path"] == "add.jl"


def test_extracts_inline_comments():
    (func,) = extract_functions(MULTI_LINE, "add.jl")
    assert any("trivial" in comment for comment in func["inline_comments"])


def test_extracts_one_line_function():
    (func,) = extract_functions(ONE_LINE, "square.jl")

    assert func["type"] == "single-line"
    assert func["signature"] == "square(x::Int) ="
    assert func["body"] == "x * x"
    assert func["file_path"] == "square.jl"


def test_extracts_anonymous_function():
    (func,) = extract_functions(ANONYMOUS, "double.jl")

    assert func["type"] == "anonymous"
    assert "double" in func["signature"]
    assert func["body"] == "2 * x"
    assert func["file_path"] == "double.jl"


def test_leading_comments_become_documentation():
    code = "# Adds one\n# to its argument\nfunction inc(x)\n    x + 1\nend\n"
    (func,) = extract_functions(code, "inc.jl")
    assert func["documentation"] == "Adds one\nto its argument"


def test_no_functions_in_plain_code():
    assert extract_functions("x = 1\ny = 2\n", "consts.jl") == []


def test_split_code_comment():
    assert split_code_comment("x + 1  # add one") == ("x + 1", ["add one"])
    assert split_code_comment("x + 1") == ("x + 1", [])


def test_format_metrics_reports_counts():
    projects = [
        {
            "author": "alice",
            "project": "Demo.jl",
            "functions": extract_functions(MULTI_LINE, "add.jl"),
        }
    ]
    report = format_metrics(projects)

    assert "Number of Authors:   1" in report
    assert "Number of Functions: 1" in report
    assert "Normal Function" in report


def test_unicode_function_names_are_extracted():
    """``[A-Za-z_]`` dropped every non-ASCII name; scientific Julia is full of them."""
    code = "function área(r::Float64)\n    return 3.14 * r^2\nend\n"
    (func,) = extract_functions(code, "area.jl")

    assert func["signature"] == "function área(r::Float64)"


def test_function_is_not_emitted_twice():
    """``^\\s*`` could span newlines, matching a function below the cursor."""
    code = '"""\n    doc\n"""\nfunction f(x::Int)\n    return x\nend\n\nsquare(x::Int) = x * x\n'
    funcs = extract_functions(code, "mixed.jl")

    signatures = [f["signature"] for f in funcs]
    assert len(signatures) == len(set(signatures))
    assert signatures == ["function f(x::Int)", "square(x::Int) ="]
