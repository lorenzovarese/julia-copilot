"""Tests for the corpus filtering heuristics."""

from __future__ import annotations

from julia_copilot.filtering import (
    count_code_lines,
    extract_function_name,
    filter_corpus,
    has_good_documentation,
    is_type_annotated,
    normalize_function_body,
    remove_non_meaningful_comments,
)


def _func(signature, body, documentation="", kind="basic"):
    return {
        "signature": signature,
        "body": body,
        "documentation": documentation,
        "inline_comments": [],
        "type": kind,
    }


def _project(functions, project="Demo.jl"):
    return {"author": "alice", "project": project, "functions": functions}


def test_count_code_lines_ignores_comments_and_blanks():
    assert count_code_lines("a = 1\n\n# a comment\nb = 2\n") == 2


def test_is_type_annotated():
    assert is_type_annotated("function f(x::Int)")
    assert not is_type_annotated("function f(x)")


def test_remove_non_meaningful_comments():
    body = "# ---\n# this one is meaningful\nx = 1\n"
    kept = remove_non_meaningful_comments(body)

    assert "# ---" not in kept
    assert "meaningful" in kept
    assert "x = 1" in kept


def test_normalize_strips_indentation_so_duplicates_collide():
    assert normalize_function_body("    x = 1\n    return x\n") == normalize_function_body(
        "x = 1\nreturn x\n"
    )


def test_extract_function_name():
    assert extract_function_name("function add(x::Int)") == "add"
    assert extract_function_name("add(x) = x") is None


def test_documentation_is_good_when_it_restates_the_name():
    assert has_good_documentation(_func("function add(x::Int)", "x", "add(x) adds things"))


def test_documentation_is_good_when_it_has_an_example():
    assert has_good_documentation(
        _func("function add(x::Int)", "x", "Sums stuff.\n```jldoctest\nadd(1)\n```")
    )


def test_documentation_is_bad_when_terse():
    assert not has_good_documentation(_func("function add(x::Int)", "x", "helper"))


def test_filter_drops_untyped_and_non_basic_functions():
    # Bodies must differ: deduplication runs before the annotation check, so
    # identical bodies would be counted as duplicates instead.
    functions = [
        _func("function typed(x::Int)", "return x", "typed(x) returns x"),
        _func("function untyped(x)", "return x + 1", "untyped(x) returns x"),
        _func("lambda(x::Int) =", "return x + 2", "lambda", kind="single-line"),
    ]
    filtered, good, bad, stats = filter_corpus([_project(functions)])

    assert stats.kept == 1
    assert stats.not_annotated == 1
    assert filtered[0]["functions"][0]["signature"] == "function typed(x::Int)"
    assert len(good) == 1
    assert bad == []


def test_filter_removes_duplicate_bodies_across_projects():
    body = "return x + 1"
    projects = [
        _project([_func("function inc(x::Int)", body, "inc(x)")], project="A.jl"),
        _project([_func("function bump(x::Int)", f"    {body}", "bump(x)")], project="B.jl"),
    ]
    _, _, _, stats = filter_corpus(projects)

    assert stats.duplicates == 1
    assert stats.kept == 1


def test_filter_excludes_the_benchmark_project():
    projects = [_project([_func("function f(x::Int)", "x", "f(x)")], project="HumanEval.jl")]
    filtered, _, _, stats = filter_corpus(projects)

    assert filtered == []
    assert stats.kept == 0


def test_filter_drops_test_functions():
    projects = [_project([_func("function test_add(x::Int)", "x", "test_add(x)")])]
    _, _, _, stats = filter_corpus(projects)

    assert stats.kept == 0
