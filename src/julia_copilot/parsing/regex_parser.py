"""Regex-based extraction of Julia functions.

Deliberately line-oriented rather than grammar-driven: a large fraction of the
scraped corpus does not parse cleanly, and a tolerant scanner recovers far more
functions than tree-sitter does. See
:mod:`julia_copilot.parsing.treesitter_parser` for the strict alternative.

Three declaration forms are recognised::

    function name(args)      # -> "basic"
        ...
    end

    name(args) = expr        # -> "single-line"

    name = (args) -> expr    # -> "anonymous"
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import pandas as pd

from julia_copilot.config import init_pandarallel
from julia_copilot.parsing import FunctionRecord, FunctionType, ProjectRecord

logger = logging.getLogger(__name__)

MAX_FILE_LINES = 10_000
"""Files longer than this are generated or vendored; parsing them is wasted work."""

# Julia identifiers admit a lot of punctuation (`push!`, `<:`, `+`, ...) and may
# start with any Unicode letter -- scientific Julia is full of `sigma`, `area`,
# `Delta`. ``[^\W\d]`` is "word character that is not a digit", i.e. a letter or
# underscore in any script; the original ``[A-Za-z_]`` dropped all of them.
_NAME = r"[^\W\d][\w!?@#$%^&*+\-]*"

# Leading whitespace is matched with ``[ \t]`` rather than ``\s``: ``\s`` also
# matches newlines, so an anchored ``.match()`` could skip ahead and attribute a
# function found several lines below to the scanner's current position -- which
# emitted that function twice, once here and once when the cursor reached it.

MULTI_LINE_FUNC_PATTERN = re.compile(
    rf"""
    ^[ \t]*function[ \t]+
    (?P<name>{_NAME})[ \t]*
    \([ \t]*(?P<params>[^)]*)[ \t]*\)[ \t]*\n
    (?P<body>(?:.|\n)*?)        # non-greedy: stop at the first `end`
    ^[ \t]*end\b
    """,
    re.MULTILINE | re.VERBOSE,
)

ONE_LINE_FUNC_PATTERN = re.compile(
    rf"""
    ^[ \t]*
    (?P<name>{_NAME})[ \t]*
    \([ \t]*(?P<params>[^)]*)[ \t]*\)[ \t]*=[ \t]*
    (?P<body>.*)
    """,
    re.MULTILINE | re.VERBOSE,
)

ANONYMOUS_FUNC_PATTERN = re.compile(
    rf"""
    ^[ \t]*
    (?P<var>{_NAME})[ \t]*=[ \t]*
    \([ \t]*(?P<params>[^)]*)[ \t]*\)[ \t]*->[ \t]*
    (?P<body>.*)
    """,
    re.MULTILINE | re.VERBOSE,
)

COMMENT_PATTERN = re.compile(r"#(.*)")


def split_code_comment(line: str) -> tuple[str, list[str]]:
    """Split ``line`` into its code part and any trailing comments."""
    code, *comments = line.split("#")
    return code.strip(), [c.strip() for c in comments]


def _read_documentation(lines: list[str], index: int) -> tuple[str, int]:
    """Consume leading comments and/or a docstring starting at ``index``.

    Returns the documentation text and the index of the first line after it.
    A ``\"\"\"...\"\"\"`` docstring wins over preceding ``#`` comments, matching
    Julia's own convention.
    """
    total = len(lines)
    comments: list[str] = []

    while index < total and lines[index].strip().startswith("#"):
        comments.append(lines[index].strip().strip("#").strip())
        index += 1

    if index < total and lines[index].strip().startswith('"""'):
        docstring: list[str] = []
        while index < total:
            line = lines[index].strip()
            docstring.append(line)
            if line.endswith('"""') and len(docstring) > 1:
                break
            index += 1
        index += 1  # step past the closing quotes
        return "\n".join(docstring).strip('"').strip(), index

    return "\n".join(comments), index


def _read_begin_block(lines: list[str], index: int, first: str) -> tuple[str, list[str], int]:
    """Read a ``begin ... end`` body starting after ``begin`` on the current line.

    Returns the reconstructed body, the inline comments found in it, and the
    index of the line following ``end``.
    """
    body_lines = [first] if first else []
    comments: list[str] = []
    index += 1

    while index < len(lines):
        line = lines[index]
        if line.strip() == "end":
            break
        body_lines.append(line)
        comments.extend(c.strip() for c in COMMENT_PATTERN.findall(line))
        index += 1

    body = "begin\n" + "\n".join(body_lines).strip() + "\nend"
    return body, comments, index + 1


def extract_functions(code: str, file_path: str) -> list[FunctionRecord]:
    """Extract every recognised function declaration from ``code``.

    Args:
        code: Contents of a Julia source file.
        file_path: Path recorded on each result, for traceability.

    Returns:
        One record per function found, in source order.
    """
    functions: list[FunctionRecord] = []
    lines = code.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    total = len(lines)
    index = 0

    while index < total:
        documentation, index = _read_documentation(lines, index)
        if index >= total:
            break

        remaining = "\n".join(lines[index:])

        multi_match = MULTI_LINE_FUNC_PATTERN.match(remaining)
        if multi_match:
            body = multi_match.group("body").strip()
            functions.append(
                FunctionRecord(
                    documentation=documentation,
                    signature=f"function {multi_match['name']}({multi_match['params'].strip()})",
                    body=body,
                    inline_comments=re.findall(r"#.*", body),
                    file_path=file_path,
                    line_number=index + 1,
                    type="basic",
                )
            )
            index += len(body.split("\n")) + 2  # + `function` and `end` lines
            continue

        one_line_match = ONE_LINE_FUNC_PATTERN.match(remaining)
        if one_line_match:
            index = _append_assignment_form(
                functions,
                lines,
                index,
                documentation,
                file_path=file_path,
                name=one_line_match["name"],
                params=one_line_match["params"].strip(),
                expr=one_line_match["body"].strip(),
                kind="single-line",
            )
            continue

        anon_match = ANONYMOUS_FUNC_PATTERN.match(lines[index])
        if anon_match:
            index = _append_assignment_form(
                functions,
                lines,
                index,
                documentation,
                file_path=file_path,
                name=anon_match["var"],
                params=anon_match["params"].strip(),
                expr=anon_match["body"].strip(),
                kind="anonymous",
            )
            continue

        index += 1

    return functions


def _append_assignment_form(
    functions: list[FunctionRecord],
    lines: list[str],
    index: int,
    documentation: str,
    *,
    file_path: str,
    name: str,
    params: str,
    expr: str,
    kind: FunctionType,
) -> int:
    """Append a ``name(args) = ...`` or ``name = (args) -> ...`` record.

    Handles both the single-expression and the ``begin ... end`` block form.
    Returns the index of the next unconsumed line.
    """
    label = f"Anonymous Function assigned to {name}" if kind == "anonymous" else name

    if expr.startswith("begin"):
        start = index
        body, comments, index = _read_begin_block(lines, index, expr[len("begin") :].strip())
        functions.append(
            FunctionRecord(
                documentation=documentation,
                signature=f"{label}({params}) = begin",
                body=body,
                inline_comments=comments,
                file_path=file_path,
                line_number=start + 1,
                type=kind,
            )
        )
        return index

    body, comments = split_code_comment(expr)
    functions.append(
        FunctionRecord(
            documentation=documentation,
            signature=f"{label}({params}) =",
            body=body,
            inline_comments=comments,
            file_path=file_path,
            line_number=index + 1,
            type=kind,
        )
    )
    return index + 1


def is_file_too_long(file_path: Path, max_lines: int = MAX_FILE_LINES) -> bool:
    """Whether ``file_path`` exceeds ``max_lines``; unreadable files count as too long."""
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            return any(i > max_lines for i, _ in enumerate(handle, 1))
    except OSError as exc:
        logger.error("Error reading %s: %s", file_path, exc)
        return True


def process_author(author_path: Path) -> list[ProjectRecord]:
    """Extract functions from every project belonging to one GitHub author."""
    developer = author_path.name
    started = time.time()
    projects: list[ProjectRecord] = []

    for project_path in sorted(p for p in author_path.iterdir() if p.is_dir()):
        functions: list[FunctionRecord] = []

        for file_path in project_path.rglob("*.jl"):
            if is_file_too_long(file_path):
                logger.debug("Skipping %s: over %d lines.", file_path, MAX_FILE_LINES)
                continue
            try:
                code = file_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                logger.debug("Skipping %s: %s", file_path, exc)
                continue
            functions.extend(extract_functions(code, str(file_path)))

        if functions:
            projects.append(
                ProjectRecord(author=developer, project=project_path.name, functions=functions)
            )

    elapsed = time.time() - started
    if elapsed > 10:
        logger.debug("Author %r took %.2fs to process.", developer, elapsed)

    return projects


def process_repos(authors: list[Path]) -> list[ProjectRecord]:
    """Run :func:`process_author` across ``authors`` in parallel."""
    init_pandarallel(progress_bar=True)
    frame = pd.DataFrame({"author_path": authors})
    frame["projects"] = frame.parallel_apply(lambda row: process_author(row.author_path), axis=1)
    return [project for projects in frame["projects"] for project in projects]


def compute_metrics(project_data: list[ProjectRecord]) -> dict[str, dict[str, int]]:
    """Summarise how many functions carry documentation or inline comments."""
    counters: dict[str, dict[str, int]] = {
        kind: {"count": 0, "documented": 0, "commented": 0}
        for kind in ("basic", "single-line", "anonymous")
    }

    for project in project_data:
        for func in project["functions"]:
            bucket = counters.get(func.get("type", ""))
            if bucket is None:
                continue
            bucket["count"] += 1
            if func.get("documentation"):
                bucket["documented"] += 1
            if func.get("inline_comments"):
                bucket["commented"] += 1

    return counters


def format_metrics(project_data: list[ProjectRecord]) -> str:
    """Render :func:`compute_metrics` as a plain-text table."""
    counters = compute_metrics(project_data)
    labels = {
        "basic": "Normal Function",
        "single-line": "One-Line Function",
        "anonymous": "Anonymous Function",
    }

    lines = [
        "=== Overall Metrics ===",
        f"Number of Authors:   {len({p['author'] for p in project_data})}",
        f"Number of Projects:  {len(project_data)}",
        f"Number of Functions: {sum(len(p['functions']) for p in project_data)}",
        "",
        "=== Function Type Metrics ===",
        f"{'Function Type':<25}{'Count':<10}{'% Documented':<15}{'% Commented':<15}",
        "-" * 65,
    ]
    for kind, data in counters.items():
        count = data["count"]
        documented = 100 * data["documented"] / count if count else 0.0
        commented = 100 * data["commented"] / count if count else 0.0
        lines.append(f"{labels[kind]:<25}{count:<10}{documented:<15.2f}{commented:<15.2f}")

    return "\n".join(lines)
