"""Grammar-driven extraction of Julia functions via tree-sitter.

Rewritten against the tree-sitter >= 0.23 API. The old code called
``Language.build_library()`` and ``Parser.set_language()``, both removed in
0.22; the grammar is now supplied by the ``tree-sitter-julia`` wheel, so there
is no longer a compilation step or a ``build/julia.so`` artefact.

Slicing is done on ``bytes`` rather than ``str``. tree-sitter reports *byte*
offsets, so indexing a decoded string with them silently corrupts every file
containing non-ASCII characters -- which, in a corpus of scientific Julia code,
is a large share of them.
"""

from __future__ import annotations

import logging
import re
import zipfile
from functools import lru_cache

from julia_copilot.parsing import FunctionRecord, ProjectRecord

logger = logging.getLogger(__name__)

# ``\w`` is ASCII-only when matching bytes, which silently dropped every
# function with a non-ASCII name (common in scientific Julia: ``area``, ``sigma``,
# ...). Matching "not whitespace, not an open paren" keeps those.
BASIC_SIGNATURE_PATTERN = re.compile(rb"^\s*function\s+([^\s(]+\(.*?\))", re.MULTILINE)
SINGLE_LINE_SIGNATURE_PATTERN = re.compile(rb"^\s*([^\s(]+\(.*?\))\s*=")
ASSIGNMENT_FUNC_PATTERN = re.compile(rb"^\s*[^\s(]+\(.*?\)\s*=")


@lru_cache(maxsize=1)
def get_parser():
    """Build (once) a tree-sitter parser configured for Julia."""
    import tree_sitter_julia
    from tree_sitter import Language, Parser

    return Parser(Language(tree_sitter_julia.language()))


def _text(node, source: bytes) -> bytes:
    return source[node.start_byte : node.end_byte]


def collect_inline_comments(node, source: bytes) -> list[str]:
    """Collect every ``#`` comment beneath ``node``, depth-first."""
    comments: list[str] = []
    for child in node.children:
        if child.type == "line_comment":
            comments.append(_text(child, source).decode("utf-8", "replace").strip())
        comments.extend(collect_inline_comments(child, source))
    return comments


def _cleaned_body(segment: bytes, signature: bytes) -> str:
    """Return the body between the signature line and the closing ``end``.

    The closing ``end`` is dropped so that this backend agrees with
    :mod:`~julia_copilot.parsing.regex_parser`, whose pattern excludes it.
    :mod:`julia_copilot.encode` appends ``end`` itself when assembling the
    training string, so leaving it in would emit it twice.
    """
    body_lines: list[bytes] = []
    seen_signature = False
    for line in segment.splitlines():
        if not seen_signature:
            if signature and signature in line:
                seen_signature = True
            continue
        body_lines.append(line.strip())

    while body_lines and body_lines[-1] in (b"", b"end"):
        body_lines.pop()

    return b"\n".join(body_lines).decode("utf-8", "replace")


def extract_function_details(
    node, source: bytes, file_path: str, function_type: str = "basic"
) -> FunctionRecord | None:
    """Build a record for one function node, or ``None`` if it should be skipped."""
    segment = _text(node, source)

    if function_type == "basic":
        match = BASIC_SIGNATURE_PATTERN.search(segment)
        signature = match.group(0).strip() if match else b""
        body = _cleaned_body(segment, signature)
    elif function_type == "single-line":
        match = SINGLE_LINE_SIGNATURE_PATTERN.match(segment)
        signature = match.group(1).strip() if match else b""
        _, _, rhs = segment.partition(b"=")
        body = rhs.strip().decode("utf-8", "replace")
        signature = signature + b" =" if signature else signature
    else:
        return None

    signature_text = signature.decode("utf-8", "replace")
    if "test" in signature_text.lower():
        return None

    previous = node.prev_sibling
    documentation = (
        _text(previous, source).decode("utf-8", "replace").strip()
        if previous is not None and previous.type == "string_literal"
        else ""
    )

    return FunctionRecord(
        signature=signature_text,
        documentation=documentation,
        inline_comments=collect_inline_comments(node, source),
        body=body,
        file_path=file_path,
        line_number=source[: node.start_byte].count(b"\n") + 1,
        type=function_type,  # type: ignore[typeddict-item]
    )


def extract_anonymous_function(node, source: bytes, file_path: str) -> FunctionRecord | None:
    """Build a record for an ``args -> expr`` assignment."""
    segment = _text(node, source)
    if b"->" not in segment:
        return None

    _, _, rhs = segment.partition(b"->")
    return FunctionRecord(
        signature=segment.splitlines()[0].strip().decode("utf-8", "replace"),
        documentation="",  # anonymous functions are not documented in Julia
        inline_comments=collect_inline_comments(node, source),
        body=rhs.strip().decode("utf-8", "replace"),
        file_path=file_path,
        line_number=source[: node.start_byte].count(b"\n") + 1,
        type="anonymous",
    )


def extract_functions(
    node, source: bytes, file_path: str, module: str = ""
) -> list[FunctionRecord]:
    """Walk the syntax tree, recursing through nested modules."""
    functions: list[FunctionRecord] = []

    for child in node.children:
        record: FunctionRecord | None = None

        if child.type == "function_definition":
            record = extract_function_details(child, source, file_path, "basic")
        elif child.type == "assignment":
            segment = _text(child, source)
            if b"->" in segment:
                record = extract_anonymous_function(child, source, file_path)
            elif ASSIGNMENT_FUNC_PATTERN.match(segment):
                record = extract_function_details(child, source, file_path, "single-line")
        elif child.type == "module":
            tokens = _text(child, source).split()
            nested = tokens[1].decode("utf-8", "replace") if len(tokens) > 1 else "unknown_module"
            functions.extend(extract_functions(child, source, file_path, nested))
            continue
        else:
            functions.extend(extract_functions(child, source, file_path, module))
            continue

        if record is not None:
            if module:
                record["module"] = module
            functions.append(record)

    return functions


def process_zipped_repos(
    zip_file: zipfile.ZipFile, max_authors: int | None = None
) -> list[ProjectRecord]:
    """Extract functions from every Julia file inside ``zip_file``.

    Args:
        zip_file: Archive produced by :func:`julia_copilot.clone.zipped_repos`,
            laid out as ``author/project/...``.
        max_authors: Stop after this many distinct authors. ``None`` processes
            the whole archive; the original script was hard-coded to 10, which
            silently truncated the corpus.
    """
    parser = get_parser()
    projects: dict[str, ProjectRecord] = {}
    seen_authors: set[str] = set()

    with zip_file as archive:
        for name in archive.namelist():
            if not name.endswith(".jl") or "test" in name.lower():
                continue

            parts = name.split("/")
            if len(parts) < 3:
                continue
            author, project = parts[0], parts[1]

            if author not in seen_authors:
                if max_authors is not None and len(seen_authors) >= max_authors:
                    break
                seen_authors.add(author)

            with archive.open(name) as handle:
                source = handle.read()

            try:
                tree = parser.parse(source)
            except ValueError as exc:
                logger.warning("Failed to parse %s: %s", name, exc)
                continue

            if tree.root_node.has_error:
                logger.debug("Syntax error in %s, skipping.", name)
                continue

            key = f"{author}/{project}"
            entry = projects.setdefault(
                key, ProjectRecord(author=author, project=project, functions=[])
            )
            entry["functions"].extend(extract_functions(tree.root_node, source, name))

    return list(projects.values())
