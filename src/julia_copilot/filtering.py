"""Filter the extracted corpus and split it by documentation quality.

The heuristics are unchanged from the original ``utils/filter_functions.py``;
what changed is that running this module no longer executes a hard-coded
pipeline at import time.

Retained functions must be ``type == "basic"``, type-annotated, shorter than
:data:`MAX_CODE_LINES`, free of the substring ``test`` in their signature, and
unique after normalisation. "Good" documentation either opens with the function
name or contains a runnable example block.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from julia_copilot.config import EXCLUDED_PROJECTS
from julia_copilot.parsing import FunctionRecord, ProjectRecord

logger = logging.getLogger(__name__)

MAX_CODE_LINES = 25
"""Longer bodies rarely fit the context window once the docstring is prepended."""

MIN_COMMENT_LETTERS = 3
"""A comment with fewer letters than this is a separator or commented-out code."""

EXAMPLE_MARKERS = ("# Example", "```jldoctest", "```julia-repl")

FUNCTION_NAME_PATTERN = re.compile(r"function\s+([A-Za-z_]\w*)\(")


@dataclass
class FilterStats:
    """Counts collected while filtering, reported at the end of the run."""

    total: int = 0
    kept: int = 0
    good_doc: int = 0
    bad_doc: int = 0
    duplicates: int = 0
    non_standard_signature: int = 0
    not_annotated: int = 0
    extra: dict[str, int] = field(default_factory=dict)

    def render(self) -> str:
        return "\n".join(
            [
                f"Total functions before filtering:      {self.total:,}",
                f"Total functions after filtering:       {self.kept:,}",
                f"With good documentation:               {self.good_doc:,}",
                f"With poor documentation:               {self.bad_doc:,}",
                f"Duplicates removed:                    {self.duplicates:,}",
                f"Non-standard signatures:               {self.non_standard_signature:,}",
                f"Without type annotations:              {self.not_annotated:,}",
            ]
        )


def count_code_lines(body: str) -> int:
    """Count non-empty, non-comment lines in a function body."""
    return sum(1 for line in body.split("\n") if line.strip() and not line.strip().startswith("#"))


def is_type_annotated(signature: str) -> bool:
    """Whether the signature carries at least one Julia type annotation."""
    return "::" in signature


def remove_non_meaningful_comments(body: str) -> str:
    """Drop comment lines with fewer than :data:`MIN_COMMENT_LETTERS` letters."""
    kept = [
        line
        for line in body.split("\n")
        if not line.strip().startswith("#")
        or len(re.findall(r"[a-zA-Z]", line)) >= MIN_COMMENT_LETTERS
    ]
    return "\n".join(kept)


def normalize_function_body(body: str) -> str:
    """Normalise a body for duplicate detection: no comments, no indentation."""
    body = remove_non_meaningful_comments(body)
    lines = (line.strip() for line in body.split("\n"))
    return "\n".join(line for line in lines if line and not line.startswith("#"))


def extract_function_name(signature: str) -> str | None:
    """Pull the function name out of a ``function name(...)`` signature."""
    match = FUNCTION_NAME_PATTERN.match(signature)
    return match.group(1) if match else None


def has_good_documentation(func: FunctionRecord) -> bool:
    """Whether a docstring is worth training on.

    Good means it either restates the function name (Julia's documented
    convention) or contains an example block.
    """
    documentation = func.get("documentation", "").strip()
    name = extract_function_name(func.get("signature", "").strip())

    if not documentation or not name:
        return False
    if documentation.startswith(name):
        return True
    return any(marker in documentation for marker in EXAMPLE_MARKERS)


def _keep(func: FunctionRecord, seen_bodies: set[str], stats: FilterStats) -> bool:
    """Apply every retention rule to one function, updating ``stats``."""
    if func.get("type") != "basic":
        return False
    if "test" in func.get("signature", "").lower():
        return False

    func["body"] = remove_non_meaningful_comments(func.get("body", ""))
    normalized = normalize_function_body(func["body"])
    if normalized in seen_bodies:
        stats.duplicates += 1
        return False
    seen_bodies.add(normalized)

    signature = func.get("signature", "")
    if not signature.strip().startswith("function"):
        stats.non_standard_signature += 1

    if not is_type_annotated(signature):
        stats.not_annotated += 1
        return False

    return count_code_lines(func["body"]) < MAX_CODE_LINES


def filter_corpus(
    projects: list[ProjectRecord],
) -> tuple[list[ProjectRecord], list[ProjectRecord], list[ProjectRecord], FilterStats]:
    """Split ``projects`` into all-kept / good-doc / bad-doc corpora.

    Returns:
        ``(filtered, good_doc, bad_doc, stats)``, each corpus preserving the
        ``author``/``project`` grouping of the input.
    """
    stats = FilterStats(total=sum(len(p["functions"]) for p in projects))
    seen_bodies: set[str] = set()
    filtered: list[ProjectRecord] = []
    good: list[ProjectRecord] = []
    bad: list[ProjectRecord] = []

    for entry in projects:
        if entry["project"] in EXCLUDED_PROJECTS:
            continue

        kept: list[FunctionRecord] = []
        good_funcs: list[FunctionRecord] = []
        bad_funcs: list[FunctionRecord] = []

        for func in entry["functions"]:
            if not _keep(func, seen_bodies, stats):
                continue
            kept.append(func)
            if has_good_documentation(func):
                good_funcs.append(func)
                stats.good_doc += 1
            else:
                bad_funcs.append(func)
                stats.bad_doc += 1

        for bucket, funcs in ((filtered, kept), (good, good_funcs), (bad, bad_funcs)):
            if funcs:
                bucket.append(
                    ProjectRecord(author=entry["author"], project=entry["project"], functions=funcs)
                )

    stats.kept = sum(len(p["functions"]) for p in filtered)
    return filtered, good, bad, stats


def load_corpus(path: Path) -> list[ProjectRecord]:
    """Read a corpus JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_corpus(projects: list[ProjectRecord], path: Path) -> None:
    """Write a corpus JSON file, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(projects, indent=4, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %d projects to %s", len(projects), path)
