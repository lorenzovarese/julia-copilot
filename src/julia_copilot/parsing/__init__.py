"""Two interchangeable Julia function extractors.

Both backends emit the same record schema so the downstream stages
(:mod:`julia_copilot.filtering`, :mod:`julia_copilot.encode`) do not care which
one produced the corpus:

* :mod:`~julia_copilot.parsing.regex_parser` -- line-oriented, dependency-free,
  tolerant of files that do not parse. This is the backend used for the corpus
  that ships with the repository.
* :mod:`~julia_copilot.parsing.treesitter_parser` -- grammar-driven, more
  accurate on nested modules, but skips any file with a syntax error.
"""

from __future__ import annotations

from typing import Literal, TypedDict

FunctionType = Literal["basic", "single-line", "anonymous"]


class FunctionRecord(TypedDict, total=False):
    """A single extracted Julia function."""

    documentation: str
    signature: str
    body: str
    inline_comments: list[str]
    file_path: str
    line_number: int
    type: FunctionType
    module: str


class ProjectRecord(TypedDict):
    """All functions extracted from one ``author/project`` pair."""

    author: str
    project: str
    functions: list[FunctionRecord]


__all__ = ["FunctionRecord", "FunctionType", "ProjectRecord"]
