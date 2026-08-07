"""Flatten MultiPL-E result files into a single JSON table.

The harness writes one gzipped JSON per problem, deep inside a nested output
directory. This walks that tree and produces a flat, problem-ordered table with
one row per problem: ``problem``, ``prompt``, ``prediction``, ``passed``.

The original built the table with ``pd.concat`` inside the loop, reallocating
the whole frame on every file (quadratic in the number of problems). Rows are
now accumulated in a list and the frame is built once.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RESULTS_SUFFIX = ".results.json.gz"
PROBLEM_NUMBER_PATTERN = re.compile(r"(\d+)")


def read_gz_json(path: Path) -> dict:
    """Read one gzipped JSON result file."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _strip_scaffolding(program: str, prompt: str, tests: str) -> str:
    """Recover just the model's completion from the assembled program.

    The harness stores prompt + completion + tests concatenated; slicing off the
    known prefix and suffix leaves the generated body. ``tests`` may be empty,
    in which case the original's ``[:-len(tests)]`` would have discarded the
    entire completion.
    """
    body = program[len(prompt) :] if program.startswith(prompt) else program
    if tests and body.endswith(tests):
        body = body[: -len(tests)]
    return body


def extract_results(results_directory: Path | str) -> pd.DataFrame:
    """Collect every ``*.results.json.gz`` under ``results_directory``.

    Returns:
        A frame sorted by the numeric part of the problem name, with columns
        ``problem``, ``prompt``, ``prediction`` and ``passed``.
    """
    results_directory = Path(results_directory)
    rows: list[dict] = []

    for path in sorted(results_directory.rglob(f"*{RESULTS_SUFFIX}")):
        try:
            data = read_gz_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable result %s: %s", path, exc)
            continue

        results = data.get("results") or []
        if not results:
            logger.warning("No results recorded in %s, skipping.", path)
            continue

        result = results[0]
        rows.append(
            {
                "problem": data["name"],
                "prompt": data["prompt"],
                "prediction": _strip_scaffolding(
                    result["program"], data["prompt"], data.get("tests", "")
                ),
                "passed": result["status"] == "OK",
            }
        )

    frame = pd.DataFrame(rows, columns=["problem", "prompt", "prediction", "passed"])
    if frame.empty:
        logger.warning("No result files found under %s.", results_directory)
        return frame

    order = frame["problem"].str.extract(PROBLEM_NUMBER_PATTERN, expand=False).astype(int)
    return frame.assign(_order=order).sort_values("_order").drop(columns="_order")


def export_results(results_directory: Path | str, output_filepath: Path | str) -> pd.DataFrame:
    """Extract results and write them to ``output_filepath`` as JSON records."""
    frame = extract_results(results_directory)
    output_filepath = Path(output_filepath)
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    frame.to_json(output_filepath, orient="records")
    logger.info("Wrote %d results to %s", len(frame), output_filepath)
    return frame
