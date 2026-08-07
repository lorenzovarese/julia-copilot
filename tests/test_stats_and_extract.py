"""Tests for the paired significance test and the benchmark result extractor."""

from __future__ import annotations

import gzip
import json

import pytest

from julia_copilot.benchmark.extract import _strip_scaffolding, export_results, extract_results
from julia_copilot.stats import load_passed_results, mcnemar_test

# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #


def test_mcnemar_builds_the_contingency_table():
    first = [True, True, False, False, True]
    second = [True, False, True, False, False]
    result = mcnemar_test(first, second)

    assert result.both_passed == 1
    assert result.only_first == 2
    assert result.only_second == 1
    assert result.both_failed == 1
    assert result.n_discordant == 3
    assert 0.0 <= result.p_value <= 1.0


def test_mcnemar_identical_models_are_not_significant():
    outcomes = [True, False, True, True, False]
    result = mcnemar_test(outcomes, list(outcomes))

    assert result.n_discordant == 0
    assert result.p_value == pytest.approx(1.0)


def test_mcnemar_rejects_misaligned_runs():
    with pytest.raises(ValueError, match="same problems"):
        mcnemar_test([True, False], [True])


def test_load_passed_results(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(json.dumps([{"passed": True}, {"passed": False}]), encoding="utf-8")

    assert load_passed_results(path) == [True, False]


# --------------------------------------------------------------------------- #
# benchmark extraction
# --------------------------------------------------------------------------- #


def test_strip_scaffolding_removes_prompt_and_tests():
    assert _strip_scaffolding("PROMPTbodyTESTS", "PROMPT", "TESTS") == "body"


def test_strip_scaffolding_survives_empty_tests():
    """The original sliced with ``[:-len(tests)]``, which erased the body here."""
    assert _strip_scaffolding("PROMPTbody", "PROMPT", "") == "body"


def _write_result(directory, name, status, number):
    payload = {
        "name": f"HumanEval_{number}_{name}",
        "prompt": "PROMPT",
        "tests": "TESTS",
        "results": [{"program": "PROMPTbodyTESTS", "status": status}],
    }
    path = directory / f"HumanEval_{number}_{name}.results.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_extract_results_sorts_numerically(tmp_path):
    _write_result(tmp_path, "b", "OK", 10)
    _write_result(tmp_path, "a", "Exception", 2)

    frame = extract_results(tmp_path)

    assert list(frame["problem"]) == ["HumanEval_2_a", "HumanEval_10_b"]
    assert list(frame["passed"]) == [False, True]
    assert list(frame["prediction"]) == ["body", "body"]


def test_extract_results_on_empty_directory(tmp_path):
    frame = extract_results(tmp_path)

    assert frame.empty
    assert list(frame.columns) == ["problem", "prompt", "prediction", "passed"]


def test_export_results_writes_json(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_result(results_dir, "a", "OK", 1)

    output = tmp_path / "out" / "results.json"
    export_results(results_dir, output)

    assert json.loads(output.read_text(encoding="utf-8"))[0]["problem"] == "HumanEval_1_a"
