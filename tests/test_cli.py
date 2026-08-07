"""Tests for CLI wiring.

These only exercise argument parsing: the handlers import torch/transformers
lazily, so the parser stays testable without the ``train`` extra installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from julia_copilot.cli import build_parser


@pytest.fixture
def parser():
    return build_parser()


def test_requires_a_subcommand(parser):
    with pytest.raises(SystemExit):
        parser.parse_args([])


@pytest.mark.parametrize(
    "command",
    ["clone", "parse", "filter", "encode", "train", "query", "extract", "mcnemar"],
)
def test_every_stage_is_registered(parser, command):
    assert command in parser.format_help()


def test_clone_defaults(parser):
    args = parser.parse_args(["clone"])

    assert args.csv == Path("data/julia.csv.gz")
    assert args.zip == Path("data/repos.zip")
    assert args.force is False


def test_parse_backend_choice(parser):
    args = parser.parse_args(["parse", "data/repos", "--backend", "tree-sitter"])

    assert args.backend == "tree-sitter"
    assert args.source == "data/repos"


def test_parse_rejects_unknown_backend(parser):
    with pytest.raises(SystemExit):
        parser.parse_args(["parse", "data/repos", "--backend", "ast"])


def test_train_flags(parser):
    args = parser.parse_args(
        ["train", "--quantized", "--frac-of-data", "0.35", "--batch-size", "4"]
    )

    assert args.quantized is True
    assert args.frac_of_data == 0.35
    assert args.batch_size == 4
    assert args.model == "HuggingFaceTB/SmolLM2-135M"


def test_query_requires_checkpoint_and_query(parser):
    with pytest.raises(SystemExit):
        parser.parse_args(["query", "-c", "checkpoints/run"])


def test_mcnemar_takes_two_files(parser):
    args = parser.parse_args(["mcnemar", "a.json", "b.json", "--alpha", "0.01"])

    assert (args.file1, args.file2) == ("a.json", "b.json")
    assert args.alpha == 0.01
    assert args.chi2 is False


def test_verbose_is_global(parser):
    assert parser.parse_args(["-v", "clone"]).verbose is True


def test_worker_count_respects_slurm_allocation(monkeypatch):
    """cpu_count() reports the node, not the allocation, and would oversubscribe."""
    import importlib

    from julia_copilot import config

    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    monkeypatch.delenv("JULIA_COPILOT_NUM_WORKERS", raising=False)
    assert importlib.reload(config).NUM_WORKERS == 8

    monkeypatch.setenv("JULIA_COPILOT_NUM_WORKERS", "3")
    assert importlib.reload(config).NUM_WORKERS == 3
