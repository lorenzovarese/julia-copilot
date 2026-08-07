"""Command-line entry point for the whole pipeline.

Each of the original scripts carried its own ``if __name__ == "__main__"``
block with its own flag spellings (``-f`` meant *force* in one and *file* in
another). They are unified here behind ``julia-copilot <stage>``, so the modules
themselves stay importable and side-effect free.

Heavy dependencies (torch, transformers, tree-sitter) are imported inside the
handlers, so ``julia-copilot --help`` stays instant and the parsing stages run
on a machine with no CUDA stack installed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from julia_copilot import __version__
from julia_copilot.config import (
    BENCHMARK_FILE,
    CHECKPOINTS_DIR,
    COMBINED_PROJECTS_JSON,
    DEFAULT_MODEL,
    ENCODED_DATA_ROOT,
    JULIA_CSV,
    REPOS_DIR,
    REPOS_ZIP,
    setup_logging,
)

# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #


def _cmd_clone(args: argparse.Namespace) -> int:
    from julia_copilot.clone import zipped_repos

    zipped_repos(
        zip_path=args.zip,
        csv_with_repos=args.csv,
        keep_repos_dir=args.keep,
        force=args.force,
        repos_dir=args.repos_dir,
    )
    return 0


def _cmd_parse(args: argparse.Namespace) -> int:
    from julia_copilot.filtering import save_corpus

    if args.backend == "regex":
        from julia_copilot.parsing.regex_parser import format_metrics, process_repos

        root = Path(args.source)
        if not root.is_dir():
            print(f"{root} is not a directory.", file=sys.stderr)
            return 1

        authors = sorted(p for p in root.iterdir() if p.is_dir())
        print(f"Selected {len(authors)} authors for processing.")
        projects = process_repos(authors)
        save_corpus(projects, args.output)
        print(format_metrics(projects))
        return 0

    import zipfile

    from julia_copilot.parsing.treesitter_parser import process_zipped_repos

    source = Path(args.source)
    archive = zipfile.ZipFile(source, "r") if source.is_file() else None
    if archive is None:
        print(f"The tree-sitter backend expects a zip archive, got {source}.", file=sys.stderr)
        return 1

    projects = process_zipped_repos(archive, max_authors=args.max_authors)
    save_corpus(projects, args.output)
    return 0


def _cmd_filter(args: argparse.Namespace) -> int:
    from julia_copilot.filtering import filter_corpus, load_corpus, save_corpus

    projects = load_corpus(args.input)
    filtered, good, bad, stats = filter_corpus(projects)

    save_corpus(filtered, args.output)
    save_corpus(good, args.good_doc_output)
    save_corpus(bad, args.bad_doc_output)
    print(stats.render())
    return 0


def _cmd_encode(args: argparse.Namespace) -> int:
    from julia_copilot.encode import encode_data

    dataset = encode_data(
        encoded_data_root=args.encoded_data_root,
        data_path=args.data,
        model_name=args.model,
        simple_input=args.simple_input,
        first_line_of_doc=args.first_line,
        frac_of_data=args.frac,
        force=args.force,
        benchmark_file=args.benchmark,
    )
    print(f"Encoded {len(dataset):,} instances.")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from julia_copilot.config import default_cuda_device
    from julia_copilot.encode import encode_data
    from julia_copilot.train import checkpoint_dir, trainer_for_model, trainer_for_quantized_model

    default_cuda_device()

    dataset = encode_data(
        encoded_data_root=args.encoded_data_root,
        data_path=args.data,
        model_name=args.model,
        simple_input=args.simple_input,
        first_line_of_doc=args.first_line,
        frac_of_data=args.frac_of_data,
        benchmark_file=args.benchmark,
    )
    print(f"Training on {len(dataset):,} instances.")

    output_dir = checkpoint_dir(
        args.model,
        args.frac_of_data,
        simple_input=args.simple_input,
        first_line=args.first_line,
        root=args.checkpoints,
    )
    print(f"Saving checkpoints to '{output_dir}'")

    build = trainer_for_quantized_model if args.quantized else trainer_for_model
    trainer = build(args.model, dataset, batch_size=args.batch_size, output_dir=output_dir)
    trainer.train()
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    from julia_copilot.query import generate_completions, load_model_and_tokenizer

    model, tokenizer, device = load_model_and_tokenizer(args.checkpoint, args.tokenizer)
    print("Generating completions...")
    completions = generate_completions(model, tokenizer, args.query, device, top_n=args.top_n)
    for i, completion in enumerate(completions, start=1):
        print(f"{i}. {completion}")
    return 0


def _cmd_extract(args: argparse.Namespace) -> int:
    from julia_copilot.benchmark.extract import export_results

    frame = export_results(args.results_directory, args.output_filepath)
    if not frame.empty:
        print(f"pass@1: {frame['passed'].mean():.4f} over {len(frame)} problems")
    return 0


def _cmd_mcnemar(args: argparse.Namespace) -> int:
    from julia_copilot.stats import load_passed_results, mcnemar_test

    result = mcnemar_test(
        load_passed_results(args.file1),
        load_passed_results(args.file2),
        exact=not args.chi2,
    )
    print(result.render(alpha=args.alpha))
    return 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Assemble the top-level parser and every stage subparser."""
    parser = argparse.ArgumentParser(
        prog="julia-copilot",
        description="Fine-tune and benchmark small language models on Julia code.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    parser.add_argument("--logfile", type=Path, default=None, help="Also write logs to this file.")

    sub = parser.add_subparsers(dest="command", required=True, metavar="<stage>")

    # -- clone ------------------------------------------------------------- #
    clone = sub.add_parser("clone", help="Clone and clean Julia repositories.")
    clone.add_argument("-c", "--csv", type=Path, default=JULIA_CSV, help="CSV of repository names.")
    clone.add_argument("-z", "--zip", type=Path, default=REPOS_ZIP, help="Output archive.")
    clone.add_argument("--repos-dir", type=Path, default=REPOS_DIR, help="Working checkout dir.")
    clone.add_argument("-k", "--keep", action="store_true", help="Keep checkouts after zipping.")
    clone.add_argument("-f", "--force", action="store_true", help="Rebuild an existing archive.")
    clone.set_defaults(func=_cmd_clone)

    # -- parse ------------------------------------------------------------- #
    parse = sub.add_parser("parse", help="Extract Julia functions into a JSON corpus.")
    parse.add_argument(
        "source",
        help="Directory of author/project checkouts (regex backend) or a repos.zip (tree-sitter).",
    )
    parse.add_argument(
        "-b",
        "--backend",
        choices=("regex", "tree-sitter"),
        default="regex",
        help="Extraction backend. Default: regex.",
    )
    parse.add_argument(
        "-o", "--output", type=Path, default=COMBINED_PROJECTS_JSON, help="Output corpus JSON."
    )
    parse.add_argument(
        "--max-authors",
        type=int,
        default=None,
        help="Tree-sitter backend only: stop after N authors. Default: all.",
    )
    parse.set_defaults(func=_cmd_parse)

    # -- filter ------------------------------------------------------------ #
    filt = sub.add_parser("filter", help="Filter the corpus and split it by doc quality.")
    filt.add_argument(
        "-i", "--input", type=Path, default=COMBINED_PROJECTS_JSON, help="Corpus to filter."
    )
    filt.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/filtered_functions.json"),
        help="All retained functions.",
    )
    filt.add_argument(
        "--good-doc-output",
        type=Path,
        default=Path("data/good_doc_functions.json"),
        help="Functions with good documentation.",
    )
    filt.add_argument(
        "--bad-doc-output",
        type=Path,
        default=Path("data/bad_doc_functions.json"),
        help="Functions with poor documentation.",
    )
    filt.set_defaults(func=_cmd_filter)

    # -- encode ------------------------------------------------------------ #
    encode = sub.add_parser("encode", help="Tokenise the corpus into a HuggingFace dataset.")
    _add_encoding_arguments(encode)
    encode.add_argument("-f", "--force", action="store_true", help="Re-encode, ignoring the cache.")
    encode.add_argument("--frac", type=float, default=1.0, help="Fraction of data to use.")
    encode.set_defaults(func=_cmd_encode)

    # -- train ------------------------------------------------------------- #
    train = sub.add_parser("train", help="Fine-tune a model on the encoded dataset.")
    _add_encoding_arguments(train)
    train.add_argument("-q", "--quantized", action="store_true", help="4-bit NF4 + LoRA.")
    train.add_argument("-b", "--batch-size", type=int, default=2, help="Per-device batch size.")
    train.add_argument("--frac-of-data", type=float, default=1.0, help="Fraction of data to use.")
    train.add_argument(
        "--checkpoints", type=Path, default=CHECKPOINTS_DIR, help="Checkpoint root directory."
    )
    train.set_defaults(func=_cmd_train)

    # -- query ------------------------------------------------------------- #
    query = sub.add_parser("query", help="Generate completions from a checkpoint.")
    query.add_argument("-c", "--checkpoint", required=True, help="Checkpoint directory.")
    query.add_argument("-q", "--query", required=True, help="Prompt (docstring + signature).")
    query.add_argument("--top-n", type=int, default=1, help="Number of completions.")
    query.add_argument("--tokenizer", default=None, help="Tokenizer source, if not the checkpoint.")
    query.set_defaults(func=_cmd_query)

    # -- extract ----------------------------------------------------------- #
    extract = sub.add_parser("extract", help="Flatten MultiPL-E results into one JSON file.")
    extract.add_argument(
        "-i", "--results-directory", required=True, help="Directory of MultiPL-E results."
    )
    extract.add_argument(
        "-o", "--output-filepath", default="results.json", help="Output JSON file."
    )
    extract.set_defaults(func=_cmd_extract)

    # -- mcnemar ----------------------------------------------------------- #
    mcnemar = sub.add_parser("mcnemar", help="Compare two benchmark runs (paired test).")
    mcnemar.add_argument("file1", help="Extracted results for model 1.")
    mcnemar.add_argument("file2", help="Extracted results for model 2.")
    mcnemar.add_argument("--alpha", type=float, default=0.05, help="Significance level.")
    mcnemar.add_argument(
        "--chi2", action="store_true", help="Use the chi-square approximation instead of exact."
    )
    mcnemar.set_defaults(func=_cmd_mcnemar)

    return parser


def _add_encoding_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags shared by the ``encode`` and ``train`` stages."""
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help="Base model.")
    parser.add_argument(
        "--data", type=Path, default=Path("data/combined_projects.zip"), help="Corpus to encode."
    )
    parser.add_argument(
        "--encoded-data-root", type=Path, default=ENCODED_DATA_ROOT, help="Encoded dataset cache."
    )
    parser.add_argument(
        "--benchmark", type=Path, default=BENCHMARK_FILE, help="HumanEval-jl prompts."
    )
    parser.add_argument(
        "--first-line", action="store_true", help="Keep only the first sentence of each docstring."
    )
    parser.add_argument(
        "--simple-input",
        action="store_true",
        help="Input is docstring + signature and the label is the body, "
        "instead of the whole function being both.",
    )


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and dispatch to the selected stage."""
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(verbose=args.verbose, logfile=args.logfile)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
