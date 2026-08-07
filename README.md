# Julia Copilot — Fine-Tuning and Benchmarking

This repository fine-tunes and evaluates small language models for Julia code completion. It covers the whole path from raw data to a significance test: scraping and cleaning GitHub repositories, extracting documented functions, encoding them into a training set, fine-tuning (optionally with 4-bit quantisation and LoRA), and benchmarking the result on HumanEval-jl through [MultiPL-E](https://github.com/nuprl/MultiPL-E).

## Example

Install, then fine-tune `SmolLM2-135M` on 35% of the corpus and ask it to complete a function:

```bash
# 1. Install (uv creates the virtualenv and resolves from uv.lock)
uv sync --extra train

# 2. Fine-tune: quantised, 35% of the data, batch size 4
uv run julia-copilot train \
    --model HuggingFaceTB/SmolLM2-135M \
    --quantized \
    --frac-of-data 0.35 \
    --batch-size 4

# 3. Complete a function from its docstring and signature
uv run julia-copilot query \
    --checkpoint checkpoints/HuggingFaceTB_SmolLM2-135M/035 \
    --query '"""
Add two numbers `x` and `y`.
"""
function add(x::Int, y::Int)'
```

which prints something along the lines of:

```julia
"""
Add two numbers `x` and `y`.
"""
function add(x::Int, y::Int)
    return x + y
end
```

Every stage is a subcommand of the same entry point — run `julia-copilot --help` to list them:

| Stage | What it does |
|---|---|
| `clone` | Clone GitHub repositories and strip everything that is not `.jl` |
| `parse` | Extract functions into a JSON corpus (`--backend regex` or `tree-sitter`) |
| `filter` | Deduplicate, drop untyped/oversized functions, split by documentation quality |
| `encode` | Tokenise the corpus into a cached HuggingFace dataset |
| `train` | Fine-tune a causal LM, optionally 4-bit + LoRA |
| `query` | Generate completions from a checkpoint |
| `extract` | Flatten MultiPL-E result files into one JSON table |
| `mcnemar` | Compare two benchmark runs with a paired significance test |

## Repository Structure

```
.
├── README.md
├── pyproject.toml            # Project metadata, dependencies, ruff config
├── uv.lock                   # Fully resolved, reproducible dependency set
├── benchmark/                # MultiPL-E harness and stored results
│   ├── evaluate.sh           # Run the Docker evaluator and export results
│   ├── generate.sh           # Generate completions for a model
│   ├── humaneval-jl-reworded.jsonl
│   └── results/
├── clustering-attempt/       # Clustering experiments (negative result)
│   └── report_clustering.md
├── data/
│   ├── combined_projects.zip # Extracted Julia projects and functions
│   └── julia.csv.gz          # Repository metadata
├── doc/
│   └── requirements.pdf
├── scripts/
│   └── train.slurm           # SLURM batch script for cluster training
├── src/julia_copilot/
│   ├── cli.py                # Single entry point for every stage
│   ├── config.py             # Shared paths, defaults, logging
│   ├── clone.py              # Repository cloning and cleaning
│   ├── parsing/
│   │   ├── regex_parser.py       # Tolerant, line-oriented extractor
│   │   └── treesitter_parser.py  # Grammar-driven extractor
│   ├── filtering.py          # Corpus filtering and doc-quality split
│   ├── encode.py             # Tokenisation into a HuggingFace dataset
│   ├── train.py              # Full and LoRA fine-tuning
│   ├── query.py              # Completion generation
│   ├── stats.py              # McNemar's test
│   └── benchmark/extract.py  # MultiPL-E result flattening
└── tests/
```

## Installation

The project uses [uv](https://docs.astral.sh/uv/). `uv.lock` pins the full dependency graph, so everyone resolves to the same versions.

```bash
# Parsing and dataset building only (no CUDA stack, ~100 MB)
uv sync

# Everything, including torch, peft, trl and bitsandbytes
uv sync --extra train
```

Dependencies are split deliberately: the scraping and corpus stages run fine on a laptop, and only `train`/`query` pull the multi-gigabyte GPU wheels.

## Running the Pipeline

Repositories are selected by having Julia as their primary language, then cloned and stripped of non-Julia files. The cleaned dataset is roughly 350 MB, too large to commit, so download it rather than re-cloning:

```bash
scp <YOUR_USER>@gym.si.usi.ch:/home/SA24-G3/project2/data/repos.zip data/
```

Then, from the repository root:

```bash
uv run julia-copilot clone                       # only if you skipped the download
uv run julia-copilot parse data/repos.zip --backend tree-sitter
uv run julia-copilot filter
uv run julia-copilot encode --model HuggingFaceTB/SmolLM2-135M
uv run julia-copilot train  --model HuggingFaceTB/SmolLM2-135M --quantized
```

### Benchmarking

`generate.sh` and `evaluate.sh` wrap MultiPL-E; both clone it and pin the same commit on first run. `evaluate.sh` additionally needs Docker.

```bash
cd benchmark
./generate.sh HuggingFaceTB/SmolLM2-135M
./evaluate.sh HuggingFaceTB/SmolLM2-135M
```

To compare two runs, extract each and apply McNemar's test. The models are evaluated on the same problems, so the outcomes are paired and a paired test is the correct one:

```bash
uv run julia-copilot extract -i benchmark/results/baseline -o baseline.json
uv run julia-copilot extract -i benchmark/results/finetuned -o finetuned.json
uv run julia-copilot mcnemar baseline.json finetuned.json
```

### Training on a SLURM cluster

`scripts/train.slurm` wraps the `train` stage as a batch job. Partition names, GPU
flags and module names are site-specific and marked `SITE-SPECIFIC` in the file.

```bash
sbatch scripts/train.slurm
sbatch --export=ALL,MODEL=bigcode/starcoderbase-1b,FRAC=0.35,BATCH_SIZE=8 scripts/train.slurm
sbatch --partition=gpu-long --time=48:00:00 --gres=gpu:a100:1 scripts/train.slurm
```

Parameters are read from the environment: `MODEL`, `DATA`, `FRAC`, `BATCH_SIZE`,
`QUANTIZED`, `FIRST_LINE`, `SIMPLE_INPUT`.

Two things the script handles that are easy to get wrong:

- **Caches go to scratch.** `HF_HOME`, `UV_CACHE_DIR` and `TRITON_CACHE_DIR` are
  redirected off `$HOME`, which is usually quota'd and NFS-backed.

Run `uv sync --locked --extra train` on the login node before the first submission:
compute nodes are often network-isolated and cannot reach PyPI or the HuggingFace Hub.

## Training Parameters

| Parameter | Default | Description |
|---|---|---|
| `--model` | `HuggingFaceTB/SmolLM2-135M` | Pre-trained model to fine-tune |
| `--quantized` | `False` | 4-bit NF4 quantisation with LoRA adapters |
| `--first-line` | `False` | Keep only the first sentence of each docstring |
| `--frac-of-data` | `1.0` | Fraction of the dataset to train on |
| `--batch-size` | `2` | Per-device batch size; raise it on a larger GPU |
| `--simple-input` | `False` | Input is docstring + signature, label is the body, instead of the whole function being both |
| `--encoded-data-root` | `data/encoded_data` | Where the tokenised dataset is cached |
| `--checkpoints` | `checkpoints` | Checkpoint root directory |

Encoded datasets and checkpoints are keyed by model, data fraction and flags (for example `data/encoded_data/HuggingFaceTB_SmolLM-135M/035_first_line`), so runs with different settings never overwrite one another.

## Development

```bash
uv sync                  # dev tools come with the default dependency group
uv run ruff check .      # lint
uv run ruff format .     # format
uv run pytest            # tests
```

Ruff handles both linting and formatting; its configuration lives in `pyproject.toml`. CI runs all three on every push and pull request.

## Contributors

- Arnaud Fauconnet
- Lorenzo Varese
- Vittorio Perozzi
