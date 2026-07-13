# Installation

## Requirements

- **Python ≥ 3.10.** CAP uses `list[int]` / `X | None` builtin generics and other
  modern typing features throughout the source, and the pinned dependency stack
  (numpy 2.x, transformers, torch 2.x) is only tested on 3.10+. Older interpreters
  are not supported.
- **PyTorch 2.x.** A CUDA GPU is recommended for real models; `gpt2` and the smoke
  tests run fine on CPU. Approximate bf16 VRAM: Qwen3-8B ≈ 16 GB, Qwen3-14B ≈ 28 GB.
- **Linux** is the tested platform.

## Install from source

CAP is a research project and is installed from a checkout rather than PyPI:

```bash
git clone <your-fork-or-clone-url> cap
cd cap
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

This installs the `cap` command and the `cap` Python package. Verify:

```bash
cap --help
```

## Optional extras

Benchmark evaluation (GSM8K / MMLU / HellaSwag via `lm-eval`) is an optional extra so
the core install stays light:

```bash
pip install -e ".[lm-eval]"
```

Development tooling (pytest, ruff, mypy, pre-commit) and the documentation toolchain:

```bash
pip install -e ".[dev]"     # tests + linters
pip install -e ".[docs]"    # mkdocs + mkdocs-material + mkdocstrings
```

## A note on `--trust-remote-code`

Some HuggingFace repositories ship **custom Python** that is executed on your machine
when the model loads. CAP keeps this **off by default**. Only pass
`--trust-remote-code` when you trust the source of the model:

```bash
cap --trust-remote-code capture --model some-org/custom-model ...
```

Like the other global flags, `--trust-remote-code` goes **before** the subcommand.
