# CAP — Contrastive Activation Patching

**Which neurons handle faithfulness differently across languages?**

CAP is a research pipeline for locating and patching the neurons an LLM uses to judge
factuality and faithfulness. Give it two contrastive groups of text — *faithful* vs.
*unfaithful*, or the same prompt in English vs. Mandarin — and it captures per-layer
activations, runs a **Welch t-test with FDR correction** to find the neurons that separate
the groups, **patches** (scales/ablates) them to test whether they matter on a benchmark,
and renders interactive HTML viewers for inspection and cross-language comparison.

> Tested on Qwen3-8B / Qwen3-14B across English, Egyptian Arabic, Mandarin Chinese,
> German, and Spanish (WMT MIST). The SLURM templates under `examples/slurm/` are
> cluster-specific references — set `--partition` / `--gres` / `--mail-user` before use.

## Quickstart

```bash
cap --device cpu run \
  --model gpt2 --type faithfulness \
  --data "$(python -c 'from importlib.resources import files; print(files("cap.data")/"files"/"english_smol.csv")')" \
  --text-column srcs --label-column label \
  --n-samples 100 --downsample 32,64 \
  --output runs/en
```

That runs the bundled English faithfulness set on `gpt2`/CPU in under a minute: it captures
activations for both label groups, runs the Welch t-test + FDR contrast (finding significant
neurons across many layers), and writes an interactive stats viewer into `runs/en/` (with
32- and 64-resolution heatmaps you can toggle in the browser). Full walkthrough:
[docs/quickstart.md](docs/quickstart.md).

Global flags (`--device`, `--seed`, `--trust-remote-code`) go **before** the subcommand.

## Installation

Requires **Python ≥ 3.10** and PyTorch 2.x. Installed from a checkout (not PyPI):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .                 # core
pip install -e ".[lm-eval]"      # + GSM8K / MMLU / HellaSwag benchmarks
pip install -e ".[docs]"         # + documentation toolchain
```

See [docs/installation.md](docs/installation.md).

## Bundled data

Three small English faithfulness / fact-checking CSVs ship inside the package
(`cap.data`) — no download needed. See [docs/datasets.md](docs/datasets.md).

## Documentation

Full docs live under [`docs/`](docs/) and build with MkDocs Material:

```bash
pip install -e ".[docs]"
mkdocs serve      # http://127.0.0.1:8000
```

- [Concepts](docs/concepts.md) — the pipeline and the statistics
- [CLI Reference](docs/cli_reference.md) — every subcommand and flag
- [Tutorials](docs/tutorials/bring_your_own_data.md) — bring your own data, new models, cross-language neurons, SLURM

## Citation

If you use CAP, please cite the repository:

```bibtex
@software{cap,
  title  = {CAP: Contrastive Activation Patching},
  author = {Semenikhin, Aleksandr},
  year   = {2026},
  url    = {https://github.com/Humeruzz/cap}
}
```

## License

MIT — see [LICENSE](LICENSE). Release notes in [CHANGELOG.md](CHANGELOG.md).
