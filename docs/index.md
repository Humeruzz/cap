# CAP — Contrastive Activation Patching

**Which neurons in your LLM handle faithfulness differently across languages?**

CAP is a research pipeline for **locating and patching** the neurons a language model
uses to judge factuality and faithfulness. You give it two contrastive groups of
text — for example *faithful* vs. *unfaithful* statements, or the *same* prompt in
English vs. Mandarin — and CAP:

1. **captures** per-layer activations for both groups,
2. runs a **Welch t-test with FDR correction** to find neurons that separate the groups,
3. **patches** (scales or ablates) those neurons and re-evaluates on a benchmark, and
4. renders **interactive HTML viewers** for inspection and cross-language comparison.

The whole thing runs from one CLI (`cap`) or from the Python API.

## One command

```bash
cap --device cpu run \
  --model gpt2 --type faithfulness \
  --data "$(python -c 'from importlib.resources import files; print(files("cap.data")/"files"/"english_smol.csv")')" \
  --text-column srcs --label-column label \
  --n-samples 100 --downsample 32,64 \
  --output runs/en
```

That runs the bundled English faithfulness set on `gpt2`/CPU in under a minute: it captures
activations for both label groups, writes the statistics, and generates the interactive
stats viewer under `runs/en/`.

!!! tip "Global flags come first"
    `--device`, `--seed`, and `--trust-remote-code` attach to the root command, so
    they go **before** the subcommand: `cap --device cpu run …`.

## Where to go next

- **[Installation](installation.md)** — install CAP and the optional benchmark extra.
- **[Quickstart](quickstart.md)** — a full `gpt2` run on CPU in under a minute, CLI and Python.
- **[Concepts](concepts.md)** — the pipeline, contrastive groups, and the statistics.
- **[Datasets](datasets.md)** — the 3 bundled English CSVs and how to bring your own.
- **[CLI Reference](cli_reference.md)** — every subcommand and flag.
