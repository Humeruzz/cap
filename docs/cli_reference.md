# CLI Reference

The `cap` command has five subcommands: `capture`, `stats`, `patch`, `visualize`, and
`run`. Run `cap <command> --help` for the live signature.

## Global flags

These attach to the root command and must appear **before** the subcommand:

| Flag | Default | Description |
|---|---|---|
| `--seed INTEGER` | `42` | Random seed for reproducibility. |
| `--device TEXT` | `auto` | `cpu`, `cuda`, `mps`, or `auto`. |
| `--trust-remote-code` | off | Allow a model repo to execute custom code on download. Only for trusted sources. |

```bash
cap --device cpu --seed 0 capture --model gpt2 ...
```

## `cap capture`

Capture activations from two contrastive groups and write `activations.h5`,
`statistics.h5`, and `manifest.json`.

| Flag | Default | Description |
|---|---|---|
| `--model TEXT` | *required* | HuggingFace model name or local path. |
| `--output PATH` | *required* | Output directory for the run artifacts. |
| `--data PATH` | — | CSV with a text column and a binary label column. |
| `--text-column TEXT` | — | Column containing the text to feed the model. |
| `--label-column TEXT` | — | Binary label column (used with `--data`). |
| `--group-a PATH` | — | CSV for group A (used with `--group-b`). |
| `--group-b PATH` | — | CSV for group B (used with `--group-a`). |
| `--type TEXT` | — | Built-in prompt template (see [Prompt types](#prompt-types)). |
| `--prompt TEXT` | — | Custom prompt template string containing `{text}`. |
| `--n-samples INTEGER` | `0` | Max rows per group (`0` = all). |

Provide **either** `--data` (with `--text-column` + `--label-column`) **or**
`--group-a`/`--group-b` (with `--text-column`), and exactly one of `--type`/`--prompt`.

## `cap stats`

Recompute Welch t-test + FDR statistics on an existing capture (no model load).

| Flag | Default | Description |
|---|---|---|
| `--experiment PATH` | *required* | Experiment directory written by `cap capture`. |

## `cap patch`

Select significant neurons, scale/ablate them, and evaluate on a benchmark.

| Flag | Default | Description |
|---|---|---|
| `--experiment PATH` | *required* | Experiment directory from `cap capture`. |
| `--evaluator TEXT` | — | Custom evaluator as `module::ClassName`. |
| `--benchmark TEXT` | — | Built-in benchmark: `gsm8k`, `mmlu`, `hellaswag`. |
| `--d-threshold FLOAT` | `1.0` | Minimum Cohen's d for a neuron to be selected. |
| `--std-threshold FLOAT` | `0.1` | Minimum pooled std for a neuron to be selected. |
| `--scale FLOAT` | `0.0` | Multiplicative factor on selected neurons: `0.0` ablates, `1.0` leaves them unchanged, `>1` amplifies. Use `0.0` for a causal ablation test, or a value near `1` (e.g. `1.05`) to **steer** the behaviour — see [Concepts → Steering](concepts.md#steering). |
| `--n-eval-samples INTEGER` | `100` | Benchmark examples to evaluate on. |

Provide exactly one of `--benchmark` or `--evaluator`. The built-in benchmarks require the
`lm-eval` extra (`pip install -e ".[lm-eval]"`).

## `cap visualize`

Render the interactive stats viewer, or the cross-language similarity viewer.

| Flag | Default | Description |
|---|---|---|
| `--experiment PATH` | — | Single experiment dir (for `--mode stats`). |
| `--experiments PATH` | — | Experiment dirs for `--mode similarity`. Comma-separate them (`--experiments A,B`) or repeat the flag (`--experiments A --experiments B`); the two forms can be mixed. Space-separated values do **not** work — the shell splits them before the CLI sees them. A path containing a comma must use the repeated form. |
| `--mode TEXT` | `stats` | `stats` or `similarity`. |
| `--output PATH` | — | Output dir for the similarity HTML (`--mode similarity` only). |
| `--downsample TEXT` | `32,64,128` | Comma-separated heatmap resolutions for the stats HTML (e.g. `32,64,128`). |

## `cap run`

Run the full **capture → visualize** pipeline in one call. (The patch step stays separate —
it needs a benchmark and evaluator.) Accepts the same data/prompt flags as `cap capture`,
plus:

| Flag | Default | Description |
|---|---|---|
| `--model TEXT` | *required* | HuggingFace model name or local path. |
| `--output PATH` | *required* | Output directory for the run artifacts. |
| `--skip-viz` | off | Capture only; skip the HTML stats viewer. |
| `--downsample TEXT` | `32,64,128` | Comma-separated heatmap resolutions for the stats HTML (e.g. `32,64,128`). |

## Prompt types

`--type` selects a built-in template; each wraps the row's text in `{text}`.

| Type | Template |
|---|---|
| `factuality` | `Is the following statement true or false?\n{text}\nAnswer:` |
| `faithfulness` | `Is the following text faithful to the source material?\n{text}\nAnswer yes or no:` |
| `negation` | `The following statement is {label_name}.\n{text}\nComplete the sentence: The statement is ` |
| `raw` | `{text}` |

For anything else, pass a custom template with `--prompt "Q: {text}\nA:"` — it must contain
the `{text}` placeholder.
