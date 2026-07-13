# Quickstart

This runs a complete CAP pipeline on `gpt2`, on **CPU**, in under a minute, using a dataset
that ships with the package — nothing to download except the `gpt2` weights.

## 1. The dataset

CAP's single-file input is any CSV with a **text column** and a **binary label column**.
Rows with label `1` become group A; the rest become group B. CAP bundles a ready-to-run
example — `english_smol.csv`, an English faithfulness set (text column `srcs`, binary
`label`) — so you can run the pipeline immediately. Resolve its packaged path with
`importlib.resources`:

```bash
DATA="$(python -c 'from importlib.resources import files; print(files("cap.data")/"files"/"english_smol.csv")')"
```

To use your own data instead, point `--data` at any CSV and name its columns with
`--text-column` / `--label-column`. See [Datasets](datasets.md).

## 2. Run the pipeline (CLI)

```bash
cap --device cpu run \
  --model gpt2 \
  --type faithfulness \
  --data "$DATA" --text-column srcs --label-column label \
  --n-samples 100 --downsample 32,64 \
  --output runs/quickstart
```

`--type faithfulness` wraps each row in the built-in faithfulness prompt
(`Is the following text faithful to the source material? … Answer yes or no:`). See
[all prompt types](cli_reference.md#prompt-types). `--n-samples 100` caps each label group so
the run stays under a minute on CPU, and `--downsample 32,64` writes two heatmap resolutions
you can toggle in the viewer.

When it finishes, `runs/quickstart/` contains:

| File | What it is |
|---|---|
| `activations.h5` | Per-layer activations for both groups |
| `statistics.h5` | Welch t-test + FDR results per layer (t-stats, corrected p, Cohen's d) |
| `manifest.json` | Model, data spec, prompt template, seed — everything to reproduce the run |
| `interactive_stats.html` | Interactive stats viewer (open it in a browser) |

The run reports how many neurons separate the two groups — on `gpt2` with this data you will
see significant neurons across many layers.

The `patch` step is not part of `cap run` — patching needs a benchmark and an evaluator, so
run it separately once you have activations (built-in benchmarks need the `[lm-eval]` extra):

```bash
cap --device cpu patch --experiment runs/quickstart --benchmark hellaswag --n-eval-samples 20
```

## 3. The same thing in Python

The CLI is a thin wrapper over the public API. Here is the capture + statistics half:

```python
from importlib.resources import files

from cap.experiments.capture import CaptureExperiment
from cap.data.loaders import LabeledDataset
from cap.data.prompt_templates import FAITHFULNESS

# Load the bundled CSV and split into two contrastive groups by label.
csv = files("cap.data") / "files" / "english_smol.csv"
dataset = LabeledDataset.from_csv(
    csv, text_col="srcs", label_col="label", n_samples=100, seed=42
)
group_a, group_b = dataset.as_groups()

# Wrap each text in the faithfulness prompt template.
prompts_a = [FAITHFULNESS.apply(t) for t in group_a]
prompts_b = [FAITHFULNESS.apply(t) for t in group_b]

# Capture activations and compute contrastive statistics.
exp = CaptureExperiment(model_path="gpt2", device="cpu", seed=42)
exp.run_contrast(
    prompts1=prompts_a,
    prompts2=prompts_b,
    label1="faithful",
    label2="unfaithful",
    output_path="runs/quickstart_py",
)
```

`runs/quickstart_py/` now holds the same `activations.h5` / `statistics.h5` as the CLI run.
Load the statistics back with [`H5Store`](api/core.md):

```python
from cap.utils.h5_utils import H5Store

stats = H5Store.load_statistics("runs/quickstart_py/statistics.h5")
for layer, s in stats.items():
    print(layer, "significant neurons:", int(s["significant"].sum()))
```

## Next

- Swap `gpt2` for any HuggingFace model → [New Model](tutorials/new_model.md).
- Explore the bundled datasets and bring your own → [Datasets](datasets.md).
- Compare neurons across languages → [Cross-Language Neurons](tutorials/cross_language_neurons.md).
