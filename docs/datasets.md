# Datasets

CAP ships **three small English CSVs** inside the package (`cap.data`, under `files/`) so
every example and the quickstart run without a download — they install with the package.
They are demonstration data; for real analysis you bring your own (see below).

## Bundled files

| File | Rows | Text column | Label column | Purpose |
|---|---|---|---|---|
| `quickstart_synthetic.csv` | 16 | `text` | `label` | Hand-authored true/false facts for the 30-second quickstart. |
| `english_smol.csv` | 303 | `srcs` | `label` | English faithfulness subset. |
| `english_smol_with_gen.csv` | 519 | `srcs` | `label` | Same, plus rows produced by the text-generation stage. |

In every file the binary `label` column is what CAP splits on (`1` = true/faithful,
`0` = false/unfaithful).

- **`quickstart_synthetic.csv`** — schema `text, label`. The minimal dataset used by the
  quickstart and the example scripts; runs end-to-end on `gpt2`/CPU in seconds.
- **`english_smol*.csv`** — schema `id, srcs, 1, 2, 3, avg, sum, label`. `srcs` is the
  statement; `1`/`2`/`3` are human annotation scores aggregated into `avg`/`sum` and
  thresholded into the binary `label`. The `_with_gen` variant additionally contains rows
  produced by the text-generation stage.

## Loading a bundled CSV

Resolve a packaged file to a real path with `importlib.resources`:

```python
from importlib.resources import files
from cap.data.loaders import LabeledDataset

csv = files("cap.data") / "files" / "english_smol_with_gen.csv"

dataset = LabeledDataset.from_csv(
    csv, text_col="srcs", label_col="label", n_samples=200, seed=42
)
group_a, group_b = dataset.as_groups()
```

For the CLI, pass the resolved path as a string:

```bash
cap --device cpu run --model gpt2 --type factuality \
  --data "$(python -c 'from importlib.resources import files; print(files("cap.data")/"files"/"english_smol_with_gen.csv")')" \
  --text-column srcs --label-column label \
  --output runs/en
```

## Bring your own data

Any CSV works — nothing about the bundled files is special. You need either:

- **one CSV** with a text column and a **binary label column** (`--data … --text-column …
  --label-column …`), or
- **two CSVs** that share a text column, one per group (`--group-a … --group-b …
  --text-column …`).

For a step-by-step walkthrough see
[Bring Your Own Data](tutorials/bring_your_own_data.md). For the loader API see the
[Data API](api/data.md).
