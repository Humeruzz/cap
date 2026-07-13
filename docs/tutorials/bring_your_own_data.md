# Bring Your Own Data

CAP works on any CSV. This walks from a raw file to an interpreted result.

## 1. Prepare a CSV

You need a **text column** and a **binary label column**. The label defines the two
contrastive groups: rows labelled `1` are group A, everything else is group B.

```csv
statement,is_true
The mitochondria is the powerhouse of the cell.,1
Goldfish have a three-second memory.,0
Honey never spoils.,1
Lightning never strikes the same place twice.,0
```

No fixed column names are required — you tell CAP which columns to use.

## 2. Capture and visualize

```bash
cap --device cpu run \
  --model gpt2 \
  --type factuality \
  --data mydata.csv --text-column statement --label-column is_true \
  --output runs/mydata
```

Prefer two separate files (say, one per condition) over a single labelled file? Use the
two-group form instead:

```bash
cap --device cpu run --model gpt2 --type factuality \
  --group-a true_statements.csv --group-b false_statements.csv \
  --text-column statement \
  --output runs/mydata
```

## 3. Interpret the HTML

Open the `*.html` file written into `runs/mydata/`. The stats viewer shows a **per-layer
heatmap**; each row is a layer, each column a neuron. The values come straight from
`statistics.h5`:

- **t-stat** — how strongly a neuron separates group A from group B (sign = direction).
- **corrected p-value** — significance after FDR correction; small = trustworthy.
- **Cohen's d** — effect size, independent of sample size.

Bright, high-`|d|`, low-`p` neurons clustered in particular layers are your candidates:
those are the units that respond differently to true vs. false statements. Note which
layers they live in — that is where the model appears to do this work.

## 4. Validate on your own metric (optional)

Correlation is not causation. To test whether those neurons *matter*, ablate them and
measure a metric **you** define. Implement `EvaluatorProtocol` — a single `evaluate`
method returning a `{metric: value}` dict:

```python
# my_eval.py
class MyEvaluator:
    def evaluate(self, model, tokenizer, *, n_samples: int, device: str) -> dict[str, float]:
        # run `model` on your held-out data and score it however you like
        score = ...  # float
        return {"my_metric": score}
```

Then point `cap patch` at it — no edits to CAP:

```bash
cap --device cpu patch \
  --experiment runs/mydata \
  --evaluator my_eval::MyEvaluator \
  --scale 0.0 --d-threshold 1.0
```

`--scale 0.0` ablates the selected neurons entirely; compare the metric with and without to
see how much they were contributing. For a ready-made CSV metric, see
[`CSVEvaluator`](../api/core.md).
