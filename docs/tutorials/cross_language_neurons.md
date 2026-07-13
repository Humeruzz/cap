# Cross-Language Neurons

CAP's headline question is *which neurons handle faithfulness differently across
languages?* You answer it by running the **same contrast** in several languages, then
comparing the resulting neuron signatures with the similarity viewer.

## 1. Capture each language into its own directory

Run the same faithfulness contrast per language, keeping each run separate. Supply one CSV
per language (CAP bundles only an English demo set — bring your own for the other
languages). Give them a shared text/label schema so only `--data` and `--output` change:

```bash
# Map each language to its CSV (same text column + binary label column in each).
declare -A DATA=(
  [english]=data/english.csv
  [egyptian_arabic]=data/egyptian_arabic.csv
  [mandarin_chinese]=data/mandarin_chinese.csv
)

for lang in "${!DATA[@]}"; do
  cap --device cuda capture \
    --model Qwen/Qwen3-8B --type faithfulness \
    --data "${DATA[$lang]}" --text-column srcs --label-column label \
    --output "runs/${lang}"
done
```

Each `runs/<lang>/` now holds a `statistics.h5` with that language's per-layer contrast.

!!! tip "Parallel content"
    For a strict comparison on the *same items*, use one parallel CSV that carries each
    row's text in every language (e.g. columns `srcs`, `srcs_arz`, `srcs_zh`) and point a
    different `--text-column` at each run.

## 2. Compare with the similarity viewer

Point `cap visualize --mode similarity` at two or more experiment directories:

```bash
cap visualize --mode similarity \
  --experiments runs/english runs/egyptian_arabic runs/mandarin_chinese \
  --output runs/similarity
```

This writes an HTML viewer to `runs/similarity/` showing the **per-layer cosine
similarity** between each pair of languages' statistics. High similarity at a layer means
the languages recruit the *same* neurons for faithfulness there; low similarity means the
model handles that language differently at that depth. Layers where similarity drops are
where "language-specific faithfulness" lives.

Each directory is labelled by its folder name in the viewer, so name your `--output`
directories descriptively.
