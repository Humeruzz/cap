"""Compare neuron signatures across several runs — the Python form of
`cap visualize --mode similarity`.

CAP's headline question is *which neurons handle faithfulness differently across
languages?* You answer it by running the **same contrast** per language into its own
directory, then comparing the resulting statistics with the similarity viewer, which
plots the per-layer cosine similarity between each pair of runs.

A real cross-language study swaps in a multilingual model (e.g. Qwen/Qwen3-8B) and feeds
the same contrast in each language. To stay runnable on CPU with no downloads, this demo
uses `gpt2` and two bundled English datasets — the *mechanics* are identical. Run it with:

    python examples/cross_language_similarity.py
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from cap.data.loaders import LabeledDataset
from cap.data.prompt_templates import FACTUALITY
from cap.experiments.capture import CaptureExperiment
from cap.plot.similarity_html import create_similarity_viewer
from cap.utils.h5_utils import H5Store

MODEL = "gpt2"
BASE = Path("runs/cross_language")

# Each entry becomes one experiment directory. For a real study, keep the (text_col,
# label_col, template) fixed and vary only the dataset/language.
RUNS = {
    "quickstart": ("quickstart_synthetic.csv", "text", "label"),
    "english_smol": ("english_smol.csv", "srcs", "label"),
}


def main() -> None:
    exp = CaptureExperiment(model_path=MODEL, device="cpu")

    # 1. Capture the same factuality contrast for each run into its own directory.
    for name, (filename, text_col, label_col) in RUNS.items():
        csv_path = files("cap.data") / "files" / filename
        group_a, group_b = LabeledDataset.from_csv(
            csv_path, text_col=text_col, label_col=label_col, n_samples=8
        ).as_groups()
        exp.run_contrast(
            prompts1=[FACTUALITY.apply(t) for t in group_a],
            prompts2=[FACTUALITY.apply(t) for t in group_b],
            label1="true",
            label2="false",
            output_path=BASE / name,
        )

    # 2. Load each run's statistics and render the pairwise cosine-similarity viewer.
    #    Each directory is labelled by its name in the output HTML.
    labeled_stats = [
        (name, H5Store.load_statistics(BASE / name / "statistics.h5")) for name in RUNS
    ]
    html_path = create_similarity_viewer(labeled_stats, BASE / "similarity")
    print(f"\nCross-run similarity viewer: {html_path}")


if __name__ == "__main__":
    main()
