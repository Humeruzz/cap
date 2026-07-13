"""Run the full `cap run` pipeline on a labeled CSV — via the Python API, not a subprocess.

This mirrors `cap run --model gpt2 --type factuality --data <csv> ...`: it loads a
labeled CSV, captures per-layer activations for the two label groups, computes the
Welch t-test + FDR contrast, and renders the interactive stats viewer.

Runs on `gpt2` on CPU in well under a minute. Run it with:

    python examples/labeled_dataset_analysis.py
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from cap.data.loaders import LabeledDataset
from cap.data.prompt_templates import FACTUALITY
from cap.experiments.capture import CaptureExperiment
from cap.plot.stats_html import create_interactive_viewer
from cap.utils.h5_utils import H5Store

MODEL = "gpt2"
OUTPUT = Path("runs/labeled_dataset_analysis")


def main() -> None:
    # 1. Load any labeled CSV. Rows with label 1 form group A, the rest group B.
    #    Here we use the bundled quickstart dataset; swap in your own CSV + column names.
    csv_path = files("cap.data") / "files" / "quickstart_synthetic.csv"
    dataset = LabeledDataset.from_csv(csv_path, text_col="text", label_col="label")
    group_a, group_b = dataset.as_groups()
    print(f"Loaded {len(dataset)} rows: {len(group_a)} true, {len(group_b)} false")

    # 2. Run the capture → statistics pipeline (the heart of `cap run`) on CPU.
    #    Apply the factuality prompt template to every row, exactly as the CLI does.
    exp = CaptureExperiment(model_path=MODEL, device="cpu")
    exp.run_contrast(
        prompts1=[FACTUALITY.apply(t) for t in group_a],
        prompts2=[FACTUALITY.apply(t) for t in group_b],
        label1="true",
        label2="false",
        output_path=OUTPUT,
    )

    # 3. Load the saved statistics and render the interactive HTML viewer.
    statistics = H5Store.load_statistics(OUTPUT / "statistics.h5")
    stats_for_html = {
        "t_stats": {layer: statistics[layer]["t_stats"] for layer in statistics},
        "p_values": {layer: statistics[layer]["p_corrected"] for layer in statistics},
        "cohens_d": {layer: statistics[layer]["cohens_d"] for layer in statistics},
        "mean_diff": {layer: statistics[layer]["mean_diff"] for layer in statistics},
        "pooled_std": {layer: statistics[layer]["pooled_std"] for layer in statistics},
    }
    html_path = create_interactive_viewer(stats_results=stats_for_html, output_path=OUTPUT)
    print(f"\nInteractive stats viewer: {html_path}")


if __name__ == "__main__":
    main()
