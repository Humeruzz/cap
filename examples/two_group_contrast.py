"""Contrast two *separate* CSV files — one per group, no label column.

Use ``TwoGroupDataset.from_csv_pair`` when your two groups already live in different files
(e.g. `faithful.csv` vs `unfaithful.csv`) instead of one labeled file. Both files must
share the same text column name; there is no label column.

This is the Python equivalent of::

    cap capture --model gpt2 --type factuality \
      --group-a true.csv --group-b false.csv --text-column text \
      --output runs/two_group

Runs on `gpt2` on CPU. Run it with::

    python examples/two_group_contrast.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cap.data.loaders import TwoGroupDataset
from cap.data.prompt_templates import FACTUALITY
from cap.experiments.capture import CaptureExperiment

MODEL = "gpt2"
OUTPUT = Path("runs/two_group_contrast")


def _write_example_csvs() -> tuple[Path, Path]:
    """Create two single-column CSVs (no label column) to stand in for your own data."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    true_path, false_path = OUTPUT / "true.csv", OUTPUT / "false.csv"
    pd.DataFrame(
        {
            "text": [
                "The Eiffel Tower is in Paris.",
                "Water is made of hydrogen and oxygen.",
                "Honey is produced by bees.",
                "A triangle has three sides.",
            ]
        }
    ).to_csv(true_path, index=False)
    pd.DataFrame(
        {
            "text": [
                "The Moon is made of cheese.",
                "The Sun orbits the Earth.",
                "Spiders have six legs.",
                "A square has five corners.",
            ]
        }
    ).to_csv(false_path, index=False)
    return true_path, false_path


def main() -> None:
    path_a, path_b = _write_example_csvs()

    # Load the two files as group A / group B. Both must share the `text` column.
    group_a, group_b = TwoGroupDataset.from_csv_pair(path_a, path_b, text_col="text").as_groups()
    print(f"Group A: {len(group_a)} rows, Group B: {len(group_b)} rows")

    # Capture the contrast and compute statistics, same as `cap capture --group-a/--group-b`.
    exp = CaptureExperiment(model_path=MODEL, device="cpu")
    exp.run_contrast(
        prompts1=[FACTUALITY.apply(t) for t in group_a],
        prompts2=[FACTUALITY.apply(t) for t in group_b],
        label1="group_a",
        label2="group_b",
        output_path=OUTPUT,
    )
    print(f"\nStatistics written to {OUTPUT / 'statistics.h5'}")


if __name__ == "__main__":
    main()
