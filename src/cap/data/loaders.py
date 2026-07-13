from __future__ import annotations

import random
from pathlib import Path
from typing import Union

import pandas as pd


def _subsample(group_a, group_b, n_samples, seed):
    """Cap each group at n_samples rows (0 = keep all), drawing from one shared RNG."""
    if n_samples <= 0:
        return group_a, group_b
    rng = random.Random(seed)
    if len(group_a) > n_samples:
        group_a = rng.sample(group_a, n_samples)
    if len(group_b) > n_samples:
        group_b = rng.sample(group_b, n_samples)
    return group_a, group_b


class LabeledDataset:
    """
    Load any CSV with a text column and a binary label column.
    Splits rows by label value to form two contrastive groups.
    """

    def __init__(self, group_a: list[str], group_b: list[str]) -> None:
        self._group_a = group_a
        self._group_b = group_b

    @classmethod
    def from_csv(
        cls,
        path: Union[str, Path],
        *,
        text_col: str,
        label_col: str,
        group_a_label: Union[int, str] = 1,
        n_samples: int = 0,
        seed: int = 42,
    ) -> "LabeledDataset":
        """
        Parameters
        ----------
        path : path to the CSV file or importlib.resources path object
        text_col : name of the column containing the text to feed to the model
        label_col : name of the binary label column
        group_a_label : rows with this label value become group A; all others become group B
        n_samples : max rows to draw from each group (0 = use all rows)
        seed : random seed for sampling
        """
        df = pd.read_csv(path)

        if text_col not in df.columns:
            raise ValueError(
                f"text_col '{text_col}' not found in {path}. Available columns: {list(df.columns)}"
            )
        if label_col not in df.columns:
            raise ValueError(
                f"label_col '{label_col}' not found in {path}. Available columns: {list(df.columns)}"
            )

        mask = df[label_col] == group_a_label
        group_a_rows = df[mask][text_col].dropna().tolist()
        group_b_rows = df[~mask][text_col].dropna().tolist()

        return cls(*_subsample(group_a_rows, group_b_rows, n_samples, seed))

    def as_groups(self) -> tuple[list[str], list[str]]:
        """Return (group_a_texts, group_b_texts)."""
        return list(self._group_a), list(self._group_b)

    def __len__(self) -> int:
        return len(self._group_a) + len(self._group_b)


class TwoGroupDataset:
    """
    Load two separate CSV files as group A and group B.
    Both files must share the same text column name.
    """

    def __init__(self, group_a: list[str], group_b: list[str]) -> None:
        self._group_a = group_a
        self._group_b = group_b

    @classmethod
    def from_csv_pair(
        cls,
        path_a: Union[str, Path],
        path_b: Union[str, Path],
        *,
        text_col: str,
        n_samples: int = 0,
        seed: int = 42,
    ) -> "TwoGroupDataset":
        """
        Parameters
        ----------
        path_a : CSV file for group A
        path_b : CSV file for group B
        text_col : column name that must exist in both files
        n_samples : max rows per group (0 = all)
        seed : random seed for sampling
        """
        df_a = pd.read_csv(path_a)
        df_b = pd.read_csv(path_b)

        for path, df in [(path_a, df_a), (path_b, df_b)]:
            if text_col not in df.columns:
                raise ValueError(
                    f"text_col '{text_col}' not found in {path}. Available columns: {list(df.columns)}"
                )

        group_a = df_a[text_col].dropna().tolist()
        group_b = df_b[text_col].dropna().tolist()

        return cls(*_subsample(group_a, group_b, n_samples, seed))

    def as_groups(self) -> tuple[list[str], list[str]]:
        return list(self._group_a), list(self._group_b)

    def __len__(self) -> int:
        return len(self._group_a) + len(self._group_b)
