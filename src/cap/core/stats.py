from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests


class LayerStatistics(TypedDict):
    t_stats: np.ndarray
    p_values: np.ndarray
    p_corrected: np.ndarray
    cohens_d: np.ndarray
    mean_diff: np.ndarray
    pooled_std: np.ndarray
    significant: np.ndarray


def _ttest_elementwise(
    group1_array: np.ndarray, group2_array: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean_diff_shape = group1_array.mean(axis=0).shape
    t_stats = np.zeros(mean_diff_shape)
    p_values = np.ones(mean_diff_shape)
    for idx in np.ndindex(mean_diff_shape):
        vals1 = group1_array[(slice(None), *idx)]
        vals2 = group2_array[(slice(None), *idx)]
        if np.var(vals1) + np.var(vals2) > 1e-10:
            t, p = stats.ttest_ind(vals1, vals2)
            t_stats[idx] = t
            p_values[idx] = p
    return t_stats, p_values


def _mannwhitney_elementwise(
    group1_array: np.ndarray, group2_array: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean_diff_shape = group1_array.mean(axis=0).shape
    u_stats = np.zeros(mean_diff_shape)
    p_values = np.ones(mean_diff_shape)
    for idx in np.ndindex(mean_diff_shape):
        vals1 = group1_array[(slice(None), *idx)]
        vals2 = group2_array[(slice(None), *idx)]
        if len(np.unique(np.concatenate([vals1, vals2]))) > 1:
            u, p = stats.mannwhitneyu(vals1, vals2, alternative="two-sided")
            u_stats[idx] = u
            p_values[idx] = p
    return u_stats, p_values


class ActivationStats:
    @staticmethod
    def compute_contrast(
        group1_acts: list[np.ndarray], group2_acts: list[np.ndarray], test: str = "ttest"
    ) -> dict[str, Any]:
        n1, n2 = len(group1_acts), len(group2_acts)

        if n1 == 0 or n2 == 0:
            raise ValueError("Empty group(s)")

        shapes = [a.shape for a in group1_acts + group2_acts]
        if len(set(shapes)) > 1:
            min_shape = min(shapes)
            group1_acts = [
                a[: min_shape[0]] if len(a.shape) == 1 else a[: min_shape[0], : min_shape[1]]
                for a in group1_acts
            ]
            group2_acts = [
                a[: min_shape[0]] if len(a.shape) == 1 else a[: min_shape[0], : min_shape[1]]
                for a in group2_acts
            ]

        group1_array = np.stack(group1_acts)
        group2_array = np.stack(group2_acts)

        mean1 = np.mean(group1_array, axis=0)
        mean2 = np.mean(group2_array, axis=0)
        std1 = np.std(group1_array, axis=0, ddof=1)
        std2 = np.std(group2_array, axis=0, ddof=1)

        mean_diff = mean1 - mean2
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        cohens_d = mean_diff / (pooled_std + 1e-10)

        if test == "ttest":
            t_stats, p_values = _ttest_elementwise(group1_array, group2_array)

        elif test == "welch":
            t_stats = mean_diff / np.sqrt(std1**2 / n1 + std2**2 / n2 + 1e-10)
            df = (std1**2 / n1 + std2**2 / n2) ** 2 / (
                (std1**2 / n1) ** 2 / (n1 - 1) + (std2**2 / n2) ** 2 / (n2 - 1) + 1e-10
            )
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df))
            invalid = ~np.isfinite(t_stats) | ~np.isfinite(p_values)
            if invalid.any():
                t_stats[invalid] = 0.0
                p_values[invalid] = 1.0

        elif test == "mannwhitney":
            t_stats, p_values = _mannwhitney_elementwise(group1_array, group2_array)

        else:
            raise ValueError(f"Unknown test: {test}")

        return {
            "mean_diff": mean_diff,
            "t_stats": t_stats,
            "p_values": p_values,
            "cohens_d": cohens_d,
            "pooled_std": pooled_std,
            "mean1": mean1,
            "mean2": mean2,
            "std1": std1,
            "std2": std2,
            "n1": n1,
            "n2": n2,
        }

    @staticmethod
    def multiple_comparison_correction(
        p_values: np.ndarray, method: str = "fdr_bh", alpha: float = 0.05
    ) -> dict[str, Any]:
        p_flat = p_values.flatten()

        if method == "bonferroni":
            p_corrected = np.minimum(p_flat * len(p_flat), 1.0)
            significant = p_corrected < alpha
        else:
            significant, p_corrected, _, _ = multipletests(p_flat, alpha=alpha, method=method)

        return {
            "p_corrected": p_corrected.reshape(p_values.shape),
            "significant": significant.reshape(p_values.shape),
            "n_significant": np.sum(significant),
            "n_total": len(p_flat),
        }

    @staticmethod
    def compute_rsa_similarity(
        acts1: np.ndarray, acts2: np.ndarray, metric: str = "correlation"
    ) -> float:
        flat1 = acts1.flatten()
        flat2 = acts2.flatten()
        min_len = min(len(flat1), len(flat2))
        flat1 = flat1[:min_len]
        flat2 = flat2[:min_len]

        if metric == "correlation":
            if np.std(flat1) > 1e-10 and np.std(flat2) > 1e-10:
                return np.corrcoef(flat1, flat2)[0, 1]
            return 0.0
        elif metric == "cosine":
            norm1 = np.linalg.norm(flat1)
            norm2 = np.linalg.norm(flat2)
            if norm1 > 1e-10 and norm2 > 1e-10:
                return np.dot(flat1, flat2) / (norm1 * norm2)
            return 0.0
        elif metric == "euclidean":
            return -np.linalg.norm(flat1 - flat2)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    @staticmethod
    def summarize_contrast(contrast_results: dict[str, Any], top_n: int = 10) -> dict[str, Any]:
        summary = {
            "total_dims": contrast_results["p_values"].size,
            "significant_dims": np.sum(contrast_results["p_values"] < 0.05),
            "significant_corrected": 0,
            "mean_effect_size": np.mean(np.abs(contrast_results["cohens_d"])),
            "max_effect_size": np.max(np.abs(contrast_results["cohens_d"])),
            "top_differences": [],
        }
        flat_d = np.abs(contrast_results["cohens_d"].flatten())
        top_indices = np.argsort(flat_d)[-top_n:][::-1]
        for idx in top_indices:
            summary["top_differences"].append(
                {
                    "index": idx,
                    "cohens_d": contrast_results["cohens_d"].flat[idx],
                    "p_value": contrast_results["p_values"].flat[idx],
                    "mean_diff": contrast_results["mean_diff"].flat[idx],
                }
            )
        return summary
