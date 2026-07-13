import warnings

import numpy as np
import pytest
from cap.core.stats import ActivationStats


def test_identical_groups_zero_t_stat():
    data = [np.random.randn(64) for _ in range(10)]
    result = ActivationStats.compute_contrast(data, data, test="welch")
    assert np.allclose(result["t_stats"], 0.0, atol=1e-6)


def test_empty_groups_raises():
    with pytest.raises(ValueError):
        ActivationStats.compute_contrast([], [], test="welch")


def test_fdr_correction_applied():
    rng = np.random.default_rng(0)
    a = [rng.standard_normal(1000) for _ in range(5)]
    b = [rng.standard_normal(1000) for _ in range(5)]
    result = ActivationStats.compute_contrast(a, b, test="welch")
    correction = ActivationStats.multiple_comparison_correction(result["p_values"])
    assert np.all(correction["p_corrected"] >= result["p_values"] - 1e-10)


def test_ttest_path_produces_pvalues():
    rng = np.random.default_rng(1)
    a = [rng.standard_normal(8) + 5.0 for _ in range(6)]
    b = [rng.standard_normal(8) for _ in range(6)]
    result = ActivationStats.compute_contrast(a, b, test="ttest")
    assert result["t_stats"].shape == (8,)
    assert np.all(result["p_values"] >= 0) and np.all(result["p_values"] <= 1)
    # clearly separated means -> at least one significant dimension
    assert np.any(result["p_values"] < 0.05)


def test_mannwhitney_path_produces_pvalues():
    rng = np.random.default_rng(2)
    a = [rng.standard_normal(8) + 5.0 for _ in range(6)]
    b = [rng.standard_normal(8) for _ in range(6)]
    result = ActivationStats.compute_contrast(a, b, test="mannwhitney")
    assert result["p_values"].shape == (8,)
    assert np.all(result["p_values"] >= 0) and np.all(result["p_values"] <= 1)


def test_unknown_test_raises():
    data = [np.random.randn(4) for _ in range(3)]
    with pytest.raises(ValueError, match="Unknown test"):
        ActivationStats.compute_contrast(data, data, test="nope")


def test_mismatched_shapes_are_trimmed():
    a = [np.random.randn(6), np.random.randn(4)]  # differing lengths
    b = [np.random.randn(5), np.random.randn(5)]
    result = ActivationStats.compute_contrast(a, b, test="welch")
    # trimmed to the smallest length across all inputs (4)
    assert result["mean_diff"].shape == (4,)


def test_welch_invalid_values_sanitized():
    # single-sample groups -> ddof=1 std is nan -> welch t/p go non-finite and get sanitized
    a = [np.array([1.0, 2.0, 3.0])]
    b = [np.array([4.0, 5.0, 6.0])]
    with warnings.catch_warnings():  # the nan/inf here is the exact condition under test
        warnings.simplefilter("ignore", RuntimeWarning)
        result = ActivationStats.compute_contrast(a, b, test="welch")
    assert np.all(np.isfinite(result["t_stats"]))
    assert np.all(result["p_values"] == 1.0)


def test_bonferroni_correction():
    p_values = np.array([0.001, 0.5, 0.02, 0.9])
    correction = ActivationStats.multiple_comparison_correction(p_values, method="bonferroni")
    assert correction["p_corrected"].shape == p_values.shape
    assert np.all(correction["p_corrected"] <= 1.0)
    assert correction["n_total"] == 4


def test_rsa_similarity_metrics():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([2.0, 4.0, 6.0, 8.0])
    assert ActivationStats.compute_rsa_similarity(a, b, metric="correlation") == pytest.approx(1.0)
    assert ActivationStats.compute_rsa_similarity(a, b, metric="cosine") == pytest.approx(1.0)
    assert ActivationStats.compute_rsa_similarity(a, b, metric="euclidean") < 0.0


def test_rsa_similarity_zero_variance_returns_zero():
    flat = np.zeros(4)
    assert ActivationStats.compute_rsa_similarity(flat, flat, metric="correlation") == 0.0
    assert ActivationStats.compute_rsa_similarity(flat, flat, metric="cosine") == 0.0


def test_rsa_similarity_unknown_metric_raises():
    a = np.ones(4)
    with pytest.raises(ValueError, match="Unknown metric"):
        ActivationStats.compute_rsa_similarity(a, a, metric="nope")


def test_summarize_contrast_reports_top_differences():
    contrast = ActivationStats.compute_contrast(
        [np.random.randn(16) + 3.0 for _ in range(5)],
        [np.random.randn(16) for _ in range(5)],
        test="welch",
    )
    summary = ActivationStats.summarize_contrast(contrast, top_n=3)
    assert summary["total_dims"] == 16
    assert len(summary["top_differences"]) == 3
    # top_differences are sorted by descending |cohens_d|
    effects = [abs(d["cohens_d"]) for d in summary["top_differences"]]
    assert effects == sorted(effects, reverse=True)
