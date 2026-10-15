"""Tests for `cap patch --benchmark` registration and validation.

No model is loaded and no dataset is downloaded: the Evaluator's benchmark method and
PatchingExperiment are stubbed, so these pin only the registry and the CLI plumbing.
"""

import json

import pytest
from typer.testing import CliRunner

import cap.experiments.patch as patch_module
from cap.cli.main import app
from cap.core.evaluation import BUILTIN_EVALUATORS, Evaluator, _BenchmarkEvaluator

runner = CliRunner()


class _FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    padding_side = "right"


class _FakeModel:
    def eval(self):
        return self


class _StubPatchingExperiment:
    """Records what `cap patch` passes to run(); never loads a model."""

    last_run_kwargs: dict | None = None

    def __init__(self, **kwargs):
        pass

    def run(self, **kwargs):
        type(self).last_run_kwargs = kwargs
        return {}


@pytest.fixture
def experiment_dir(tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_text(json.dumps({"model": "stub-model"}))
    _StubPatchingExperiment.last_run_kwargs = None
    # patch_cmd builds PatchingExperiment (which loads the model) before it validates
    # --benchmark, so the stub is needed even for the unknown-benchmark path.
    monkeypatch.setattr(patch_module, "PatchingExperiment", _StubPatchingExperiment)
    return tmp_path


def _invoke_patch(experiment_dir, benchmark):
    return runner.invoke(
        app,
        [
            "--device",
            "cpu",
            "patch",
            "--experiment",
            str(experiment_dir),
            "--benchmark",
            benchmark,
        ],
    )


def test_mmmlu_is_registered():
    evaluator = BUILTIN_EVALUATORS["mmmlu"]()
    assert isinstance(evaluator, _BenchmarkEvaluator)
    assert evaluator._benchmark == "mmmlu"


def test_mmmlu_reports_mean_and_per_language_scores(monkeypatch):
    calls = {}

    def fake_evaluate_mmmlu(self, *, n_samples, **kwargs):
        calls.update(n_samples=n_samples, **kwargs)
        return {"en": 0.5, "de_de": 0.25}

    monkeypatch.setattr(Evaluator, "evaluate_mmmlu", fake_evaluate_mmmlu)
    result = BUILTIN_EVALUATORS["mmmlu"]().evaluate(
        _FakeModel(), _FakeTokenizer(), n_samples=3, device="cpu"
    )
    # Same keys as BaseExperiment.evaluate: the mean under "mmmlu", one key per language.
    assert result == pytest.approx({"mmmlu": 0.375, "mmmlu_en": 0.5, "mmmlu_de_de": 0.25})
    assert calls == {"n_samples": 3, "n_shots": 5}


def test_patch_accepts_mmmlu_benchmark(experiment_dir):
    result = _invoke_patch(experiment_dir, "mmmlu")
    assert result.exit_code == 0, result.output
    run_kwargs = _StubPatchingExperiment.last_run_kwargs
    assert run_kwargs["benchmarks"] == ["mmmlu"]
    assert isinstance(run_kwargs["evaluator"], _BenchmarkEvaluator)
    assert run_kwargs["evaluator"]._benchmark == "mmmlu"


def test_patch_unknown_benchmark_exits_1(experiment_dir):
    result = _invoke_patch(experiment_dir, "nope")
    assert result.exit_code == 1
    assert "Unknown --benchmark 'nope'" in result.output
    assert _StubPatchingExperiment.last_run_kwargs is None
