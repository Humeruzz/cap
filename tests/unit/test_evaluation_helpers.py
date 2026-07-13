"""Unit tests for the pure, model-free surfaces of cap.core.evaluation.

The Evaluator's evaluate_* methods drive a live model over downloaded benchmarks and are
covered by the integration suite, not here. These tests pin the deterministic helpers and
the evaluator-interface plumbing that need no model or network.
"""

from cap.core.evaluation import (
    BUILTIN_EVALUATORS,
    CSVEvaluator,
    EvaluatorProtocol,
    _BenchmarkEvaluator,
    _extract_gsm8k_answer,
    _is_correct,
)
from cap.data.prompt_templates import FACTUALITY


def test_extract_gsm8k_answer_found():
    assert _extract_gsm8k_answer("reasoning...\n#### 42") == "42"
    assert _extract_gsm8k_answer("#### -3,000") == "-3,000"


def test_extract_gsm8k_answer_missing():
    assert _extract_gsm8k_answer("no marker here") is None


def test_is_correct_matches_gold():
    assert _is_correct("#### 42", 42.0) is True


def test_is_correct_rejects_wrong_and_none():
    assert _is_correct("#### 42", 41.0) is False
    assert _is_correct("#### 42", None) is False
    assert _is_correct("no gold", 42.0) is False


def test_builtin_evaluators_build_benchmark_adapters():
    for name, factory in BUILTIN_EVALUATORS.items():
        evaluator = factory()
        assert isinstance(evaluator, _BenchmarkEvaluator)
        assert evaluator._benchmark == name
        # each adapter satisfies the plug-in protocol
        assert isinstance(evaluator, EvaluatorProtocol)


def test_benchmark_evaluator_stores_kwargs():
    ev = _BenchmarkEvaluator("gsm8k", n_shots=4)
    assert ev._benchmark == "gsm8k"
    assert ev._method_kwargs == {"n_shots": 4}


def test_evaluator_protocol_rejects_non_evaluator():
    assert not isinstance(object(), EvaluatorProtocol)


def test_csv_evaluator_constructs_without_model(tmp_csv):
    ev = CSVEvaluator(tmp_csv, text_col="srcs", label_col="label", prompt_template=FACTUALITY)
    assert isinstance(ev, EvaluatorProtocol)
    group_a, group_b = ev._dataset.as_groups()
    assert len(group_a) == 5 and len(group_b) == 5
