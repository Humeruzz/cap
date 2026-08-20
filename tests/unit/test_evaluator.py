"""Unit tests for cap.core.evaluation.Evaluator scoring logic.

The real evaluate_gsm8k/hellaswag/mmlu methods download benchmark datasets and run a live
model, so they belong in the integration suite. Here we drive the model-agnostic scoring
paths (_run_binary_eval, CSVEvaluator.evaluate) with a tiny deterministic fake model and
tokenizer — no network, no model download, no lm_eval. This pins the accuracy accounting
and the extract_binary integration exactly.
"""

import pytest
import torch

from cap.core.evaluation import CSVEvaluator, Evaluator
from cap.data.prompt_templates import FACTUALITY


class _FakeBatch(dict):
    """Stand-in for a transformers BatchEncoding: a dict with a no-op .to()."""

    def to(self, device):
        return self


class _FakeTokenizer:
    """Encodes every prompt to a fixed-width id row; decodes a generated id via answer_map."""

    def __init__(self, *, answer_map, pad_token="<pad>"):
        self.answer_map = answer_map
        self.pad_token = pad_token
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.padding_side = "right"
        self.prompt_len = 3

    def __call__(self, prompts, return_tensors=None, padding=None):
        batch = len(prompts)
        return _FakeBatch(
            input_ids=torch.zeros(batch, self.prompt_len, dtype=torch.long),
            attention_mask=torch.ones(batch, self.prompt_len, dtype=torch.long),
        )

    def decode(self, ids, skip_special_tokens=True):
        if len(ids) == 0:
            return ""
        return self.answer_map.get(int(ids[0]), "")


class _FakeModel:
    """generate() appends a single fixed answer token to every row."""

    def __init__(self, answer_token):
        self.answer_token = answer_token

    def eval(self):
        return self

    def generate(self, *, input_ids, attention_mask=None, max_new_tokens=1, **kwargs):
        batch = input_ids.shape[0]
        tail = torch.full((batch, 1), self.answer_token, dtype=torch.long)
        return torch.cat([input_ids, tail], dim=1)


def _make_evaluator(*, answer_token=7, answer_map=None, pad_token="<pad>"):
    answer_map = answer_map if answer_map is not None else {7: "1"}
    tokenizer = _FakeTokenizer(answer_map=answer_map, pad_token=pad_token)
    model = _FakeModel(answer_token)
    return Evaluator(model=model, tokenizer=tokenizer, device="cpu", verbose=False)


def test_init_sets_pad_token_and_padding_side():
    ev = _make_evaluator(pad_token=None)  # tokenizer has no pad token
    assert ev.tokenizer.pad_token == ev.tokenizer.eos_token  # filled from eos
    assert ev.tokenizer.padding_side == "left"


def test_run_binary_eval_scores_correct_fraction():
    ev = _make_evaluator(answer_token=7, answer_map={7: "1"})  # always predicts 1
    score = ev._run_binary_eval(
        prompts=["a", "b", "c", "d"],
        expected_answers=[1, 1, 0, 0],
        dataset_name="t",
        max_new_tokens=1,
        batch_size=2,
        desc="t",
    )
    assert score == pytest.approx(0.5)  # 2 of 4 expected == 1


def test_run_binary_eval_unparseable_response_scores_zero():
    ev = _make_evaluator(answer_token=9, answer_map={9: "no digit here"})  # extract_binary -> None
    score = ev._run_binary_eval(
        prompts=["a", "b"],
        expected_answers=[1, 0],
        dataset_name="t",
        max_new_tokens=1,
        batch_size=8,
        desc="t",
    )
    assert score == 0.0


def test_run_binary_eval_all_correct():
    ev = _make_evaluator(answer_token=3, answer_map={3: "0"})  # always predicts 0
    score = ev._run_binary_eval(
        prompts=["a", "b", "c"],
        expected_answers=[0, 0, 0],
        dataset_name="t",
        max_new_tokens=1,
        batch_size=1,
        desc="t",
    )
    assert score == pytest.approx(1.0)


def test_csv_evaluator_end_to_end(tmp_csv):
    # tmp_csv: 5 rows label==1 (group A -> expected 1) + 5 rows label==0 (expected 0).
    # Fake model always predicts 1, so accuracy == 5/10.
    ev = CSVEvaluator(tmp_csv, text_col="srcs", label_col="label", prompt_template=FACTUALITY)
    tokenizer = _FakeTokenizer(answer_map={7: "1"})
    model = _FakeModel(7)
    result = ev.evaluate(model, tokenizer, device="cpu")
    assert result == {"accuracy": pytest.approx(0.5)}
