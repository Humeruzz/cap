"""Implement a custom benchmark as an ``EvaluatorProtocol`` and plug it into `cap patch`.

`cap patch` accepts any object with this method::

    def evaluate(self, model, tokenizer, *, n_samples: int, device: str) -> dict[str, float]

That is the entire contract (`cap.core.evaluation.EvaluatorProtocol`). The patch flow calls
it on the model *after* patching neurons, so your metric measures the patch's effect.

To use this evaluator from the CLI, put this folder on your import path and pass
``module::ClassName``::

    PYTHONPATH=examples cap patch \
      --experiment runs/quickstart \
      --evaluator custom_evaluator::KeywordEvaluator \
      --scale 0.0

Run standalone (loads `gpt2` on CPU, prints a score dict) with::

    python examples/custom_evaluator.py
"""

from __future__ import annotations

import torch

from cap.core.evaluation import EvaluatorProtocol

# A tiny hand-built benchmark: does the model answer these true/false prompts correctly?
QUESTIONS = [
    ("Is the following statement true or false?\nParis is the capital of France.\nAnswer:", "true"),
    ("Is the following statement true or false?\nThe Moon is made of cheese.\nAnswer:", "false"),
    ("Is the following statement true or false?\nFish live in water.\nAnswer:", "true"),
    ("Is the following statement true or false?\nThe Sun orbits the Earth.\nAnswer:", "false"),
]


class KeywordEvaluator:
    """Score how often the model's continuation contains the expected keyword.

    Conforms to ``EvaluatorProtocol`` structurally — no base class to inherit.
    """

    def evaluate(
        self, model, tokenizer, *, n_samples: int = 0, device: str = "cpu"
    ) -> dict[str, float]:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        items = QUESTIONS[:n_samples] if n_samples > 0 else QUESTIONS
        correct = 0
        for prompt, expected in items:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=3,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            reply = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            if expected in reply.lower():
                correct += 1

        return {"keyword_accuracy": correct / len(items) if items else 0.0}


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # The protocol is runtime-checkable, so we can assert conformance before shipping.
    assert isinstance(KeywordEvaluator(), EvaluatorProtocol)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()

    scores = KeywordEvaluator().evaluate(model, tokenizer, device="cpu")
    print(f"KeywordEvaluator scores: {scores}")


if __name__ == "__main__":
    main()
