from __future__ import annotations

import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import datasets
import numpy as np
import torch
from datasets import load_dataset
from torch.nn import functional as F
from tqdm import tqdm

from cap.data.builtin_datasets import DatasetLoader
from cap.utils.answer_utils import extract_binary, extract_last_number

datasets.logging.set_verbosity_error()

if TYPE_CHECKING:
    from cap.data.prompt_templates import PromptTemplate


@runtime_checkable
class EvaluatorProtocol(Protocol):
    """Implement this to plug in any custom benchmark."""

    def evaluate(
        self, model: Any, tokenizer: Any, *, n_samples: int, device: str
    ) -> dict[str, float]: ...


def _extract_gsm8k_answer(text: str) -> str | None:
    match = re.search(r"#### (\-?[0-9\.\,]+)", text)
    return match.group(1) if match else None


def _is_correct(gold_text: str, pred: float | None) -> bool:
    gold = extract_last_number(_extract_gsm8k_answer(gold_text) or "")
    return gold is not None and pred is not None and math.isclose(gold, pred, abs_tol=1e-4)


class Evaluator:
    def __init__(
        self, *, model: Any, tokenizer: Any, device: str = "cuda", verbose: bool = True
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.verbose = verbose

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model.eval()

    def evaluate_gsm8k(
        self,
        *,
        n_samples: int = 100,
        batch_size: int = 256,
        n_shots: int = 5,
        max_new_tokens: int = 256,
        return_details: bool = False,
    ) -> float | tuple[list[dict[str, Any]] | None, float]:
        if self.verbose:
            print(f"  Evaluating GSM8K ({n_samples} samples, {n_shots}-shot)...")

        dataset = load_dataset("openai/gsm8k", "main", split="test")
        dataset = dataset.shuffle(seed=42).select(range(min(n_samples, len(dataset))))

        few_shot_prompt = ""
        if n_shots > 0:
            train_dataset = load_dataset("openai/gsm8k", "main", split="train")
            few_shot_examples = train_dataset.shuffle(seed=42).select(range(n_shots))
            for ex in few_shot_examples:
                few_shot_prompt += (
                    f"Question: {ex['question']}\nLet's think step by step\n{ex['answer']}\n\n"
                )

        correct = 0
        total = 0
        details: list[dict[str, Any]] | None = [] if return_details else None

        prompts = [
            few_shot_prompt + f"Question: {ex['question']}\nLet's think step by step\n"
            for ex in dataset
        ]
        answers = [ex["answer"] for ex in dataset]

        for i in tqdm(range(0, len(prompts), batch_size), desc="GSM8K", disable=not self.verbose):
            batch_prompts = prompts[i : i + batch_size]
            batch_answers = answers[i : i + batch_size]

            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True).to(
                self.device
            )

            stop_tokens = [
                t
                for t in [
                    self.tokenizer.eos_token_id,
                    self.tokenizer.convert_tokens_to_ids("<|endoftext|>"),
                    self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                ]
                if t is not None and t != self.tokenizer.unk_token_id
            ]

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=stop_tokens or None,
                )

            for j, output in enumerate(outputs):
                response = self.tokenizer.decode(
                    output[inputs["input_ids"][j].shape[0] :], skip_special_tokens=True
                )
                response = (
                    response.split("<|endoftext|>")[0]
                    .split("\n\n\n")[0]
                    .split("\n\n")[0]
                    .split("Question:")[0]
                )

                predicted = extract_last_number(response)
                is_correct_flag = _is_correct(batch_answers[j], predicted)

                if is_correct_flag:
                    correct += 1
                total += 1

                if details is not None:
                    details.append(
                        {
                            "expected": _extract_gsm8k_answer(batch_answers[j]),
                            "generation": response,
                            "predicted": predicted,
                            "correct": is_correct_flag,
                        }
                    )

        score = correct / total if total > 0 else 0
        if self.verbose:
            print(f"    gsm8k: {score:.3f} ({score * 100:.1f}%)")
        return (details, score) if return_details else score

    def evaluate_hellaswag(self, *, n_samples: int = 100) -> float:
        if self.verbose:
            print(f"  Evaluating HellaSwag ({n_samples} samples)...")

        dataset = load_dataset("Rowan/hellaswag", split="validation")
        dataset = dataset.shuffle(seed=42).select(range(min(n_samples, len(dataset))))

        correct = 0
        total = 0

        for example in tqdm(dataset, desc="HellaSwag", disable=not self.verbose):
            context = example["ctx"]
            endings = example["endings"]
            label = example["label"]

            context_ids = self.tokenizer(context, return_tensors="pt").input_ids.to(self.device)
            context_len = context_ids.shape[1]

            scores = []
            for ending in endings:
                inputs = self.tokenizer(
                    context + " " + ending, return_tensors="pt", max_length=512, truncation=True
                ).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = inputs["input_ids"][:, 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits[:, context_len - 1 :].reshape(-1, shift_logits.size(-1)),
                        shift_labels[:, context_len - 1 :].reshape(-1),
                        reduction="mean",
                    )
                    scores.append(-loss.item())

            if np.argmax(scores) == int(label):
                correct += 1
            total += 1

        score = correct / total if total > 0 else 0
        if self.verbose:
            print(f"    hellaswag: {score:.3f} ({score * 100:.1f}%)")
        return score

    def evaluate_mmlu(
        self, *, n_samples: int = 100, n_shots: int = 5, subjects: list[str] | None = None
    ) -> float:
        if subjects is None:
            subjects = ["high_school_geography", "philosophy", "prehistory", "sociology"]
        if self.verbose:
            print(
                f"  Evaluating MMLU ({n_samples} samples, {n_shots}-shot, {len(subjects)} subjects)..."
            )

        def format_example(example, *, include_answer=True):
            prompt = example["question"] + "\n"
            for i, choice in enumerate(["A", "B", "C", "D"]):
                prompt += f"{choice}. {example['choices'][i]}\n"
            prompt += "Answer:"
            if include_answer:
                prompt += f" {['A', 'B', 'C', 'D'][example['answer']]}\n\n"
            return prompt

        all_scores = []

        for subject in tqdm(subjects, desc="MMLU subjects", disable=not self.verbose):
            dataset = load_dataset("cais/mmlu", subject)
            dev_set = dataset["dev"]
            test_set = (
                dataset["test"].shuffle(seed=42).select(range(min(n_samples, len(dataset["test"]))))
            )

            few_shot_prompt = "Answer the following multiple choice questions.\n\n"
            for example in dev_set.select(range(min(n_shots, len(dev_set)))):
                few_shot_prompt += format_example(example, include_answer=True)

            correct = 0
            total = 0

            for example in test_set:
                prompt = few_shot_prompt + format_example(example, include_answer=False)
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=1,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                response = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )
                predicted = response.strip().upper()
                if (
                    predicted
                    and predicted[0] in ["A", "B", "C", "D"]
                    and ["A", "B", "C", "D"].index(predicted[0]) == example["answer"]
                ):
                    correct += 1
                total += 1

            subject_score = correct / total if total > 0 else 0
            all_scores.append(subject_score)
            if self.verbose:
                print(f"      {subject}: {subject_score:.3f}")

        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
        if self.verbose:
            print(f"    mmlu: {avg_score:.3f} ({avg_score * 100:.1f}%)")
        return avg_score

    # evaluate_mmmlu and evaluate_gem are NOT rewritten — keep them from the original file

    def evaluate_mmmlu(
        self,
        *,
        n_samples: int = 100,
        n_shots: int = 5,
        languages: list[str] | None = None,
    ) -> dict[str, float]:
        if languages is None:
            languages = getattr(self, "mmmlu_languages", ["EN", "AR_XY", "ZH_CN", "DE_DE", "ES_ES"])

        if self.verbose:
            print(
                f"  Evaluating MMMLU ({n_samples} samples, {n_shots}-shot, {len(languages)} languages)..."
            )

        results = {}
        for lang_code in languages:
            if lang_code.upper() == "EN":
                results["en"] = self.evaluate_mmlu(n_samples=n_samples, n_shots=n_shots)
            else:
                results[lang_code.lower()] = self._evaluate_mmmlu_lang(
                    lang_code, n_samples=n_samples, n_shots=n_shots
                )

        mean_score = sum(results.values()) / len(results) if results else 0
        if self.verbose:
            print(f"    mmmlu mean: {mean_score:.3f} ({mean_score * 100:.1f}%)")

        return results

    def _evaluate_mmmlu_lang(
        self, lang_code: str, *, n_samples: int = 100, n_shots: int = 5
    ) -> float:
        # Few-shot from English cais/mmlu dev split (openai/MMMLU has no dev split)
        dev_set = load_dataset("cais/mmlu", "all")["dev"]

        def format_cais_example(ex, *, include_answer=True):
            prompt = ex["question"] + "\n"
            for i, letter in enumerate(["A", "B", "C", "D"]):
                prompt += f"{letter}. {ex['choices'][i]}\n"
            prompt += "Answer:"
            if include_answer:
                prompt += f" {['A', 'B', 'C', 'D'][ex['answer']]}\n\n"
            return prompt

        # openai/MMMLU columns: "Question", "A", "B", "C", "D", "Answer" (letter)
        def format_mmmlu_example(ex, *, include_answer=True):
            prompt = ex["Question"] + "\n"
            for letter in ["A", "B", "C", "D"]:
                prompt += f"{letter}. {ex[letter]}\n"
            prompt += "Answer:"
            if include_answer:
                prompt += f" {ex['Answer']}\n\n"
            return prompt

        few_shot_prompt = "Answer the following multiple choice questions.\n\n"
        for ex in dev_set.select(range(min(n_shots, len(dev_set)))):
            few_shot_prompt += format_cais_example(ex, include_answer=True)

        test_set = load_dataset("openai/MMMLU", lang_code, split="test")
        test_set = test_set.shuffle(seed=42).select(range(min(n_samples, len(test_set))))

        correct = 0
        total = 0

        for ex in tqdm(test_set, desc=f"MMMLU-{lang_code}", disable=not self.verbose):
            prompt = few_shot_prompt + format_mmmlu_example(ex, include_answer=False)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            response = (
                self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )
                .strip()
                .upper()
            )

            if response and response[0] == str(ex["Answer"]).strip().upper():
                correct += 1
            total += 1

        score = correct / total if total > 0 else 0
        if self.verbose:
            print(f"      mmmlu_{lang_code.lower()}: {score:.3f}")
        return score

    def evaluate_gem(
        self,
        *,
        n_samples: int = 100,
        n_shots: int = 3,
        gem_task: str = "GEM/web_nlg",
        subset: str = "en",
    ) -> dict[str, float]:
        import json as _json

        import evaluate as hf_evaluate

        if self.verbose:
            print(f"  Evaluating GEM/web_nlg ({n_samples} samples, {n_shots}-shot)...")

        gem_data_path = getattr(self, "gem_data_path", None)
        if gem_data_path is not None:
            with open(gem_data_path) as f:
                cached = _json.load(f)
            train_examples = cached["train"]
            test_examples = cached["test"][:n_samples]
        else:
            dataset = load_dataset(gem_task, subset)
            train_examples = list(
                dataset["train"]
                .shuffle(seed=42)
                .select(range(min(n_shots + 50, len(dataset["train"]))))
            )
            test_examples = list(
                dataset["test"].shuffle(seed=42).select(range(min(n_samples, len(dataset["test"]))))
            )

        def format_triples(triples):
            parts = []
            for t in triples:
                if isinstance(t, dict):
                    parts.append(f"{t['subject']} | {t['predicate']} | {t['object']}")
                else:
                    parts.append(str(t).replace("_", " | "))
            return " ; ".join(parts)

        def make_prompt(triples, target=None):
            triple_str = format_triples(triples)
            prompt = f"Given the triples: {triple_str}\nGenerate a description:"
            if target is not None:
                prompt += f" {target}\n\n"
            return prompt

        few_shot_prompt = ""
        shot_count = 0
        for ex in train_examples:
            if shot_count >= n_shots:
                break
            refs = ex.get("references") or [ex.get("target", "")]
            if not refs or not refs[0]:
                continue
            few_shot_prompt += make_prompt(ex["input"], target=refs[0])
            shot_count += 1

        bleu_metric = hf_evaluate.load("bleu")
        rouge_metric = hf_evaluate.load("rouge")

        predictions = []
        references = []

        for ex in tqdm(test_examples, desc="GEM", disable=not self.verbose):
            refs = ex.get("references") or [ex.get("target", "")]
            prompt = few_shot_prompt + make_prompt(ex["input"])
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            response = response.split("\n")[0].strip()

            predictions.append(response)
            references.append(refs)

        bleu_val = bleu_metric.compute(predictions=predictions, references=references)["bleu"]
        rouge_scores = rouge_metric.compute(
            predictions=predictions, references=[r[0] for r in references]
        )
        rouge_l_val = rouge_scores["rougeL"]

        if self.verbose:
            print(f"    gem bleu: {bleu_val:.3f}, rouge_l: {rouge_l_val:.3f}")

        return {"bleu": bleu_val, "rouge_l": rouge_l_val}

    def _run_binary_eval(
        self,
        *,
        prompts: list[str],
        expected_answers: list[int],
        dataset_name: str,
        max_new_tokens: int,
        batch_size: int,
        desc: str,
    ) -> float:
        if self.verbose:
            print(f"  Evaluating {desc} ({dataset_name}, {len(prompts)} samples)...")

        correct = 0
        total = 0

        for i in tqdm(range(0, len(prompts), batch_size), desc=desc, disable=not self.verbose):
            batch_prompts = prompts[i : i + batch_size]
            batch_expected = expected_answers[i : i + batch_size]

            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True).to(
                self.device
            )

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            for j, output in enumerate(outputs):
                response = self.tokenizer.decode(
                    output[inputs["input_ids"][j].shape[0] :], skip_special_tokens=True
                )
                predicted = extract_binary(response)
                if predicted is not None and predicted == batch_expected[j]:
                    correct += 1
                total += 1

        score = correct / total if total > 0 else 0
        if self.verbose:
            print(f"    {desc}: {score:.3f} ({score * 100:.1f}%)")
        return score

    def evaluate_fact_check_csv(
        self,
        *,
        csv_path: str,
        dataset_name: str,
        n_samples: int = 0,
        max_new_tokens: int = 1,
        batch_size: int = 32,
    ) -> float:
        prompts, answers = DatasetLoader.load_fact_check_csv(csv_path, n_samples)
        return self._run_binary_eval(
            prompts=prompts,
            expected_answers=answers,
            dataset_name=dataset_name,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            desc="FactCheck",
        )

    def evaluate_faithful_check_csv(
        self,
        *,
        csv_path: str,
        dataset_name: str,
        n_samples: int = 0,
        max_new_tokens: int = 1,
        batch_size: int = 32,
    ) -> float:
        prompts, answers = DatasetLoader.load_faithfulness_csv(csv_path, n_samples)
        return self._run_binary_eval(
            prompts=prompts,
            expected_answers=answers,
            dataset_name=dataset_name,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            desc="FaithCheck",
        )


class CSVEvaluator:
    """
    Evaluate a model on any labeled CSV.
    prompt_template is required — it must match the template used during cap capture.
    """

    def __init__(
        self,
        csv_path: str | Path,
        *,
        text_col: str,
        label_col: str,
        prompt_template: PromptTemplate,  # required, no default
    ) -> None:
        from cap.data.loaders import LabeledDataset

        self._dataset = LabeledDataset.from_csv(csv_path, text_col=text_col, label_col=label_col)
        self._template = prompt_template

    def evaluate(
        self, model, tokenizer, *, n_samples: int = 0, device: str = "cuda"
    ) -> dict[str, float]:
        group_a, group_b = self._dataset.as_groups()
        if n_samples > 0:
            group_a, group_b = group_a[:n_samples], group_b[:n_samples]

        # group_a rows carry the positive label (group_a_label, default 1); group_b the rest.
        prompts = [self._template.apply(t) for t in group_a] + [
            self._template.apply(t) for t in group_b
        ]
        expected = [1] * len(group_a) + [0] * len(group_b)

        score = Evaluator(model=model, tokenizer=tokenizer, device=device)._run_binary_eval(
            prompts=prompts,
            expected_answers=expected,
            dataset_name="csv",
            max_new_tokens=1,
            batch_size=32,
            desc="CSVEval",
        )
        return {"accuracy": score}


class _BenchmarkEvaluator:
    """Adapt a built-in Evaluator benchmark to the EvaluatorProtocol.

    Lets the standard benchmarks be passed wherever a custom evaluator is accepted, so the
    CLI and the patch flow speak a single evaluator interface — no special-casing.
    """

    def __init__(self, benchmark: str, **method_kwargs: Any) -> None:
        self._benchmark = benchmark
        self._method_kwargs = method_kwargs

    def evaluate(
        self, model, tokenizer, *, n_samples: int = 100, device: str = "cuda"
    ) -> dict[str, float]:
        evaluator = Evaluator(model=model, tokenizer=tokenizer, device=device)
        method = getattr(evaluator, f"evaluate_{self._benchmark}")
        return {self._benchmark: method(n_samples=n_samples, **self._method_kwargs)}


# Name → zero-arg factory. Add a benchmark by registering a factory here; no dispatch edits.
BUILTIN_EVALUATORS = {
    "gsm8k": lambda: _BenchmarkEvaluator("gsm8k", n_shots=4),
    "mmlu": lambda: _BenchmarkEvaluator("mmlu", n_shots=5),
    "hellaswag": lambda: _BenchmarkEvaluator("hellaswag"),
}
