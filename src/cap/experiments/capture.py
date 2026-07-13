from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ..core.capture import ActivationCapture
from ..core.stats import ActivationStats
from ..utils.activation_utils import flatten_activations_by_layer
from ..utils.answer_utils import extract_binary, extract_last_number
from ..utils.h5_utils import H5Store


class CaptureExperiment:
    def __init__(
        self,
        *,
        model_path: str,
        device: str | None = None,
        seed: int = 42,
        trust_remote_code: bool = False,
    ) -> None:
        resolved_device = (
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"CaptureExperiment init: using device={resolved_device}", flush=True)
        self.capture = ActivationCapture(
            model_path, resolved_device, seed=seed, trust_remote_code=trust_remote_code
        )
        self.model_path = model_path

    # ── static extractors — pass one of these as extract_fn ───────────────────

    @staticmethod
    def gsm8k_extractor(text):
        """Extract last number from generated text (GSM8K-style answers)."""
        return extract_last_number(text)

    @staticmethod
    def binary_extractor(text):
        """Extract trailing 0 or 1 from generated text (fact-check / faithfulness)."""
        return extract_binary(text)

    # ── public API ─────────────────────────────────────────────────────────────

    def run_contrast(
        self,
        *,
        prompts1,
        prompts2,
        label1,
        label2,
        output_path,
        max_new_tokens=1,
        batch_size=1,
        test="welch",
        correction_method="fdr_bh",
        skip_if_exists=True,
        dataset_info=None,
    ) -> Path:
        """Capture activations for two pre-loaded prompt lists and save contrastive stats."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        n_samples = max(len(prompts1), len(prompts2))
        info = {
            "label1": label1,
            "label2": label2,
            "max_new_tokens": max_new_tokens,
            "test": test,
            "correction_method": correction_method,
            **(dataset_info or {}),
        }

        if skip_if_exists and self._check_exists(output_path, n_samples, info):
            return output_path

        print(f"\nCapturing {label1} activations ({len(prompts1)} prompts)...")
        acts1, texts1 = self.capture.capture_prompts(
            prompts1, max_new_tokens=max_new_tokens, batch_size=batch_size
        )

        print(f"\nCapturing {label2} activations ({len(prompts2)} prompts)...")
        acts2, texts2 = self.capture.capture_prompts(
            prompts2, max_new_tokens=max_new_tokens, batch_size=batch_size
        )

        return self._compute_and_save(
            acts1=acts1,
            texts1=texts1,
            prompts1=list(prompts1),
            acts2=acts2,
            texts2=texts2,
            prompts2=list(prompts2),
            label1=label1,
            label2=label2,
            output_path=output_path,
            n_samples=n_samples,
            dataset_info=info,
            test=test,
            correction_method=correction_method,
        )

    def run_correct_incorrect(
        self,
        *,
        prompts,
        answers,
        extract_fn,
        output_path,
        max_new_tokens=1,
        batch_size=1,
        test="welch",
        correction_method="fdr_bh",
        skip_if_exists=True,
        dataset_info=None,
    ) -> Path:
        """Run prompts through the model, split by correctness, save contrastive stats.

        extract_fn(generated_text) -> predicted answer (or None if unparseable).
        Use CaptureExperiment.gsm8k_extractor or CaptureExperiment.binary_extractor,
        or supply your own callable.
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        n_samples = len(prompts)
        info = {
            "label1": "correct",
            "label2": "incorrect",
            "max_new_tokens": max_new_tokens,
            "test": test,
            "correction_method": correction_method,
            **(dataset_info or {}),
        }

        if skip_if_exists and self._check_exists(output_path, n_samples, info):
            return output_path

        print(f"\nCapturing activations for {n_samples} prompts...")
        all_acts, all_texts = self.capture.capture_prompts(
            prompts, max_new_tokens=max_new_tokens, batch_size=batch_size
        )

        print("\nClassifying correct vs incorrect...")
        (
            correct_acts,
            correct_texts,
            correct_prompts,
            incorrect_acts,
            incorrect_texts,
            incorrect_prompts,
            accuracy,
        ) = self._split_by_correctness(all_acts, all_texts, prompts, answers, extract_fn)

        print(
            f"\nResults: {len(correct_acts)} correct, {len(incorrect_acts)} incorrect ({accuracy:.2%})"
        )

        if not correct_acts or not incorrect_acts:
            raise ValueError(
                "Need both correct and incorrect samples — check extract_fn or prompts"
            )

        return self._compute_and_save(
            acts1=correct_acts,
            texts1=correct_texts,
            prompts1=correct_prompts,
            acts2=incorrect_acts,
            texts2=incorrect_texts,
            prompts2=incorrect_prompts,
            label1="correct",
            label2="incorrect",
            output_path=output_path,
            n_samples=n_samples,
            dataset_info=info,
            test=test,
            correction_method=correction_method,
            extra_info=f"Model accuracy: {accuracy:.2%}",
        )

    @staticmethod
    def compute_statistics(
        output_path, *, model_name="model", test="welch", correction_method="fdr_bh"
    ) -> Path:
        """Recompute statistics.h5 from a saved activations.h5 — no model required.

        `cap capture` already writes statistics.h5 during capture; this lets `cap stats`
        re-run the contrast from disk (e.g. with a different test) without re-capturing
        or loading the model onto the GPU.
        """
        output_path = Path(output_path)
        activations_dict, _texts, _prompts = H5Store.load_activations(
            output_path / "activations.h5"
        )
        conditions = list(activations_dict)
        if len(conditions) != 2:
            raise ValueError(
                f"Expected exactly 2 capture conditions, found {len(conditions)}: {conditions}"
            )

        # On disk each activation is flattened to (D,); re-add the singleton batch dim so
        # it matches the in-memory (1, D) shape that _compute_statistics expects.
        for samples in activations_dict.values():
            for sample in samples:
                for step in sample:
                    for name in list(step):
                        step[name] = np.asarray(step[name]).reshape(1, -1)

        label1, label2 = conditions
        layer_statistics = CaptureExperiment._compute_statistics(
            activations1=activations_dict[label1],
            activations2=activations_dict[label2],
            test=test,
            correction_method=correction_method,
        )
        H5Store.save_statistics(
            statistics=layer_statistics, output_path=output_path, model_name=model_name
        )
        return output_path

    # ── private helpers ────────────────────────────────────────────────────────

    def _check_exists(self, output_path, n_samples, dataset_info):
        exists, metadata = H5Store.check_existing_data(
            output_path, n_samples=n_samples, dataset_info=dataset_info
        )
        if exists:
            print(
                f"\n✓ Found existing data (model={metadata.get('model')}, n={metadata.get('n_samples')})"
            )
            print("  Skipping capture (use skip_if_exists=False to re-run)")
            return True
        if metadata:
            print(f"\nExisting data doesn't match: {metadata}\nRe-running capture...")
        return False

    def _split_by_correctness(self, all_acts, all_texts, prompts, answers, extract_fn):
        correct_acts, correct_texts, correct_prompts = [], [], []
        incorrect_acts, incorrect_texts, incorrect_prompts = [], [], []

        for acts, texts, prompt, expected in zip(all_acts, all_texts, prompts, answers):
            predicted = extract_fn(texts[-1])
            if self._answers_match(predicted, expected):
                correct_acts.append(acts)
                correct_texts.append(texts)
                correct_prompts.append(prompt)
            else:
                incorrect_acts.append(acts)
                incorrect_texts.append(texts)
                incorrect_prompts.append(prompt)

        n_correct = len(correct_acts)
        n_total = n_correct + len(incorrect_acts)
        accuracy = n_correct / n_total if n_total > 0 else 0

        return (
            correct_acts,
            correct_texts,
            correct_prompts,
            incorrect_acts,
            incorrect_texts,
            incorrect_prompts,
            accuracy,
        )

    def _answers_match(self, predicted, expected):
        if predicted is None or expected is None:
            return False
        try:
            return math.isclose(float(predicted), float(expected), rel_tol=0, abs_tol=1e-4)
        except Exception:
            return str(predicted) == str(expected)

    def _compute_and_save(
        self,
        *,
        acts1,
        texts1,
        prompts1,
        acts2,
        texts2,
        prompts2,
        label1,
        label2,
        output_path,
        n_samples,
        dataset_info,
        test,
        correction_method,
        extra_info=None,
    ):
        print("\nComputing statistics...")
        layer_statistics = self._compute_statistics(
            activations1=acts1,
            activations2=acts2,
            test=test,
            correction_method=correction_method,
        )

        print("\nSaving to H5...")
        H5Store.save_activations(
            activations_dict={label1: acts1, label2: acts2},
            texts_dict={label1: texts1, label2: texts2},
            prompts_dict={label1: prompts1, label2: prompts2},
            output_path=output_path,
            model_name=self.model_path,
            statistics=layer_statistics,
            n_samples=n_samples,
            dataset_info=dataset_info,
        )

        n_significant = sum(
            np.sum(layer_statistics[layer]["significant"]) for layer in layer_statistics
        )
        total_dims = sum(
            len(layer_statistics[layer]["p_values"].flatten()) for layer in layer_statistics
        )

        print("\n✓ Capture complete!")
        print(f"Output: {output_path}")
        if extra_info:
            print(f"{extra_info}")
        print(
            f"Significant dims: {n_significant}/{total_dims} ({100 * n_significant / total_dims:.2f}%)"
        )

        return output_path

    @staticmethod
    def _compute_statistics(
        *, activations1, activations2, test="welch", correction_method="fdr_bh"
    ):
        acts1_by_layer = flatten_activations_by_layer(activation_list=activations1)
        acts2_by_layer = flatten_activations_by_layer(activation_list=activations2)

        common_layers = set(acts1_by_layer.keys()) & set(acts2_by_layer.keys())
        layer_statistics = {}

        print(f"Computing statistics for {len(common_layers)} layers...")

        for layer in tqdm(sorted(common_layers), desc="Computing stats"):
            vals1 = acts1_by_layer[layer]
            vals2 = acts2_by_layer[layer]

            contrast = ActivationStats.compute_contrast(vals1, vals2, test=test)
            correction = ActivationStats.multiple_comparison_correction(
                contrast["p_values"], method=correction_method, alpha=0.05
            )

            n1, n2 = contrast["n1"], contrast["n2"]
            std1, std2 = contrast["std1"], contrast["std2"]
            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

            arr1, arr2 = np.array(vals1), np.array(vals2)

            layer_statistics[layer] = {
                "t_stats": contrast["t_stats"],
                "p_values": contrast["p_values"],
                "p_corrected": correction["p_corrected"],
                "significant": correction["significant"],
                "mean_diff": contrast["mean_diff"],
                "cohens_d": contrast["cohens_d"],
                "pooled_std": pooled_std,
                "group1_mean": np.mean(arr1, axis=0),
                "group1_std": np.std(arr1, axis=0, ddof=1),
                "group2_mean": np.mean(arr2, axis=0),
                "group2_std": np.std(arr2, axis=0, ddof=1),
                "n1": len(vals1),
                "n2": len(vals2),
            }

            n_sig = correction["n_significant"]
            n_total = correction["n_total"]
            print(f"  {layer}: {n_sig}/{n_total} significant ({100 * n_sig / n_total:.2f}%)")

        return layer_statistics
