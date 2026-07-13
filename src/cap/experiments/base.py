from __future__ import annotations

import json
from pathlib import Path

from cap.core.patch import ActivationPatcher
from cap.core.evaluation import Evaluator


class BaseExperiment:
    def __init__(
        self,
        *,
        model_path: str,
        h5_path: str | Path,
        device: str = "cuda",
        train_lang: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        self.patcher = ActivationPatcher(
            model_path=model_path, device=device, trust_remote_code=trust_remote_code
        )
        self.evaluator = Evaluator(
            model=self.patcher.model,
            tokenizer=self.patcher.tokenizer,
            device=device,
        )
        self.h5_path = h5_path
        self.model_path = model_path
        self.train_lang = train_lang
        h5 = Path(h5_path)
        self.output_path = h5 if h5.is_dir() else h5.parent

    def load_statistics(self):
        print(f"\nLoading statistics from {self.h5_path}...")
        self.patcher.load_statistics(h5_path=self.h5_path)

    def identify_neurons(self, *, d_threshold, std_threshold):
        print(f"\nIdentifying significant neurons (d>{d_threshold}, std>{std_threshold})...")
        patch_targets, significant_neurons = self.patcher.identify_significant_neurons(
            d_threshold=d_threshold, std_threshold=std_threshold
        )
        n_neurons = sum(len(neurons) for neurons in patch_targets.values())
        n_modules = len(patch_targets)
        print(f"Found {n_modules} modules with {n_neurons} significant neurons")
        return patch_targets, significant_neurons

    def evaluate(
        self, *, benchmarks, n_samples, patch_targets=None, scale_factor=None, evaluator=None
    ):
        if patch_targets is not None:
            self.patcher.setup_patches(patch_targets=patch_targets, scale_factor=scale_factor)

        if evaluator is not None:
            # Custom EvaluatorProtocol: evaluate the (possibly patched) model directly.
            results = evaluator.evaluate(
                self.patcher.model,
                self.patcher.tokenizer,
                n_samples=n_samples,
                device=self.patcher.device,
            )
        else:
            results = {}
            for benchmark in benchmarks:
                if benchmark == "gsm8k":
                    results["gsm8k"] = self.evaluator.evaluate_gsm8k(n_samples=n_samples, n_shots=4)
                elif benchmark == "math":
                    results["math"] = self.evaluator.evaluate_math(n_samples=n_samples, n_shots=4)
                elif benchmark == "mmlu":
                    results["mmlu"] = self.evaluator.evaluate_mmlu(n_samples=n_samples, n_shots=5)
                elif benchmark == "gpqa":
                    results["gpqa"] = self.evaluator.evaluate_gpqa(n_samples=n_samples, n_shots=5)
                elif benchmark == "hellaswag":
                    results["hellaswag"] = self.evaluator.evaluate_hellaswag(n_samples=n_samples)
                elif benchmark == "mmmlu":
                    per_lang = self.evaluator.evaluate_mmmlu(n_samples=n_samples)
                    results["mmmlu"] = sum(per_lang.values()) / len(per_lang) if per_lang else 0
                    for lang, score in per_lang.items():
                        results[f"mmmlu_{lang}"] = score
                elif benchmark == "gem":
                    gem_scores = self.evaluator.evaluate_gem(n_samples=n_samples)
                    results["gem"] = gem_scores["rouge_l"]
                    results["gem_bleu"] = gem_scores["bleu"]
                elif benchmark == "fact_check":
                    results["fact_check"] = self.evaluator.evaluate_fact_check_csv(
                        csv_path=getattr(self, "csv_path", None),
                        dataset_name=getattr(self, "dataset_name", None),
                        n_samples=n_samples,
                    )

        if patch_targets is not None:
            self.patcher.clear_patches()

        return results

    def save_results(self, *, results, output_dir, filename):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / filename
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {results_path}")
        return results_path

    @staticmethod
    def load_results(results_path):
        with open(results_path) as f:
            return json.load(f)
