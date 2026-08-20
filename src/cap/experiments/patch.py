from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cap.experiments.base import BaseExperiment


class PatchingExperiment(BaseExperiment):
    @staticmethod
    def load_results(h5_path, n_eval_samples=None):
        """Load results from file. Requires n_eval_samples to locate the correct file."""
        base_path = Path(h5_path).parent / "patching_results"

        if n_eval_samples is None:
            all_results = list(base_path.glob("all_results_n*.json"))
            if not all_results:
                return None

            if len(all_results) > 1:
                print(
                    "Warning: Multiple result files found. Specify n_eval_samples to load a specific one:"
                )
                for f in all_results:
                    print(f"  - {f.name}")
                return None

            results_file = all_results[0]
        else:
            results_file = base_path / f"all_results_n{n_eval_samples}.json"

        if not results_file.exists():
            return None

        with open(results_file) as f:
            data = json.load(f)

        return data.get("results")

    def _get_completed_combinations(self, existing_results):
        """Extract which (d, std, scale, train_lang) combinations are already computed"""
        completed = set()

        if not existing_results or "experiments" not in existing_results:
            return completed

        for exp in existing_results["experiments"]:
            d_threshold = exp["d_threshold"]
            std_threshold = exp["std_threshold"]
            train_lang = exp.get("train_lang")

            for scale_result in exp.get("scale_results", []):
                scale = scale_result["scale_factor"]
                completed.add((d_threshold, std_threshold, scale, train_lang))

        return completed

    def _merge_results(self, existing_results, new_experiments):
        """Merge new experiments into existing results structure"""
        if not existing_results:
            return {"baseline": {}, "experiments": new_experiments}

        existing_exp_dict = {}
        for exp in existing_results.get("experiments", []):
            key = (exp["d_threshold"], exp["std_threshold"], exp.get("train_lang"))
            existing_exp_dict[key] = exp

        for new_exp in new_experiments:
            key = (new_exp["d_threshold"], new_exp["std_threshold"], new_exp.get("train_lang"))

            if key in existing_exp_dict:
                existing_scales = {
                    r["scale_factor"] for r in existing_exp_dict[key]["scale_results"]
                }

                for scale_result in new_exp["scale_results"]:
                    if scale_result["scale_factor"] not in existing_scales:
                        existing_exp_dict[key]["scale_results"].append(scale_result)

                existing_exp_dict[key]["scale_results"].sort(key=lambda x: x["scale_factor"])
            else:
                existing_exp_dict[key] = new_exp

        merged_experiments = list(existing_exp_dict.values())
        merged_experiments.sort(
            key=lambda x: (x["d_threshold"], x["std_threshold"], x.get("train_lang") or "")
        )

        return {"baseline": existing_results.get("baseline", {}), "experiments": merged_experiments}

    def _get_baseline(self, existing_results, benchmarks, n_eval_samples, evaluator=None):
        if existing_results and existing_results.get("baseline"):
            print(f"\n{'=' * 60}\nUSING EXISTING BASELINE\n{'=' * 60}")
            baseline = existing_results["baseline"]
            for bench, score in baseline.items():
                print(f"  {bench}: {score:.3f}")
            return baseline

        print(f"\n{'=' * 60}\nBASELINE EVALUATION\n{'=' * 60}")
        baseline = self.evaluate(
            benchmarks=benchmarks, n_samples=n_eval_samples, evaluator=evaluator
        )
        for bench, score in baseline.items():
            print(f"  {bench}: {score:.3f}")
        return baseline

    def _load_cached_result(self, existing_results, d, std, scale, train_lang):
        if not existing_results:
            return None
        for exp in existing_results.get("experiments", []):
            if (
                exp["d_threshold"] == d
                and exp["std_threshold"] == std
                and exp.get("train_lang") == train_lang
            ):
                for r in exp.get("scale_results", []):
                    if r["scale_factor"] == scale:
                        return r
        return None

    def run(
        self,
        *,
        scale_factors=None,
        d_thresholds=None,
        std_thresholds=None,
        n_eval_samples=100,
        benchmarks=None,
        evaluator=None,
        load_if_exists=True,
        save_checkpoints=True,
        smart_merge=True,
        results_subdir="patching_results",
    ) -> dict[str, Any]:
        # Defaults are built per call: these lists end up stored in the results dict, so a
        # shared mutable default could be aliased across runs.
        scale_factors = [0.0, 0.5, 1.0, 1.5, 2.0] if scale_factors is None else scale_factors
        d_thresholds = [0.0, 0.5, 1.0, 1.5, 2.0] if d_thresholds is None else d_thresholds
        std_thresholds = [0.0, 0.1, 0.5, 1.0] if std_thresholds is None else std_thresholds
        benchmarks = ["gsm8k", "hellaswag", "mmlu"] if benchmarks is None else benchmarks

        output_path = self.output_path / results_subdir
        output_path.mkdir(parents=True, exist_ok=True)

        results_file = output_path / f"all_results_n{n_eval_samples}.json"
        checkpoint_file = output_path / f"checkpoint_n{n_eval_samples}.json"

        existing_results = None
        completed_combinations = set()
        if load_if_exists and results_file.exists() and smart_merge:
            with open(results_file) as f:
                existing_results = json.load(f).get("results")
            completed_combinations = self._get_completed_combinations(existing_results)
            if completed_combinations:
                print(
                    f"\nSMART MERGE: {len(completed_combinations)} combinations already done, "
                    f"{len(d_thresholds) * len(std_thresholds) * len(scale_factors) - len(completed_combinations)} remaining"
                )

        self.load_statistics()

        checkpoint = None
        if load_if_exists and checkpoint_file.exists():
            with open(checkpoint_file) as f:
                checkpoint = json.load(f)
            expected = {
                "scale_factors": scale_factors,
                "d_thresholds": d_thresholds,
                "std_thresholds": std_thresholds,
                "n_eval_samples": n_eval_samples,
                "benchmarks": benchmarks,
            }
            if checkpoint.get("params") != expected:
                checkpoint_file.unlink()
                checkpoint = None
            else:
                print(f"\nRESUMING FROM CHECKPOINT: {checkpoint['completed']}")

        if checkpoint is not None:
            all_results = checkpoint["results"]
            completed = checkpoint["completed"]
        else:
            baseline = self._get_baseline(existing_results, benchmarks, n_eval_samples, evaluator)
            all_results = {"baseline": baseline, "experiments": []}
            completed = {"d_threshold": None, "std_threshold": None}
            if save_checkpoints:
                self._save_checkpoint(
                    output_path,
                    all_results,
                    completed,
                    scale_factors,
                    d_thresholds,
                    std_thresholds,
                    n_eval_samples,
                    benchmarks,
                )

        skipped_exp = 0
        total_experiments = len(d_thresholds) * len(std_thresholds)
        current_exp = 0

        for d_threshold in d_thresholds:
            if (
                checkpoint
                and completed["d_threshold"] is not None
                and d_threshold < completed["d_threshold"]
            ):
                print(f"\nSkipping d={d_threshold} (already completed)")
                continue

            for std_threshold in std_thresholds:
                if (
                    checkpoint
                    and completed["d_threshold"] == d_threshold
                    and completed["std_threshold"] is not None
                    and std_threshold <= completed["std_threshold"]
                ):
                    print(f"\nSkipping d={d_threshold}, std={std_threshold} (already completed)")
                    continue

                current_exp += 1
                print(
                    f"\n{'=' * 60}\nEXPERIMENT {current_exp}/{total_experiments}: d={d_threshold}, std={std_threshold}\n{'=' * 60}"
                )

                patch_targets, _ = self.identify_neurons(
                    d_threshold=d_threshold, std_threshold=std_threshold
                )
                n_neurons = sum(len(n) for n in patch_targets.values())

                if n_neurons == 0:
                    print("  No neurons meet criteria, skipping...")
                    continue

                threshold_results: dict[str, Any] = {
                    "d_threshold": d_threshold,
                    "std_threshold": std_threshold,
                    "n_neurons": n_neurons,
                    "n_layers": len(patch_targets),
                    "scale_results": [],
                }
                if self.train_lang is not None:
                    threshold_results["train_lang"] = self.train_lang

                for scale_idx, scale in enumerate(scale_factors, 1):
                    combo_key = (d_threshold, std_threshold, scale, self.train_lang)

                    if combo_key in completed_combinations:
                        print(f"\n  [{scale_idx}/{len(scale_factors)}] Scale={scale} [CACHED]")
                        cached = self._load_cached_result(
                            existing_results, d_threshold, std_threshold, scale, self.train_lang
                        )
                        if cached:
                            threshold_results["scale_results"].append(cached)
                        skipped_exp += 1
                        continue

                    print(f"\n  [{scale_idx}/{len(scale_factors)}] Scale={scale}")

                    results = self.evaluate(
                        benchmarks=benchmarks,
                        n_samples=n_eval_samples,
                        patch_targets=patch_targets,
                        scale_factor=scale,
                        evaluator=evaluator,
                    )

                    scale_result = {"scale_factor": scale}
                    for key, score in results.items():
                        baseline_score = all_results["baseline"].get(key)
                        scale_result[key] = score
                        if baseline_score is not None:
                            delta = score - baseline_score
                            scale_result[f"{key}_delta"] = delta
                            if key in benchmarks:
                                print(f"    {key}: {score:.3f} (Δ={delta:+.3f})")
                        elif key in benchmarks:
                            print(f"    {key}: {score:.3f}")

                    threshold_results["scale_results"].append(scale_result)

                all_results["experiments"].append(threshold_results)

                if save_checkpoints:
                    completed = {"d_threshold": d_threshold, "std_threshold": std_threshold}
                    self._save_checkpoint(
                        output_path,
                        all_results,
                        completed,
                        scale_factors,
                        d_thresholds,
                        std_thresholds,
                        n_eval_samples,
                        benchmarks,
                    )

        if smart_merge and existing_results:
            print("\nMERGING RESULTS")
            merged = self._merge_results(existing_results, all_results["experiments"])
            all_results = {"baseline": merged["baseline"], "experiments": merged["experiments"]}

        if save_checkpoints and checkpoint_file.exists():
            checkpoint_file.unlink()

        with open(results_file, "w") as f:
            json.dump(
                {
                    "model": self.model_path,
                    "d_thresholds": sorted({e["d_threshold"] for e in all_results["experiments"]}),
                    "std_thresholds": sorted(
                        {e["std_threshold"] for e in all_results["experiments"]}
                    ),
                    "scale_factors": sorted(
                        {
                            r["scale_factor"]
                            for e in all_results["experiments"]
                            for r in e["scale_results"]
                        }
                    ),
                    "benchmarks": benchmarks,
                    "n_eval_samples": n_eval_samples,
                    "results": all_results,
                },
                f,
                indent=2,
            )

        summary = self._create_summary(all_results, benchmarks)
        (output_path / "summary.txt").write_text(summary)
        print(f"\n{summary}")
        print(f"\n✓ Results saved to {results_file.name}")
        if skipped_exp > 0:
            print(f"✓ Skipped {skipped_exp} already-computed combinations")

        return all_results

    def _save_checkpoint(
        self,
        output_path,
        all_results,
        completed,
        scale_factors,
        d_thresholds,
        std_thresholds,
        n_eval_samples,
        benchmarks,
    ):
        checkpoint_file = output_path / f"checkpoint_n{n_eval_samples}.json"
        with open(checkpoint_file, "w") as f:
            json.dump(
                {
                    "results": all_results,
                    "completed": completed,
                    "params": {
                        "model": self.model_path,
                        "scale_factors": scale_factors,
                        "d_thresholds": d_thresholds,
                        "std_thresholds": std_thresholds,
                        "n_eval_samples": n_eval_samples,
                        "benchmarks": benchmarks,
                    },
                },
                f,
                indent=2,
            )

    def _create_summary(self, all_results, benchmarks):
        baseline = all_results.get("baseline", {})
        top_level = [b for b in benchmarks if b in baseline]
        sub_metrics = [
            k
            for k in baseline
            if k not in top_level
            and any(k.startswith(f"{b}_") for b in top_level)
            and not k.endswith("_delta")
        ]

        lines = ["PATCHING EXPERIMENT SUMMARY", "=" * 100]

        lines.append("\nBaseline:")
        for key in top_level + sub_metrics:
            lines.append(f"  {key}: {baseline[key]:.3f}")

        for exp in all_results["experiments"]:
            train_tag = f", train={exp['train_lang']}" if exp.get("train_lang") else ""
            lines.append(
                f"\nd={exp['d_threshold']}, std={exp['std_threshold']}{train_tag} "
                f"({exp['n_neurons']} neurons, {exp['n_layers']} layers):"
            )
            lines.append("-" * 100)

            header = f"{'Scale':>6}"
            for benchmark in top_level:
                header += f" | {benchmark:>10} | Δ{benchmark:>10}"
            lines.append(header)

            for result in exp["scale_results"]:
                row = f"{result['scale_factor']:>6.3f}"
                for benchmark in top_level:
                    if benchmark in result:
                        row += f" | {result[benchmark]:>10.3f} | {result.get(f'{benchmark}_delta', 0):>+10.3f}"
                lines.append(row)

            if sub_metrics:
                lines.append("")
                lines.append("Per-language breakdown:")
                for metric in sub_metrics:
                    lines.append(f"  {metric} (baseline={baseline[metric]:.3f}):")
                    sub_header = f"    {'Scale':>6} | {'score':>8} | {'Δ':>8}"
                    lines.append(sub_header)
                    for result in exp["scale_results"]:
                        if metric not in result:
                            continue
                        score = result[metric]
                        delta = result.get(f"{metric}_delta", 0)
                        lines.append(
                            f"    {result['scale_factor']:>6.3f} | {score:>8.3f} | {delta:>+8.3f}"
                        )

        return "\n".join(lines)
