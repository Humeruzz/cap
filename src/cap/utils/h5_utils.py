from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

if TYPE_CHECKING:
    from cap.core.stats import LayerStatistics


def _layer_key(name: str) -> str:
    return name.replace(".", "_")


def _save_meta(f, dataset_info, n_samples):
    if n_samples is not None:
        f.attrs["n_samples"] = n_samples
    if dataset_info is not None:
        for key, value in dataset_info.items():
            if isinstance(value, (str, int, float, bool)):
                f.attrs[f"dataset_{key}"] = value
            elif isinstance(value, list):
                f.attrs[f"dataset_{key}"] = str(value)


class H5Store:
    @staticmethod
    def save_activations(
        *,
        activations_dict: dict[str, Any],
        texts_dict: dict[str, Any],
        prompts_dict: dict[str, Any],
        output_path: str | Path,
        model_name: str = "model",
        statistics: dict[str, LayerStatistics] | None = None,
        n_samples: int | None = None,
        dataset_info: dict[str, Any] | None = None,
    ) -> tuple[Path, Path | None]:
        h5_path = Path(output_path) / "activations.h5"

        with h5py.File(h5_path, "w") as f:
            f.attrs["model"] = model_name
            f.attrs["n_conditions"] = len(activations_dict)
            _save_meta(f, dataset_info, n_samples)

            for condition, activations in activations_dict.items():
                texts = texts_dict[condition]
                prompts = prompts_dict.get(condition, [])

                group = f.create_group(condition)
                group.attrs["n_samples"] = len(activations)

                for i, (sample_acts, sample_texts) in enumerate(
                    zip(activations, texts, strict=False)
                ):
                    sample_grp = group.create_group(f"sample_{i:03d}")
                    sample_grp.attrs["prompt"] = prompts[i] if i < len(prompts) else ""
                    sample_grp.attrs["n_steps"] = len(sample_acts)

                    text_grp = sample_grp.create_group("texts")
                    for step_idx, text in enumerate(sample_texts):
                        text_grp.attrs[f"step_{step_idx}"] = text

                    for step_idx, step_activations in enumerate(sample_acts):
                        step_grp = sample_grp.create_group(f"step_{step_idx}")
                        step_grp.attrs["step_number"] = step_idx
                        for name, values in step_activations.items():
                            data = H5Store._prepare_tensor(values)
                            step_grp.create_dataset(_layer_key(name), data=data, compression="gzip")

        stats_path: Path | None = None
        if statistics is not None:
            stats_path = H5Store.save_statistics(
                statistics=statistics,
                output_path=output_path,
                model_name=model_name,
                n_samples=n_samples,
                dataset_info=dataset_info,
            )

        return h5_path, stats_path

    @staticmethod
    def save_statistics(
        *,
        statistics: dict[str, LayerStatistics],
        output_path: str | Path,
        model_name: str = "model",
        n_samples: int | None = None,
        dataset_info: dict[str, Any] | None = None,
    ) -> Path:
        stats_path = Path(output_path) / "statistics.h5"

        with h5py.File(stats_path, "w") as f:
            f.attrs["model"] = model_name
            f.attrs["n_layers"] = len(statistics)
            _save_meta(f, dataset_info, n_samples)

            for layer_name, layer_stats in statistics.items():
                layer_grp = f.create_group(_layer_key(layer_name))
                for stat_name, values in layer_stats.items():
                    if np.isscalar(values):
                        layer_grp.create_dataset(stat_name, data=values)
                    else:
                        layer_grp.create_dataset(stat_name, data=values, compression="gzip")

        return stats_path

    @staticmethod
    def check_existing_data(output_path, *, n_samples=None, dataset_info=None):
        h5_path = Path(output_path) / "activations.h5"
        stats_path = Path(output_path) / "statistics.h5"

        if not h5_path.exists() or not stats_path.exists():
            return False, None

        with h5py.File(stats_path, "r") as f:
            stored_n_samples = f.attrs.get("n_samples")

            if n_samples is not None and stored_n_samples != n_samples:
                return False, {
                    "reason": "n_samples mismatch",
                    "stored": stored_n_samples,
                    "requested": n_samples,
                }

            if dataset_info is not None:
                for key, value in dataset_info.items():
                    stored_value = f.attrs.get(f"dataset_{key}")
                    if stored_value is None:
                        return False, {"reason": f"missing dataset_{key}"}
                    if isinstance(value, list):
                        if str(value) != stored_value:
                            return False, {
                                "reason": f"dataset_{key} mismatch",
                                "stored": stored_value,
                                "requested": str(value),
                            }
                    elif stored_value != value:
                        return False, {
                            "reason": f"dataset_{key} mismatch",
                            "stored": stored_value,
                            "requested": value,
                        }

            metadata = {
                "model": f.attrs.get("model"),
                "n_samples": stored_n_samples,
                "n_layers": f.attrs.get("n_layers"),
            }
            for key in f.attrs:
                if key.startswith("dataset_"):
                    metadata[key] = f.attrs[key]

        return True, metadata

    @staticmethod
    def load_activations(
        h5_path: str | Path,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        activations_dict: dict[str, Any] = {}
        texts_dict: dict[str, Any] = {}
        prompts_dict: dict[str, Any] = {}

        with h5py.File(h5_path, "r") as f:
            for condition in f:
                group = f[condition]
                activations_dict[condition] = []
                texts_dict[condition] = []
                prompts_dict[condition] = []

                for sample_key in sorted(group.keys()):
                    sample_grp = group[sample_key]
                    prompts_dict[condition].append(sample_grp.attrs.get("prompt", ""))

                    sample_acts = []
                    sample_texts = []

                    text_grp = sample_grp.get("texts", {})
                    for step_idx in range(sample_grp.attrs["n_steps"]):
                        text_key = f"step_{step_idx}"
                        if text_key in text_grp.attrs:
                            sample_texts.append(text_grp.attrs[text_key])

                    for step_idx in range(sample_grp.attrs["n_steps"]):
                        step_grp = sample_grp[f"step_{step_idx}"]
                        step_acts = OrderedDict()
                        for layer_name in step_grp:
                            original_name = layer_name.replace("_", ".")
                            step_acts[original_name] = step_grp[layer_name][:]
                        sample_acts.append(step_acts)

                    activations_dict[condition].append(sample_acts)
                    texts_dict[condition].append(sample_texts)

        return activations_dict, texts_dict, prompts_dict

    @staticmethod
    def load_statistics(stats_path: str | Path) -> dict[str, LayerStatistics] | None:
        stats_path = Path(stats_path)
        if not stats_path.exists():
            return None

        statistics: dict[str, Any] = {}

        with h5py.File(stats_path, "r") as f:
            for layer_name in f:
                original_name = layer_name.replace("_", ".")
                statistics[original_name] = {}
                layer_grp = f[layer_name]
                for stat_name in layer_grp:
                    dataset = layer_grp[stat_name]
                    if dataset.shape == ():
                        statistics[original_name][stat_name] = dataset[()]
                    else:
                        statistics[original_name][stat_name] = dataset[:]

        return statistics

    @staticmethod
    def _prepare_tensor(values):
        if hasattr(values, "shape"):
            return values.flatten()
        return np.array(values).flatten()
