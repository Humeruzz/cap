from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cap.core.stats import LayerStatistics
from cap.utils.h5_utils import H5Store


class ActivationPatcher:
    def __init__(
        self, *, model_path: str, device: str = "cuda", trust_remote_code: bool = False
    ) -> None:
        # transformers stub overloads from_pretrained on the trust_remote_code kwarg -> spurious arg-type
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        ).to(device)  # type: ignore[arg-type]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        self.tokenizer.padding_side = "left"
        self.device = device
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.hooks: list[Any] = []
        self.patch_specs: dict[str, tuple[list[int], float]] = {}
        self.layer_stats: dict[str, LayerStatistics] | None = None

    def load_statistics(self, *, h5_path: str | Path) -> dict[str, LayerStatistics]:
        h5 = Path(h5_path)
        stats_path = (h5 if h5.is_dir() else h5.parent) / "statistics.h5"
        self.layer_stats = H5Store.load_statistics(stats_path)
        if self.layer_stats is None:
            raise ValueError(f"No statistics found at {stats_path}. Run comparison first.")
        print(f"Loaded statistics for {len(self.layer_stats)} layers from {stats_path}")
        return self.layer_stats

    def identify_significant_neurons(
        self, *, d_threshold: float = 0.0, std_threshold: float = 0.0
    ) -> tuple[dict[str, list[int]], list[dict[str, Any]]]:
        if self.layer_stats is None:
            raise ValueError("Must call load_statistics first")

        patch_targets: dict[str, list[int]] = {}
        significant_info: list[dict[str, Any]] = []

        for layer_name, stats_dict in self.layer_stats.items():
            cohens_d = stats_dict["cohens_d"]
            pooled_std = stats_dict["pooled_std"]
            mean_diff = stats_dict["mean_diff"]

            significant_mask = (np.abs(cohens_d) > d_threshold) & (pooled_std > std_threshold)
            significant_indices = np.where(significant_mask)[0]

            if len(significant_indices) > 0:
                patch_targets[layer_name] = significant_indices.tolist()
                for idx in significant_indices:
                    significant_info.append(
                        {
                            "layer": layer_name,
                            "neuron": int(idx),
                            "cohens_d": float(cohens_d[idx]),
                            "mean_diff": float(mean_diff[idx]),
                            "pooled_std": float(pooled_std[idx]),
                        }
                    )

        significant_info = sorted(significant_info, key=lambda x: abs(x["cohens_d"]), reverse=True)
        return patch_targets, significant_info

    def setup_patches(
        self, *, patch_targets: dict[str, list[int]], scale_factor: float = 0.0
    ) -> None:
        """Patch every layer in patch_targets with a uniform scale_factor."""
        self.setup_patches_custom(
            patch_dict={layer: (neurons, scale_factor) for layer, neurons in patch_targets.items()}
        )

    def setup_patches_custom(self, *, patch_dict: dict[str, tuple[list[int], float]]) -> None:
        """Register scaling hooks. patch_dict: {layer_name: (neuron_indices, scale_factor)}."""
        self.clear_patches()
        self.patch_specs = patch_dict

        def make_hook(neuron_indices, scale):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    out, rest = output[0], output[1:]
                elif isinstance(output, torch.Tensor):
                    out, rest = output, None
                else:
                    return output

                original_shape = out.shape
                if out.ndim == 4:
                    b, s, h, d = out.shape
                    out = out.view(b, s, h * d)
                elif out.ndim == 2:
                    out = out.unsqueeze(1)

                out[:, :, neuron_indices] *= scale

                if len(original_shape) == 4:
                    out = out.view(original_shape)
                elif len(original_shape) == 2:
                    out = out.squeeze(1)

                return out if rest is None else (out, *rest)

            return hook

        for name, module in self.model.named_modules():
            matched_key = (
                name
                if name in self.patch_specs
                else (
                    name.replace("_", ".") if name.replace("_", ".") in self.patch_specs else None
                )
            )
            if matched_key:
                neurons, scale = self.patch_specs[matched_key]
                self.hooks.append(module.register_forward_hook(make_hook(neurons, scale)))

    def clear_patches(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.patch_specs = {}
