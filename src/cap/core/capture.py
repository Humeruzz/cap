from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cap.utils.reproducibility import set_seed


class ActivationCapture:
    def __init__(
        self,
        model_path: str,
        device: str | None = None,
        seed: int = 42,
        trust_remote_code: bool = False,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"ActivationCapture init: using device={device}", flush=True)
        self.seed = seed
        set_seed(seed)
        # transformers stub overloads from_pretrained on the trust_remote_code kwarg -> spurious arg-type
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        ).to(device)  # type: ignore[arg-type]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        self.device = device
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left padding keeps the final position aligned with the final real token
        # across prompts in a padded batch, which preserves last-token capture logic.
        self.tokenizer.padding_side = "left"

    def _normalize_layer_patterns(
        self, layers_to_capture: str | list[str] | None
    ) -> list[str] | None:
        if layers_to_capture is None:
            return None
        if isinstance(layers_to_capture, str):
            return [layers_to_capture]
        return list(layers_to_capture)

    def _extract_last_token_activation(self, tensor: Any) -> torch.Tensor | None:
        if not isinstance(tensor, torch.Tensor):
            return None
        if tensor.ndim >= 3:
            return tensor[:, -1, ...]
        if tensor.ndim >= 2:
            return tensor
        if tensor.ndim == 1:
            return tensor.unsqueeze(0)
        return tensor

    def _register_capture_hooks(self, step_activations, layers_to_capture):
        layer_patterns = self._normalize_layer_patterns(layers_to_capture)
        hooks = []

        def make_hook(name):
            def hook(module, inputs, output):
                candidate = output
                if isinstance(output, (tuple, list)) and len(output) > 0:
                    candidate = output[0]
                activation = self._extract_last_token_activation(candidate)
                if activation is not None:
                    step_activations[name] = activation.detach().float().cpu()

            return hook

        for name, module in self.model.named_modules():
            if layer_patterns is not None and not any(p in name for p in layer_patterns):
                continue
            if next(module.children(), None) is not None:
                continue
            hooks.append(module.register_forward_hook(make_hook(name)))

        return hooks

    def _decode_trimmed(self, token_row, pad_len, *, skip_special_tokens):
        return self.tokenizer.decode(token_row[pad_len:], skip_special_tokens=skip_special_tokens)

    def _split_batched_activations(self, generation_activations, batch_size):
        per_prompt_activations = [[] for _ in range(batch_size)]

        for step_activation_dict in generation_activations:
            for sample_idx in range(batch_size):
                sample_step = OrderedDict()
                for name, act in step_activation_dict.items():
                    if (
                        isinstance(act, torch.Tensor)
                        and act.ndim >= 1
                        and act.shape[0] == batch_size
                    ):
                        sample_step[name] = act[sample_idx : sample_idx + 1].clone()
                    elif isinstance(act, torch.Tensor):
                        sample_step[name] = act.clone()
                    else:
                        sample_step[name] = act
                per_prompt_activations[sample_idx].append(sample_step)

        return per_prompt_activations

    def capture_generation_batch(
        self,
        input_texts,
        *,
        max_new_tokens=10,
        layers_to_capture=None,
    ):
        prompts = list(input_texts)
        if len(prompts) == 0:
            return [], []
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")

        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = encoded.input_ids.to(self.device)
        attention_mask = encoded.attention_mask.to(self.device)

        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        prompt_lengths = attention_mask.sum(dim=1).tolist()
        pad_lengths = [seq_len - int(length) for length in prompt_lengths]

        generation_activations = []
        generated_tokens = input_ids.clone()
        generated_attention_mask = attention_mask.clone()
        step_texts = [
            [self._decode_trimmed(input_ids[i], pad_lengths[i], skip_special_tokens=True)]
            for i in range(batch_size)
        ]

        step_activations = OrderedDict()
        hooks = self._register_capture_hooks(step_activations, layers_to_capture)

        print(f"    Generating {max_new_tokens} tokens (batch={batch_size})...", flush=True)

        try:
            for _step in range(max_new_tokens):
                step_activations.clear()

                with torch.inference_mode():
                    outputs = self.model(
                        input_ids=generated_tokens, attention_mask=generated_attention_mask
                    )

                next_token_ids = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                generated_tokens = torch.cat([generated_tokens, next_token_ids], dim=1)
                next_attention = torch.ones(
                    (batch_size, 1),
                    dtype=generated_attention_mask.dtype,
                    device=generated_attention_mask.device,
                )
                generated_attention_mask = torch.cat(
                    [generated_attention_mask, next_attention], dim=1
                )

                for i in range(batch_size):
                    step_texts[i].append(
                        self._decode_trimmed(
                            generated_tokens[i], pad_lengths[i], skip_special_tokens=False
                        )
                    )

                generation_activations.append(OrderedDict(step_activations))

            print(
                f"    Done [{max_new_tokens}/{max_new_tokens}] seq_len={generated_tokens.shape[1]}",
                flush=True,
            )
        finally:
            for hook in hooks:
                hook.remove()

        return self._split_batched_activations(generation_activations, batch_size), step_texts

    def capture_generation(self, input_text, *, max_new_tokens=10, layers_to_capture=None):
        batch_activations, batch_texts = self.capture_generation_batch(
            [input_text],
            max_new_tokens=max_new_tokens,
            layers_to_capture=layers_to_capture,
        )
        return batch_activations[0], batch_texts[0]

    def capture_prompts(self, prompts, *, max_new_tokens=1, layers_to_capture=None, batch_size=1):
        if batch_size <= 0:
            raise ValueError("batch_size must be >= 1")

        prompts = list(prompts)
        all_activations = []
        all_texts = []

        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            end = start + len(batch_prompts)
            print(f"  Processing prompts {start + 1}-{end}/{len(prompts)}", flush=True)

            batch_activations, batch_texts = self.capture_generation_batch(
                batch_prompts,
                max_new_tokens=max_new_tokens,
                layers_to_capture=layers_to_capture,
            )

            all_activations.extend(batch_activations)
            all_texts.extend(batch_texts)

        return all_activations, all_texts
