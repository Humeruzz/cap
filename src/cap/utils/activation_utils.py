from collections import OrderedDict

import numpy as np


def flatten_activations_by_layer(*, activation_list):
    layer_activations = OrderedDict()

    for sample_acts in activation_list:
        for step_acts in sample_acts:
            for name, act in step_acts.items():
                if name not in layer_activations:
                    layer_activations[name] = []

                if act.shape[0] != 1:
                    raise ValueError(
                        f"Expected batch size 1, got {act.shape[0]}. "
                        "Activation capture must run one sample at a time."
                    )

                layer_activations[name].append(act[0].flatten())

    return layer_activations


def extract_layer_activations(activation_dict, layer_pattern):
    layer_acts = {}
    for name, act in activation_dict.items():
        if layer_pattern in name:
            layer_acts[name] = act
    return layer_acts


def get_activation_at_position(activation, position=-1):
    if len(activation.shape) == 3:
        return activation[:, position, :]
    elif len(activation.shape) == 2:
        return activation[position, :]
    else:
        return activation.flatten()


def compute_average_activations(*, activation_list):
    avg_activations = OrderedDict()

    all_keys = set()
    for sample_acts in activation_list:
        for step_acts in sample_acts:
            all_keys.update(step_acts.keys())

    for key in sorted(all_keys):
        all_values = []
        for sample_acts in activation_list:
            for step_acts in sample_acts:
                if key in step_acts:
                    act = step_acts[key]
                    if hasattr(act, "shape"):
                        if len(act.shape) == 3:
                            act = act[:, -1, :]
                        elif len(act.shape) == 2:
                            act = act[-1, :]
                        else:
                            raise ValueError(f"Unexpected activation shape: {act.shape}")
                        all_values.append(act.flatten())
                    else:
                        all_values.append(np.array(act).flatten())

        if all_values:
            min_len = min(len(v) for v in all_values)
            all_values = [v[:min_len] for v in all_values]
            avg_activations[key] = np.stack(all_values).mean(axis=0)

    return avg_activations
