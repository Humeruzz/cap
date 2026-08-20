import math
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def group_by_layer(step_activations):
    layer_groups = {}

    for name, activation in step_activations.items():
        if ".layers." in name:
            parts = name.split(".")
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        if layer_idx not in layer_groups:
                            layer_groups[layer_idx] = {}
                        layer_groups[layer_idx][name] = activation
                        break
                    except ValueError:
                        continue
        else:
            key = name
            if key not in layer_groups:
                layer_groups[key] = {}
            layer_groups[key][name] = activation

    return layer_groups


def normalize(values):
    p1, p99 = np.percentile(values, [1, 99])
    clipped = np.clip(values, p1, p99)
    return (clipped - p1) / (p99 - p1 + 1e-8)


def combine_layer_activations(layer_dict):
    all_values = []
    for name in sorted(layer_dict.keys()):
        values = layer_dict[name].flatten()
        all_values.extend(values)
    return np.array(all_values)


def to_256x256(values):
    target_size = 256 * 256
    if len(values) < target_size:
        padded = np.zeros(target_size)
        padded[: len(values)] = values
        values = padded
    else:
        values = values[:target_size]
    return values.reshape(256, 256)


def apply_colormap(image):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("inferno")
    colored = cmap(image)
    return (colored[:, :, :3] * 255).astype(np.uint8)


def create_grid(layer_images):
    n_layers = len(layer_images)
    grid_cols = math.ceil(math.sqrt(n_layers))
    grid_rows = math.ceil(n_layers / grid_cols)

    height = grid_rows * 256
    width = grid_cols * 256

    grid = np.zeros((height, width, 3), dtype=np.uint8)

    for idx, (_, img) in enumerate(layer_images):
        row = idx // grid_cols
        col = idx % grid_cols

        y_start = row * 256
        y_end = y_start + 256
        x_start = col * 256
        x_end = x_start + 256

        colored = apply_colormap(img)
        grid[y_start:y_end, x_start:x_end] = colored

    return grid


def get_font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("/System/Library/Fonts/Arial.ttf", size)
        except Exception:
            return ImageFont.load_default()


def add_text_overlay(image, step_num, current_text, prev_text=None):
    img_array = np.array(image)
    overlay_height = 200
    new_height = img_array.shape[0] + overlay_height
    new_img = np.ones((new_height, img_array.shape[1], 3), dtype=np.uint8) * 30
    new_img[: img_array.shape[0]] = img_array
    result = Image.fromarray(new_img)

    draw = ImageDraw.Draw(result)

    font_large = get_font(28)
    font_text = get_font(18)

    y_start = img_array.shape[0] + 20
    draw.text((20, y_start), f"STEP {step_num}", fill=(255, 255, 255), font=font_large)

    text_y = y_start + 50

    if current_text:
        if prev_text and current_text.startswith(prev_text):
            new_part = current_text[len(prev_text) :]

            wrapped_old = textwrap.fill(prev_text, width=100)
            for line in wrapped_old.split("\n")[-2:]:
                if text_y < result.height - 25:
                    draw.text((20, text_y), line, fill=(150, 150, 150), font=font_text)
                    text_y += 22

            if new_part.strip():
                draw.text(
                    (20, text_y), f"+ {new_part.strip()}", fill=(100, 255, 100), font=font_text
                )
        else:
            wrapped = textwrap.fill(current_text, width=100).split("\n")
            for line in wrapped[:4]:
                if text_y < result.height - 25:
                    draw.text((20, text_y), line, fill=(255, 255, 255), font=font_text)
                    text_y += 22

    return result


def visualize_step(step_activations, step_num, current_text=None, prev_text=None):
    layer_groups = group_by_layer(step_activations)

    int_keys = [k for k in layer_groups if isinstance(k, int)]
    str_keys = [k for k in layer_groups if isinstance(k, str)]

    input_layers = []
    output_layers = []
    other_layers = []

    for k in str_keys:
        k_lower = k.lower().replace("_", "").replace(".", "")
        k_orig = k
        if "embed" in k_lower or "rotary" in k_lower:
            input_layers.append(k_orig)
        elif ("lm" in k_lower and "head" in k_lower) or "norm" in k_lower:
            output_layers.append(k_orig)
        else:
            other_layers.append(k_orig)

    input_layers.sort()
    output_layers.sort()
    other_layers.sort()

    all_keys = input_layers + sorted(int_keys) + other_layers + output_layers

    layer_images = []
    for layer_id in all_keys:
        layer_dict = layer_groups[layer_id]
        combined = combine_layer_activations(layer_dict)
        normalized = normalize(combined)

        img_array = to_256x256(normalized)
        layer_images.append((str(layer_id), img_array))

    grid = create_grid(layer_images)
    base_image = Image.fromarray(grid, mode="RGB")
    final_image = add_text_overlay(base_image, step_num, current_text, prev_text)

    return final_image


def process_all_steps(activations, step_texts, output_dir="output"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for step_idx, step_acts in enumerate(activations):
        current_text = step_texts[step_idx + 1] if step_idx + 1 < len(step_texts) else None
        prev_text = step_texts[step_idx] if step_idx < len(step_texts) else None

        img = visualize_step(step_acts, step_idx + 1, current_text, prev_text)
        filepath = output_path / f"step_{step_idx:03d}.png"
        img.save(filepath)
        print(f"  Step {step_idx + 1}/{len(activations)}")

    return output_path
