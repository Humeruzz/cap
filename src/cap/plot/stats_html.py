import math
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


def normalize(values, vmin=None, vmax=None):
    if vmin is None or vmax is None:
        p1, p99 = np.percentile(values, [1, 99])
        vmin, vmax = p1, p99
    clipped = np.clip(values, vmin, vmax)
    return (clipped - vmin) / (vmax - vmin + 1e-8)


def combine_layer_activations(layer_dict):
    all_values = []
    index_map = []
    for name in sorted(layer_dict.keys()):
        values = layer_dict[name].flatten()
        all_values.extend(values)
        for i in range(len(values)):
            index_map.append((name, i))
    return np.array(all_values), index_map


def create_activation_map(index_map, target_resolution):
    if len(index_map) == 0:
        return [[None] * target_resolution for _ in range(target_resolution)]

    total_elements = len(index_map)
    current_size = int(np.sqrt(total_elements))

    if current_size * current_size != total_elements:
        source_resolution = 256
        padded_map = index_map + [(None, -1)] * (256 * 256 - total_elements)
    else:
        source_resolution = current_size
        padded_map = index_map

    map_2d = []
    for i in range(source_resolution):
        row = []
        for j in range(source_resolution):
            idx = i * source_resolution + j
            row.append(
                padded_map[idx][0]
                if idx < len(padded_map) and padded_map[idx][0] is not None
                else None
            )
        map_2d.append(row)

    if source_resolution == target_resolution:
        return map_2d

    downsampled_map = []
    scale = source_resolution / target_resolution

    for i in range(target_resolution):
        row = []
        for j in range(target_resolution):
            src_i_start = int(i * scale)
            src_i_end = int((i + 1) * scale)
            src_j_start = int(j * scale)
            src_j_end = int((j + 1) * scale)

            names_in_region = []
            for si in range(src_i_start, min(src_i_end, source_resolution)):
                for sj in range(src_j_start, min(src_j_end, source_resolution)):
                    name = map_2d[si][sj]
                    if name is not None:
                        names_in_region.append(name)

            if names_in_region:
                most_common = max(set(names_in_region), key=names_in_region.count)
                row.append(most_common)
            else:
                row.append(None)
        downsampled_map.append(row)

    return downsampled_map


def to_resolution(values, resolution):
    target_size = resolution * resolution
    if len(values) < target_size:
        padded = np.zeros(target_size)
        padded[: len(values)] = values
        values = padded
    else:
        values = values[:target_size]
    return values.reshape(resolution, resolution)


def downsample_array(arr, target_resolution):
    if len(arr) == 0:
        return np.zeros((target_resolution, target_resolution))

    current_size = int(np.sqrt(len(arr)))
    if current_size * current_size != len(arr):
        arr_2d = to_resolution(arr, 256)
        current_size = 256
    else:
        arr_2d = arr.reshape(current_size, current_size)

    if current_size == target_resolution:
        return arr_2d

    from scipy.ndimage import zoom

    scale = target_resolution / current_size
    return zoom(arr_2d, scale, order=1)


def apply_colormap(image, cmap_name="inferno"):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(cmap_name)
    colored = cmap(image)
    return (colored[:, :, :3] * 255).astype(np.uint8)


def create_grid(layer_images):
    n_layers = len(layer_images)
    grid_cols = math.ceil(math.sqrt(n_layers))
    grid_rows = math.ceil(n_layers / grid_cols)

    height = grid_rows * 256
    width = grid_cols * 256

    grid = np.zeros((height, width, 3), dtype=np.uint8)

    for idx, (_, img, cmap) in enumerate(layer_images):
        row = idx // grid_cols
        col = idx % grid_cols

        y_start = row * 256
        y_end = y_start + 256
        x_start = col * 256
        x_end = x_start + 256

        colored = apply_colormap(img, cmap)
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


def add_text_overlay(image, title, stats_info=None):
    img_array = np.array(image)
    overlay_height = 150 if stats_info else 100
    new_height = img_array.shape[0] + overlay_height
    new_img = np.ones((new_height, img_array.shape[1], 3), dtype=np.uint8) * 30
    new_img[: img_array.shape[0]] = img_array
    result = Image.fromarray(new_img)

    draw = ImageDraw.Draw(result)
    font_large = get_font(28)
    font_small = get_font(16)

    y_start = img_array.shape[0] + 20
    draw.text((20, y_start), title, fill=(255, 255, 255), font=font_large)

    if stats_info:
        y_stats = y_start + 50
        draw.text((20, y_stats), stats_info, fill=(200, 200, 200), font=font_small)

    return result


def visualize_stats(*, stats_results, metric="t_stats", p_threshold=0.05, title=None):
    layer_groups_mean = group_by_layer(stats_results["mean_diff"])
    layer_groups_stat = group_by_layer(stats_results[metric])
    layer_groups_pval = group_by_layer(stats_results["p_values"])

    int_keys = [k for k in layer_groups_mean if isinstance(k, int)]
    str_keys = [k for k in layer_groups_mean if isinstance(k, str)]

    input_layers = []
    output_layers = []
    other_layers = []

    for k in str_keys:
        k_lower = k.lower().replace("_", "").replace(".", "")
        if "embed" in k_lower or "rotary" in k_lower:
            input_layers.append(k)
        elif ("lm" in k_lower and "head" in k_lower) or "norm" in k_lower:
            output_layers.append(k)
        else:
            other_layers.append(k)

    input_layers.sort()
    output_layers.sort()
    other_layers.sort()

    all_keys = input_layers + sorted(int_keys) + other_layers + output_layers

    layer_images = []
    n_significant_total = 0
    n_total = 0

    for layer_id in all_keys:
        stat_dict = layer_groups_stat[layer_id]
        pval_dict = layer_groups_pval[layer_id]

        combined_stat, _ = combine_layer_activations(stat_dict)
        combined_pval, _ = combine_layer_activations(pval_dict)

        significant_mask = combined_pval < p_threshold
        n_significant_total += np.sum(significant_mask)
        n_total += len(combined_pval)

        masked_stat = np.where(significant_mask, combined_stat, 0)

        normalized = normalize(masked_stat, vmin=-5, vmax=5)
        img_array = to_resolution(normalized, 256)

        layer_images.append((str(layer_id), img_array, "RdBu_r"))

    grid = create_grid(layer_images)
    base_image = Image.fromarray(grid, mode="RGB")

    if title is None:
        title = f"Statistical Map ({metric})"

    pct_sig = 100 * n_significant_total / n_total if n_total > 0 else 0
    stats_info = f"p < {p_threshold}: {n_significant_total:,}/{n_total:,} dims ({pct_sig:.2f}%)"

    final_image = add_text_overlay(base_image, title, stats_info)

    return final_image, {
        "n_significant": n_significant_total,
        "n_total": n_total,
        "pct_significant": pct_sig,
    }


def create_interactive_viewer(
    *,
    stats_results,
    output_path,
    downsample_resolution=64,
    downsample_resolutions=None,
    presets=None,
):
    import json

    if downsample_resolutions is None:
        downsample_resolutions = [downsample_resolution]

    if presets is None:
        presets = {
            "Show All": {
                "p": 1.001,
                "t": 0.0,
                "cohens": 0.0,
                "mean": 0.0,
                "pooled": 0.0,
                "range": 5.0,
            },
            "Moderate": {
                "p": 0.05,
                "t": 2.0,
                "cohens": 0.2,
                "mean": 0.0,
                "pooled": 0.0,
                "range": 5.0,
            },
            "Strict": {
                "p": 0.01,
                "t": 3.0,
                "cohens": 0.5,
                "mean": 0.0,
                "pooled": 0.0,
                "range": 5.0,
            },
        }

    layer_groups_stat = group_by_layer(stats_results["t_stats"])
    layer_groups_pval = group_by_layer(stats_results["p_values"])
    layer_groups_cohens = group_by_layer(stats_results["cohens_d"])
    layer_groups_mean = group_by_layer(stats_results["mean_diff"])
    layer_groups_pooled = group_by_layer(stats_results["pooled_std"])

    int_keys = [k for k in layer_groups_stat if isinstance(k, int)]
    str_keys = [k for k in layer_groups_stat if isinstance(k, str)]

    input_layers = []
    output_layers = []
    other_layers = []

    for k in str_keys:
        k_lower = k.lower().replace("_", "").replace(".", "")
        if "embed" in k_lower or "rotary" in k_lower:
            input_layers.append(k)
        elif ("lm" in k_lower and "head" in k_lower) or "norm" in k_lower:
            output_layers.append(k)
        else:
            other_layers.append(k)

    input_layers.sort()
    output_layers.sort()
    other_layers.sort()

    all_keys = input_layers + sorted(int_keys) + other_layers + output_layers

    # Combine activations once; downsample for each requested resolution
    combined_data = []
    for layer_id in all_keys:
        combined_stat, index_map = combine_layer_activations(layer_groups_stat[layer_id])
        combined_pval, _ = combine_layer_activations(layer_groups_pval[layer_id])
        combined_cohens, _ = combine_layer_activations(layer_groups_cohens[layer_id])
        combined_mean, _ = combine_layer_activations(layer_groups_mean[layer_id])
        combined_pooled, _ = combine_layer_activations(layer_groups_pooled[layer_id])
        combined_data.append(
            (
                str(layer_id),
                combined_stat,
                combined_pval,
                combined_cohens,
                combined_mean,
                combined_pooled,
                index_map,
            )
        )

    all_layers_data = {}
    for res in downsample_resolutions:
        layers_data = []
        for (
            layer_name,
            combined_stat,
            combined_pval,
            combined_cohens,
            combined_mean,
            combined_pooled,
            index_map,
        ) in combined_data:
            layers_data.append(
                {
                    "name": layer_name,
                    "stats": downsample_array(combined_stat, res).tolist(),
                    "pvals": downsample_array(combined_pval, res).tolist(),
                    "cohens": downsample_array(combined_cohens, res).tolist(),
                    "mean_diff": downsample_array(combined_mean, res).tolist(),
                    "pooled_std": downsample_array(combined_pooled, res).tolist(),
                    "activation_map": create_activation_map(index_map, res),
                }
            )
        all_layers_data[res] = layers_data

    print(
        f"Creating interactive HTML with {len(combined_data)} layers, resolutions: {downsample_resolutions}..."
    )

    html_path = Path(output_path) / "interactive_stats.html"

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Interactive Statistical Map</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1a1a1a;
            color: #fff;
        }}
        .controls {{
            position: sticky;
            top: 0;
            z-index: 10;
            background: #2a2a2a;
            padding: 8px 12px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .controls-row1 {{
            display: flex;
            align-items: center;
            gap: 16px;
            flex-wrap: wrap;
            margin-bottom: 6px;
            padding-bottom: 6px;
            border-bottom: 1px solid #444;
        }}
        .controls-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1px 16px;
        }}
        .control-group {{
            margin: 3px 0;
            white-space: nowrap;
        }}
        label {{
            display: inline-block;
            width: 120px;
            font-weight: bold;
            font-size: 12px;
        }}
        input[type="range"] {{
            width: 160px;
            vertical-align: middle;
        }}
        .value-input {{
            display: inline-block;
            width: 66px;
            text-align: right;
            font-family: monospace;
            background: #1a1a1a;
            color: #fff;
            border: 1px solid #444;
            border-radius: 4px;
            padding: 2px 4px;
            font-size: 12px;
            -moz-appearance: textfield;
        }}
        .value-input::-webkit-outer-spin-button,
        .value-input::-webkit-inner-spin-button {{
            -webkit-appearance: none;
            margin: 0;
        }}
        .value-input:focus {{
            outline: none;
            border-color: #7a9a7a;
            background: #2a3a2a;
        }}
        .presets-label, .res-label {{
            margin-right: 8px;
            font-weight: bold;
            font-size: 12px;
            white-space: nowrap;
        }}
        .preset-btn {{
            background: #3a3a3a;
            color: #fff;
            border: 1px solid #555;
            border-radius: 4px;
            padding: 4px 10px;
            margin-right: 4px;
            cursor: pointer;
            font-size: 12px;
        }}
        .preset-btn:hover {{
            background: #4a4a4a;
            border-color: #7a9a7a;
        }}
        .preset-btn.active {{
            background: #3a5a3a;
            border-color: #5a8a5a;
        }}
        .res-btn {{
            background: #3a3a3a;
            color: #fff;
            border: 1px solid #555;
            border-radius: 4px;
            padding: 4px 10px;
            margin-right: 4px;
            cursor: pointer;
            font-size: 12px;
        }}
        .res-btn:hover {{
            background: #4a4a4a;
            border-color: #9a7a7a;
        }}
        .res-btn.active {{
            background: #5a3a3a;
            border-color: #8a5a5a;
        }}
        .metric-btn {{
            background: #3a3a3a;
            color: #fff;
            border: 1px solid #555;
            border-radius: 4px;
            padding: 4px 10px;
            margin-right: 4px;
            cursor: pointer;
            font-size: 12px;
        }}
        .metric-btn:hover {{
            background: #4a4a4a;
            border-color: #7a7a9a;
        }}
        .metric-btn.active {{
            background: #3a3a5a;
            border-color: #5a5a8a;
        }}
        .colormap-legend {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 50;
            background: rgba(26,26,26,0.97);
            border: 1px solid #444;
            border-radius: 8px;
            padding: 10px 12px;
            min-width: 220px;
            box-shadow: 0 2px 14px rgba(0,0,0,0.6);
        }}
        .colormap-legend-title {{
            font-size: 11px;
            font-weight: bold;
            color: #aaa;
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}
        .cmap-btn {{
            display: flex;
            align-items: center;
            gap: 6px;
            background: none;
            border: 1px solid transparent;
            border-radius: 4px;
            padding: 3px 6px;
            cursor: pointer;
            color: #ccc;
            font-size: 11px;
            width: 100%;
            margin-bottom: 3px;
            box-sizing: border-box;
        }}
        .cmap-btn:hover {{ border-color: #666; background: #2a2a2a; }}
        .cmap-btn.active {{ border-color: #7a9a7a; background: #2a3a2a; color: #fff; }}
        .colorbar-wrap {{
            margin-top: 6px;
        }}
        .colorbar-labels {{
            display: flex;
            justify-content: space-between;
            font-family: monospace;
            font-size: 10px;
            color: #888;
            margin-top: 2px;
        }}
        .stats {{
            background: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .click-info {{
            background: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            min-height: 80px;
        }}
        .click-info.active {{
            background: #3a4a3a;
            border: 2px solid #5a7a5a;
        }}
        .info-row {{
            margin: 8px 0;
            font-family: monospace;
        }}
        .info-label {{
            display: inline-block;
            width: 140px;
            color: #aaa;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(256px, 1fr));
            gap: 20px;
        }}
        .layer {{
            background: #2a2a2a;
            padding: 10px;
            border-radius: 8px;
        }}
        .layer-title {{
            text-align: center;
            margin-bottom: 6px;
            font-weight: bold;
            cursor: help;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .layer-count {{
            text-align: center;
            margin-top: 6px;
            font-family: monospace;
            font-size: 11px;
            color: #888;
        }}
        canvas {{
            display: block;
            width: 100%;
            height: auto;
            image-rendering: pixelated;
            cursor: crosshair;
        }}
    </style>
</head>
<body>
    <h1>Interactive Statistical Significance Map</h1>

    <div class="controls">
        <div class="controls-row1">
            <div id="presetsBar"><span class="presets-label">Presets:</span></div>
            <div style="border-left:1px solid #444;padding-left:16px">
                <span class="presets-label">Colour by:</span>
                <div id="metricBar" style="display:inline-block"></div>
            </div>
            <div id="resSwitcher" style="display:none;border-left:1px solid #444;padding-left:16px">
                <span class="res-label">Resolution:</span>
            </div>
        </div>
        <div class="controls-grid">
            <div class="control-group">
                <label>P-value:</label>
                <input type="range" id="pThreshold" min="0.001" max="1.001" step="0.001" value="1.001">
                <input type="number" class="value-input" id="pValue" value="1.001" min="0.001" max="1.001" step="0.001">
            </div>
            <div class="control-group">
                <label>|T-stat|:</label>
                <input type="range" id="tThreshold" min="0" max="100" step="0.1" value="0">
                <input type="number" class="value-input" id="tValue" value="0.0" min="0" max="100" step="0.1">
            </div>
            <div class="control-group">
                <label>|Cohen's d|:</label>
                <input type="range" id="cohensThreshold" min="0" max="10" step="0.01" value="0">
                <input type="number" class="value-input" id="cohensValue" value="0.00" min="0" max="10" step="0.01">
            </div>
            <div class="control-group">
                <label>|Mean diff|:</label>
                <input type="range" id="meanThreshold" min="0" max="10" step="0.01" value="0">
                <input type="number" class="value-input" id="meanValue" value="0.00" min="0" max="10" step="0.01">
            </div>
            <div class="control-group">
                <label>Pooled std:</label>
                <input type="range" id="pooledThreshold" min="0" max="10" step="0.01" value="0">
                <input type="number" class="value-input" id="pooledValue" value="0.00" min="0" max="10" step="0.01">
            </div>
            <div class="control-group">
                <label>Colormap range:</label>
                <input type="range" id="colorRange" min="1" max="100" step="1" value="5">
                <span style="font-family:monospace;margin-right:2px">±</span><input type="number" class="value-input" id="rangeValue" value="5.0" min="1" max="100" step="1">
            </div>
        </div>
    </div>

    <div class="colormap-legend" id="colormapLegend">
        <div class="colormap-legend-title">Colormap</div>
        <div id="cmapBtnList"></div>
        <div class="colorbar-wrap">
            <canvas id="colorbar" width="196" height="16" style="width:100%;border-radius:3px;display:block"></canvas>
            <div class="colorbar-labels">
                <span id="cbLabelMin"></span>
                <span>0</span>
                <span id="cbLabelMax"></span>
            </div>
        </div>
    </div>
    
    <div class="stats">
        <strong>Surviving dimensions:</strong> <span id="sigCount">-</span> / <span id="totalCount">-</span> (<span id="sigPercent">-</span>%)
    </div>
    
    <div class="click-info" id="clickInfo">
        <div style="color: #888; text-align: center;">Click on any pixel to see activation details</div>
    </div>
    
    <div class="grid" id="layerGrid"></div>
    
    <script>
        const allLayersData = {json.dumps(all_layers_data)};
        const resolutions   = {json.dumps(downsample_resolutions)};
        let currentResolution = resolutions[0];
        let colorMetric = 'stats';

        const METRIC_OPTIONS = [
            {{ key: 'stats',     label: 'T-stat'    }},
            {{ key: 'cohens',    label: "Cohen's d" }},
            {{ key: 'mean_diff', label: 'Mean diff' }},
        ];

        function lerpColors(t, stops) {{
            for (let i = 0; i < stops.length - 1; i++) {{
                if (t <= stops[i + 1][0]) {{
                    const f = (t - stops[i][0]) / (stops[i + 1][0] - stops[i][0]);
                    return stops[i][1].map((c, j) => Math.round(c + f * (stops[i + 1][1][j] - c)));
                }}
            }}
            return stops[stops.length - 1][1];
        }}

        const VIRIDIS_STOPS = [[0,[68,1,84]],[0.25,[59,82,139]],[0.5,[33,145,140]],[0.75,[94,201,98]],[1,[253,231,37]]];
        const INFERNO_STOPS = [[0,[0,0,4]],[0.25,[87,16,110]],[0.5,[188,55,84]],[0.75,[251,181,9]],[1,[252,255,164]]];

        const COLORMAPS = {{
            'RdBu':    {{ label: 'RdBu',    fn: (t) => {{
                const r = Math.min(255, Math.max(0,
                    t < 0.5 ? 255 * (1 - 2 * t) : 255 * (2 * t - 1)));
                const b = Math.min(255, Math.max(0,
                    t < 0.5 ? 255 * (2 * t) : 255 * (2 - 2 * t)));
                const g = Math.min(255, Math.max(0, 255 * (1 - Math.abs(2 * t - 1))));
                return [r, g, b];
            }} }},
            'Seismic': {{ label: 'Seismic', fn: (t) => {{
                if (t < 0.5) {{
                    const s = t * 2;
                    return [Math.round(s * 255), Math.round(s * 255), 255];
                }} else {{
                    const s = (t - 0.5) * 2;
                    return [255, Math.round((1 - s) * 255), Math.round((1 - s) * 255)];
                }}
            }} }},
            'Hot':     {{ label: 'Hot',     fn: (t) => [
                Math.round(Math.min(1, t * 3) * 255),
                Math.round(Math.min(1, Math.max(0, t * 3 - 1)) * 255),
                Math.round(Math.min(1, Math.max(0, t * 3 - 2)) * 255),
            ] }},
            'Viridis': {{ label: 'Viridis', fn: (t) => lerpColors(t, VIRIDIS_STOPS) }},
            'Inferno': {{ label: 'Inferno', fn: (t) => lerpColors(t, INFERNO_STOPS) }},
        }};
        let currentColormap = 'RdBu';

        function applyColormap(value, vmin, vmax) {{
            const t = Math.max(0, Math.min(1, (value - vmin) / (vmax - vmin)));
            return COLORMAPS[currentColormap].fn(t);
        }}

        // --- Colorbar ---
        function drawColorbar() {{
            const colorRange = parseFloat(document.getElementById('colorRange').value);
            const canvas = document.getElementById('colorbar');
            const ctx = canvas.getContext('2d');
            const w = canvas.width, h = canvas.height;
            const imgData = ctx.createImageData(w, h);
            for (let x = 0; x < w; x++) {{
                const v = -colorRange + (2 * colorRange * x / (w - 1));
                const [r, g, b] = applyColormap(v, -colorRange, colorRange);
                for (let y = 0; y < h; y++) {{
                    const i = (y * w + x) * 4;
                    imgData.data[i] = r; imgData.data[i+1] = g;
                    imgData.data[i+2] = b; imgData.data[i+3] = 255;
                }}
            }}
            ctx.putImageData(imgData, 0, 0);
            const v = colorRange.toFixed(1);
            document.getElementById('cbLabelMin').textContent = `-${{v}}`;
            document.getElementById('cbLabelMax').textContent = `+${{v}}`;
        }}

        // --- Click inspector ---
        function showClickInfo(layer, x, y) {{
            const pThreshold      = parseFloat(document.getElementById('pThreshold').value);
            const tThreshold      = parseFloat(document.getElementById('tThreshold').value);
            const cohensThreshold = parseFloat(document.getElementById('cohensThreshold').value);
            const meanThreshold   = parseFloat(document.getElementById('meanThreshold').value);
            const pooledThreshold = parseFloat(document.getElementById('pooledThreshold').value);

            const stat       = layer.stats[y][x];
            const pval       = layer.pvals[y][x];
            const cohens     = layer.cohens[y][x];
            const mean_diff  = layer.mean_diff[y][x];
            const pooled_std = layer.pooled_std[y][x];
            const act_name   = layer.activation_map[y][x];

            const isSig = pval < pThreshold &&
                          Math.abs(stat)      > tThreshold &&
                          Math.abs(cohens)    > cohensThreshold &&
                          Math.abs(mean_diff) > meanThreshold &&
                          pooled_std          > pooledThreshold;

            const actRow = act_name
                ? `<div class="info-row"><span class="info-label">Activation:</span> ${{act_name}}</div>`
                : `<div class="info-row"><span class="info-label">Activation:</span> (padding region)</div>`;

            const clickInfo = document.getElementById('clickInfo');
            clickInfo.className = 'click-info active';
            clickInfo.innerHTML = `
                <div class="info-row"><span class="info-label">Layer group:</span> ${{layer.name}}</div>
                ${{actRow}}
                <div class="info-row"><span class="info-label">Pixel position:</span> (${{x}}, ${{y}}) [${{currentResolution}}\u00d7${{currentResolution}}]</div>
                <div class="info-row"><span class="info-label">T-statistic:</span> ${{stat.toFixed(4)}}</div>
                <div class="info-row"><span class="info-label">P-value:</span> ${{pval.toExponential(4)}}</div>
                <div class="info-row"><span class="info-label">Cohen's d:</span> ${{cohens.toFixed(4)}}</div>
                <div class="info-row"><span class="info-label">Mean diff:</span> ${{mean_diff.toFixed(4)}}</div>
                <div class="info-row"><span class="info-label">Pooled std:</span> ${{pooled_std.toFixed(4)}}</div>
                <div class="info-row"><span class="info-label">Survives filters:</span> ${{isSig ? 'Yes' : 'No'}}</div>
            `;
        }}

        // --- Grid build (once per resolution change) ---
        let layerEntries = [];

        function buildGrid() {{
            const layersData = allLayersData[currentResolution];
            const grid = document.getElementById('layerGrid');
            grid.innerHTML = '';
            layerEntries = [];

            layersData.forEach(layer => {{
                const div = document.createElement('div');
                div.className = 'layer';

                const title = document.createElement('div');
                title.className = 'layer-title';
                title.title = layer.name;
                const parts = layer.name.split('.');
                title.textContent = parts.length > 2
                    ? '\u2026' + parts.slice(-2).join('.') : layer.name;
                div.appendChild(title);

                const canvas = document.createElement('canvas');
                canvas.width  = currentResolution;
                canvas.height = currentResolution;
                canvas.addEventListener('click', (e) => {{
                    const rect = canvas.getBoundingClientRect();
                    const x = Math.floor((e.clientX - rect.left) * canvas.width  / rect.width);
                    const y = Math.floor((e.clientY - rect.top)  * canvas.height / rect.height);
                    showClickInfo(layer, x, y);
                }});
                div.appendChild(canvas);

                const countEl = document.createElement('div');
                countEl.className = 'layer-count';
                div.appendChild(countEl);

                grid.appendChild(div);
                layerEntries.push({{ layer, ctx: canvas.getContext('2d'), countEl }});
            }});
        }}

        // --- Render pixels (on every filter change) ---
        function renderLayers() {{
            const pThreshold      = parseFloat(document.getElementById('pThreshold').value);
            const tThreshold      = parseFloat(document.getElementById('tThreshold').value);
            const cohensThreshold = parseFloat(document.getElementById('cohensThreshold').value);
            const meanThreshold   = parseFloat(document.getElementById('meanThreshold').value);
            const pooledThreshold = parseFloat(document.getElementById('pooledThreshold').value);
            const colorRange      = parseFloat(document.getElementById('colorRange').value);

            document.getElementById('pValue').value      = pThreshold.toFixed(3);
            document.getElementById('tValue').value      = tThreshold.toFixed(1);
            document.getElementById('cohensValue').value = cohensThreshold.toFixed(2);
            document.getElementById('meanValue').value   = meanThreshold.toFixed(2);
            document.getElementById('pooledValue').value = pooledThreshold.toFixed(2);
            document.getElementById('rangeValue').value  = colorRange.toFixed(1);

            let totalSig = 0, totalDims = 0;

            layerEntries.forEach(({{ layer, ctx, countEl }}) => {{
                const imageData = ctx.createImageData(currentResolution, currentResolution);
                let layerSig = 0;

                for (let i = 0; i < currentResolution; i++) {{
                    for (let j = 0; j < currentResolution; j++) {{
                        const idx        = i * currentResolution + j;
                        const stat       = layer.stats[i][j];
                        const pval       = layer.pvals[i][j];
                        const cohens     = layer.cohens[i][j];
                        const mean_diff  = layer.mean_diff[i][j];
                        const pooled_std = layer.pooled_std[i][j];

                        const significant = pval < pThreshold &&
                                            Math.abs(stat)      > tThreshold &&
                                            Math.abs(cohens)    > cohensThreshold &&
                                            Math.abs(mean_diff) > meanThreshold &&
                                            pooled_std          > pooledThreshold;

                        const px = idx * 4;
                        if (significant) {{
                            layerSig++;
                            const [r, g, b] = applyColormap(layer[colorMetric][i][j], -colorRange, colorRange);
                            imageData.data[px]   = r;
                            imageData.data[px+1] = g;
                            imageData.data[px+2] = b;
                            imageData.data[px+3] = 255;
                        }} else {{
                            imageData.data[px] = imageData.data[px+1] = imageData.data[px+2] = 30;
                            imageData.data[px+3] = 255;
                        }}
                    }}
                }}

                ctx.putImageData(imageData, 0, 0);

                const total = currentResolution * currentResolution;
                countEl.textContent =
                    `${{layerSig.toLocaleString()}} / ${{total.toLocaleString()}} (${{(100*layerSig/total).toFixed(1)}}%)`;

                totalSig  += layerSig;
                totalDims += total;
            }});

            document.getElementById('sigCount').textContent   = totalSig.toLocaleString();
            document.getElementById('totalCount').textContent = totalDims.toLocaleString();
            document.getElementById('sigPercent').textContent = (100 * totalSig / totalDims).toFixed(2);

            drawColorbar();
        }}

        // --- Slider listeners ---
        document.getElementById('pThreshold').addEventListener('input', renderLayers);
        document.getElementById('tThreshold').addEventListener('input', renderLayers);
        document.getElementById('cohensThreshold').addEventListener('input', renderLayers);
        document.getElementById('meanThreshold').addEventListener('input', renderLayers);
        document.getElementById('pooledThreshold').addEventListener('input', renderLayers);
        document.getElementById('colorRange').addEventListener('input', renderLayers);

        // Number input → slider sync (Enter or blur)
        function addNumberInputListeners(sliderId, inputId, min, max) {{
            const input = document.getElementById(inputId);
            function apply() {{
                const raw = parseFloat(input.value);
                const val = isNaN(raw) ? min : Math.min(max, Math.max(min, raw));
                document.getElementById(sliderId).value = val;
                renderLayers();
            }}
            input.addEventListener('keydown', function(e) {{ if (e.key === 'Enter') apply(); }});
            input.addEventListener('blur', apply);
        }}
        addNumberInputListeners('pThreshold',     'pValue',      0.001, 1.001);
        addNumberInputListeners('tThreshold',      'tValue',      0,     100);
        addNumberInputListeners('cohensThreshold', 'cohensValue', 0,     10);
        addNumberInputListeners('meanThreshold',   'meanValue',   0,     10);
        addNumberInputListeners('pooledThreshold', 'pooledValue', 0,     10);
        addNumberInputListeners('colorRange',      'rangeValue',  1,     100);

        // --- Presets ---
        const PRESETS = {json.dumps(presets)};

        function applyPreset(name) {{
            const p = PRESETS[name];
            document.getElementById('pThreshold').value      = p.p;
            document.getElementById('tThreshold').value      = p.t;
            document.getElementById('cohensThreshold').value = p.cohens;
            document.getElementById('meanThreshold').value   = p.mean;
            document.getElementById('pooledThreshold').value = p.pooled;
            document.getElementById('colorRange').value      = p.range;
            renderLayers();
            document.querySelectorAll('.preset-btn').forEach(btn => btn.classList.remove('active'));
            const activeBtn = document.querySelector(`.preset-btn[data-preset="${{name}}"]`);
            if (activeBtn) activeBtn.classList.add('active');
        }}

        const presetsBar = document.getElementById('presetsBar');
        Object.keys(PRESETS).forEach((name, idx) => {{
            const btn = document.createElement('button');
            btn.className = 'preset-btn' + (idx === 0 ? ' active' : '');
            btn.dataset.preset = name;
            btn.textContent = name;
            btn.addEventListener('click', () => applyPreset(name));
            presetsBar.appendChild(btn);
        }});

        // Reset button
        const resetBtn = document.createElement('button');
        resetBtn.className = 'preset-btn';
        resetBtn.style.cssText = 'border-color:#666;margin-left:16px';
        resetBtn.textContent = 'Reset';
        resetBtn.addEventListener('click', () => applyPreset(Object.keys(PRESETS)[0]));
        presetsBar.appendChild(resetBtn);

        // --- Metric selector ---
        const metricBar = document.getElementById('metricBar');
        METRIC_OPTIONS.forEach((opt, idx) => {{
            const btn = document.createElement('button');
            btn.className = 'metric-btn' + (idx === 0 ? ' active' : '');
            btn.dataset.metric = opt.key;
            btn.textContent = opt.label;
            btn.addEventListener('click', () => {{
                colorMetric = opt.key;
                renderLayers();
                document.querySelectorAll('.metric-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
            }});
            metricBar.appendChild(btn);
        }});

        // --- Resolution switcher ---
        const resSwitcher = document.getElementById('resSwitcher');
        if (resolutions.length > 1) {{
            resSwitcher.style.display = '';
            resolutions.forEach((res, idx) => {{
                const btn = document.createElement('button');
                btn.className = 'res-btn' + (idx === 0 ? ' active' : '');
                btn.dataset.res = res;
                btn.textContent = `${{res}}\u00d7${{res}}`;
                btn.addEventListener('click', () => {{
                    currentResolution = res;
                    buildGrid();
                    renderLayers();
                    document.querySelectorAll('.res-btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                }});
                resSwitcher.appendChild(btn);
            }});
        }}

        // --- Colormap panel ---
        const cmapBtnList = document.getElementById('cmapBtnList');
        Object.keys(COLORMAPS).forEach(key => {{
            const btn = document.createElement('button');
            btn.className = 'cmap-btn' + (key === currentColormap ? ' active' : '');
            btn.dataset.cmap = key;

            const thumb = document.createElement('canvas');
            thumb.width = 50; thumb.height = 12;
            thumb.style.cssText = 'border-radius:2px;flex-shrink:0';
            const tCtx = thumb.getContext('2d');
            const tImg = tCtx.createImageData(50, 12);
            for (let x = 0; x < 50; x++) {{
                const [r, g, b] = COLORMAPS[key].fn(x / 49);
                for (let y = 0; y < 12; y++) {{
                    const i = (y * 50 + x) * 4;
                    tImg.data[i] = r; tImg.data[i+1] = g; tImg.data[i+2] = b; tImg.data[i+3] = 255;
                }}
            }}
            tCtx.putImageData(tImg, 0, 0);

            const nameSpan = document.createElement('span');
            nameSpan.textContent = COLORMAPS[key].label;
            btn.appendChild(thumb);
            btn.appendChild(nameSpan);
            btn.addEventListener('click', () => {{
                currentColormap = key;
                renderLayers();
                document.querySelectorAll('.cmap-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
            }});
            cmapBtnList.appendChild(btn);
        }});

        buildGrid();
        renderLayers();
    </script>
</body>
</html>"""

    with open(html_path, "w") as f:
        f.write(html_content)

    return html_path
