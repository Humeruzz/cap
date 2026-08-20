import json
from itertools import combinations
from pathlib import Path

import numpy as np

from cap.plot.stats_html import group_by_layer


def _sort_layer_names(layer_names):
    dummy = {name: None for name in layer_names}
    groups = group_by_layer(dummy)

    int_keys = sorted(k for k in groups if isinstance(k, int))
    str_keys = [k for k in groups if isinstance(k, str)]

    input_layers, output_layers, other_layers = [], [], []
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

    sorted_names = []
    for group_key in input_layers + int_keys + other_layers + output_layers:
        sorted_names.extend(sorted(groups[group_key].keys()))

    return sorted_names


def _shorten_name(name):
    parts = name.split(".")
    return ("\u2026" + ".".join(parts[-2:])) if len(parts) > 2 else name


def _layer_group_index(name):
    if ".layers." in name:
        parts = name.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    pass
    return name


def compute_layer_similarities(labeled_stats, t_threshold, d_threshold):
    """
    For each pair of labeled statistics dicts, compute per-layer cosine similarity
    of t_stats vectors masked by |t| >= t_threshold AND |cohens_d| >= d_threshold.

    Returns {pair_key: {layer_name: cosine_sim}}
    """
    results = {}

    for (label_a, stats_a), (label_b, stats_b) in combinations(labeled_stats, 2):
        pair_key = f"{label_a}\u2194{label_b}"
        layer_sims = {}

        common_layers = set(stats_a.keys()) & set(stats_b.keys())

        for layer in common_layers:
            t_a = np.asarray(stats_a[layer]["t_stats"], dtype=np.float64)
            d_a = np.asarray(stats_a[layer]["cohens_d"], dtype=np.float64)
            t_b = np.asarray(stats_b[layer]["t_stats"], dtype=np.float64)
            d_b = np.asarray(stats_b[layer]["cohens_d"], dtype=np.float64)

            min_len = min(len(t_a), len(t_b))
            t_a, d_a = t_a[:min_len], d_a[:min_len]
            t_b, d_b = t_b[:min_len], d_b[:min_len]

            mask_a = (np.abs(t_a) >= t_threshold) & (np.abs(d_a) >= d_threshold)
            mask_b = (np.abs(t_b) >= t_threshold) & (np.abs(d_b) >= d_threshold)

            v_a = t_a * mask_a
            v_b = t_b * mask_b

            norm_a = float(np.linalg.norm(v_a))
            norm_b = float(np.linalg.norm(v_b))

            if norm_a > 1e-10 and norm_b > 1e-10:
                sim = float(np.dot(v_a, v_b) / (norm_a * norm_b))
            else:
                sim = 0.0

            layer_sims[layer] = round(sim, 4)

        results[pair_key] = layer_sims

    return results


def compute_layer_pearson(labeled_stats, t_threshold, d_threshold):
    """
    Per-layer Pearson correlation of t_stats on the intersection of active neurons.
    Active = |t| >= t_threshold AND |d| >= d_threshold for both conditions.
    Returns {pair_key: {layer_name: pearson_r}} in [-1, 1]; 0 when fewer than 2 shared neurons.
    """
    results = {}
    for (label_a, stats_a), (label_b, stats_b) in combinations(labeled_stats, 2):
        pair_key = f"{label_a}↔{label_b}"
        layer_sims = {}
        common_layers = set(stats_a.keys()) & set(stats_b.keys())
        for layer in common_layers:
            t_a = np.asarray(stats_a[layer]["t_stats"], dtype=np.float64)
            d_a = np.asarray(stats_a[layer]["cohens_d"], dtype=np.float64)
            t_b = np.asarray(stats_b[layer]["t_stats"], dtype=np.float64)
            d_b = np.asarray(stats_b[layer]["cohens_d"], dtype=np.float64)
            min_len = min(len(t_a), len(t_b))
            t_a, d_a, t_b, d_b = t_a[:min_len], d_a[:min_len], t_b[:min_len], d_b[:min_len]
            mask_a = (np.abs(t_a) >= t_threshold) & (np.abs(d_a) >= d_threshold)
            mask_b = (np.abs(t_b) >= t_threshold) & (np.abs(d_b) >= d_threshold)
            shared = mask_a & mask_b
            if shared.sum() >= 2:
                r = float(np.corrcoef(t_a[shared], t_b[shared])[0, 1])
                layer_sims[layer] = round(r if np.isfinite(r) else 0.0, 4)
            else:
                layer_sims[layer] = 0.0
        results[pair_key] = layer_sims
    return results


def compute_layer_overlap(labeled_stats, t_threshold, d_threshold):
    """
    Per-layer Jaccard overlap of active-neuron sets.
    Active = |t| >= t_threshold AND |d| >= d_threshold.
    Returns {pair_key: {layer_name: jaccard}} in [0, 1].
    """
    results = {}
    for (label_a, stats_a), (label_b, stats_b) in combinations(labeled_stats, 2):
        pair_key = f"{label_a}↔{label_b}"
        layer_sims = {}
        common_layers = set(stats_a.keys()) & set(stats_b.keys())
        for layer in common_layers:
            t_a = np.asarray(stats_a[layer]["t_stats"], dtype=np.float64)
            d_a = np.asarray(stats_a[layer]["cohens_d"], dtype=np.float64)
            t_b = np.asarray(stats_b[layer]["t_stats"], dtype=np.float64)
            d_b = np.asarray(stats_b[layer]["cohens_d"], dtype=np.float64)
            min_len = min(len(t_a), len(t_b))
            t_a, d_a, t_b, d_b = t_a[:min_len], d_a[:min_len], t_b[:min_len], d_b[:min_len]
            mask_a = (np.abs(t_a) >= t_threshold) & (np.abs(d_a) >= d_threshold)
            mask_b = (np.abs(t_b) >= t_threshold) & (np.abs(d_b) >= d_threshold)
            union = int((mask_a | mask_b).sum())
            layer_sims[layer] = (
                round(float((mask_a & mask_b).sum()) / union, 4) if union > 0 else 0.0
            )
        results[pair_key] = layer_sims
    return results


def create_similarity_viewer(
    labeled_stats, output_path, t_thresholds=None, d_thresholds=None, presets=None
):
    """
    Precompute per-layer cosine similarity for a grid of thresholds and write similarity.html.

    labeled_stats: list of (label, stats_dict) where stats_dict = H5Store.load_statistics() result
    output_path:   directory to write similarity.html into
    """
    if t_thresholds is None:
        t_thresholds = [0.0, 1.0, 2.0, 3.0, 5.0]
    if d_thresholds is None:
        d_thresholds = [0.0, 0.1, 0.3, 0.5, 0.8]
    if presets is None:
        presets = {
            "Show All": {"t": 0.0, "cohens": 0.0},
            "Exploratory": {"t": 2.0, "cohens": 0.3},
            "Strict": {"t": 3.0, "cohens": 0.5},
        }

    if len(labeled_stats) < 2:
        raise ValueError("Need at least 2 statistics files to compare.")

    labels = [label for label, _ in labeled_stats]
    pairs = [f"{a}\u2194{b}" for (a, _), (b, _) in combinations(labeled_stats, 2)]

    all_layer_names = list(labeled_stats[0][1].keys())
    sorted_layers = _sort_layer_names(all_layer_names)

    # Alternating group indices for row shading
    layer_groups = []
    current_group = object()  # sentinel
    group_counter = -1
    for name in sorted_layers:
        g = _layer_group_index(name)
        if g != current_group:
            current_group = g
            group_counter += 1
        layer_groups.append(group_counter % 2)

    n_combos = len(t_thresholds) * len(d_thresholds)
    print(
        f"Precomputing similarity grid "
        f"({len(t_thresholds)}\u00d7{len(d_thresholds)}={n_combos} combos, "
        f"3 metrics, {len(pairs)} pairs, {len(sorted_layers)} layers)..."
    )

    data_cosine = {}
    data_pearson = {}
    data_overlap = {}
    for ti, t_thresh in enumerate(t_thresholds):
        for di, d_thresh in enumerate(d_thresholds):
            key = f"{ti}_{di}"  # index-based to avoid float-to-string inconsistencies
            cos_sims = compute_layer_similarities(labeled_stats, t_thresh, d_thresh)
            pear_sims = compute_layer_pearson(labeled_stats, t_thresh, d_thresh)
            over_sims = compute_layer_overlap(labeled_stats, t_thresh, d_thresh)
            data_cosine[key] = {
                p: [cos_sims.get(p, {}).get(layer, 0.0) for layer in sorted_layers] for p in pairs
            }
            data_pearson[key] = {
                p: [pear_sims.get(p, {}).get(layer, 0.0) for layer in sorted_layers] for p in pairs
            }
            data_overlap[key] = {
                p: [over_sims.get(p, {}).get(layer, 0.0) for layer in sorted_layers] for p in pairs
            }

    _MAX_DISPLAY_DIMS = 2048
    vectors_t = {}
    for label, stats in labeled_stats:
        vectors_t[label] = {}
        for layer in sorted_layers:
            if layer not in stats:
                continue
            arr = np.asarray(stats[layer]["t_stats"], dtype=np.float32)
            if len(arr) > _MAX_DISPLAY_DIMS:
                stride = len(arr) // _MAX_DISPLAY_DIMS
                arr = arr[::stride][:_MAX_DISPLAY_DIMS]
            vectors_t[label][layer] = [round(float(v), 3) for v in arr]

    payload = {
        "labels": labels,
        "pairs": pairs,
        "layers": sorted_layers,
        "layer_groups": layer_groups,
        "t_thresholds": t_thresholds,
        "d_thresholds": d_thresholds,
        "data_cosine": data_cosine,
        "data_pearson": data_pearson,
        "data_overlap": data_overlap,
        "vectors_t": vectors_t,
    }

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "similarity.html"

    with open(html_path, "w") as f:
        f.write(_build_html(payload, presets))

    print(f"Done. HTML written to: {html_path}")
    return html_path


def _layer_type(name):
    parts = name.split(".")
    return ".".join(parts[-2:]) if len(parts) > 2 else name


def _block_label(name):
    """Return '[N]' for transformer block layers, '' for special layers."""
    if ".layers." in name:
        parts = name.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try:
                    return f"[{int(parts[i + 1])}]"
                except ValueError:
                    pass
    return ""


def _build_html(payload, presets):
    layers = payload["layers"]
    pairs = payload["pairs"]
    layer_groups = payload["layer_groups"]

    # Unique layer types in order of first appearance
    seen = {}
    for name in layers:
        t = _layer_type(name)
        if t not in seen:
            seen[t] = len(seen)
    unique_types = list(seen.keys())

    # Build table rows — left column has two spans for normal / filtered view
    pair_cells = "".join(
        f'<td class="col-pair sim-cell" data-pair="{j}"></td>' for j in range(len(pairs))
    )
    rows_html = "\n".join(
        # The IIFE binds the two derived values per row; inlining them would recompute
        # both across several f-string slots.
        (  # noqa: PLC3002
            lambda short, blk: (
                f'<tr class="layer-row g{layer_groups[i]}" data-layer="{i}" data-type="{_layer_type(layer)}">'
                f'<td class="col-layer" title="{layer}">'
                f'<span class="ln-normal">{short}</span>'
                f'<span class="ln-filtered" style="display:none">{blk + " " if blk else ""}{short}</span>'
                f"</td>"
                f"{pair_cells}"
                f"</tr>"
            )
        )(_shorten_name(layer), _block_label(layer))
        for i, layer in enumerate(layers)
    )

    pair_headers = "".join(f'<th class="col-pair">{p}</th>' for p in pairs)
    mean_cells = "".join(
        f'<td class="col-pair mean-cell" data-pair="{j}"></td>' for j in range(len(pairs))
    )

    subtitle = " · ".join(payload["labels"])
    t_max = max(payload["t_thresholds"])
    d_max = max(payload["d_thresholds"])

    return f"""<!DOCTYPE html>
<html>
<head>
<title>Cosine Similarity \u2014 {subtitle}</title>
<meta charset="UTF-8">
<style>
  body {{
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 16px;
    background: #1a1a1a;
    color: #eee;
  }}
  h1 {{ margin: 0 0 4px; font-size: 20px; }}
  .subtitle {{ color: #aaa; margin: 0 0 14px; font-size: 13px; }}
  .controls {{
    position: sticky;
    top: 0;
    z-index: 20;
    background: #2a2a2a;
    padding: 8px 12px;
    border-radius: 8px;
    margin-bottom: 14px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }}
  .controls-row {{
    display: flex;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
  }}
  .controls-row + .controls-row {{
    border-top: 1px solid #383838;
    padding-top: 8px;
  }}
  .ctrl-group {{ display: flex; align-items: center; gap: 8px; }}
  .ctrl-label {{ font-size: 12px; font-weight: bold; white-space: nowrap; min-width: 70px; }}
  input[type=range] {{ width: 140px; vertical-align: middle; }}
  .threshold-display {{
    font-family: monospace;
    font-size: 12px;
    background: #1a1a1a;
    border: 1px solid #444;
    border-radius: 4px;
    padding: 2px 6px;
    min-width: 36px;
    text-align: right;
    color: #7af;
  }}
  .sep {{ border-left: 1px solid #444; height: 28px; align-self: center; }}
  .row-label {{ font-size: 12px; font-weight: bold; white-space: nowrap; color: #aaa; }}
  .preset-btn, .filter-btn, .guide-btn {{
    background: #3a3a3a;
    color: #eee;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 3px 9px;
    cursor: pointer;
    font-size: 12px;
    white-space: nowrap;
  }}
  .preset-btn:hover {{ background: #4a4a4a; border-color: #7a9a7a; }}
  .preset-btn.active {{ background: #3a5a3a; border-color: #5a8a5a; }}
  .filter-btn:hover {{ background: #4a4a4a; border-color: #9a8a5a; }}
  .filter-btn.active {{ background: #5a4a1a; border-color: #aa8a3a; color: #ffd; }}
  #filterAll.active {{ background: #3a3a3a; border-color: #777; color: #eee; }}
  .guide-btn  {{ border-color: #557; color: #aaf; margin-left: auto; }}
  .guide-btn:hover  {{ background: #2a2a4a; border-color: #88b; }}
  .rec-btn  {{ background: #3a3a3a; border-color: #553; color: #fea; }}
  .rec-btn:hover {{ background: #3a3a1a; border-color: #aa8; }}
  /* Modal tabs */
  .modal-tabs {{
    display: flex;
    border-bottom: 1px solid #444;
    margin-bottom: 16px;
    gap: 0;
  }}
  .modal-tab {{
    padding: 6px 18px;
    cursor: pointer;
    font-size: 13px;
    border: none;
    background: none;
    color: #888;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
  }}
  .modal-tab:hover {{ color: #ccc; }}
  .modal-tab.active {{ color: #eee; border-bottom-color: #7af; }}
  .tab-panel {{ display: none; }}
  .tab-panel.active {{ display: block; }}
  /* Recommendation entries */
  .rec-section {{ margin-bottom: 20px; }}
  .rec-section-title {{
    font-size: 12px; font-weight: bold; text-transform: uppercase;
    letter-spacing: 0.06em; color: #888; margin-bottom: 10px;
  }}
  .sim-row {{
    display: grid; grid-template-columns: 52px 1fr;
    gap: 6px 12px; padding: 5px 0; border-bottom: 1px solid #2e2e2e;
    align-items: center; font-size: 12px;
  }}
  .sim-row:last-child {{ border-bottom: none; }}
  .sim-badge {{
    font-family: monospace; font-size: 13px; font-weight: bold;
    text-align: center; border-radius: 4px; padding: 2px 4px;
  }}
  .layer-row2 {{
    display: grid; grid-template-columns: max-content 1fr;
    gap: 6px 12px; padding: 5px 0; border-bottom: 1px solid #2e2e2e;
    align-items: baseline; font-size: 12px;
  }}
  .layer-row2:last-child {{ border-bottom: none; }}
  .preset-row {{
    display: grid; grid-template-columns: 110px 1fr;
    gap: 6px 12px; padding: 5px 0; border-bottom: 1px solid #2e2e2e;
    align-items: baseline; font-size: 12px;
  }}
  .preset-row:last-child {{ border-bottom: none; }}
  .preset-badge {{
    font-size: 11px; background: #3a5a3a; border: 1px solid #5a8a5a;
    color: #9d9; border-radius: 3px; padding: 1px 5px;
    white-space: nowrap; justify-self: start;
  }}
  .condition-block {{
    background: #222; border-left: 3px solid #7af;
    border-radius: 0 6px 6px 0; padding: 10px 14px; margin-bottom: 10px;
  }}
  .condition-title {{
    font-size: 12px; font-weight: bold; color: #7af; margin-bottom: 6px;
  }}
  .condition-steps {{ margin: 0; padding-left: 18px; color: #ccc; font-size: 12px; line-height: 1.7; }}
  .workflow-steps {{
    counter-reset: step; list-style: none; padding: 0; margin: 0;
  }}
  .workflow-steps li {{
    counter-increment: step; padding: 6px 0 6px 32px; position: relative;
    font-size: 12px; color: #ccc; border-bottom: 1px solid #2e2e2e; line-height: 1.5;
  }}
  .workflow-steps li:last-child {{ border-bottom: none; }}
  .workflow-steps li::before {{
    content: counter(step);
    position: absolute; left: 0; top: 6px;
    width: 20px; height: 20px; border-radius: 50%;
    background: #3a3a6a; color: #aaf; font-size: 11px; font-weight: bold;
    display: flex; align-items: center; justify-content: center;
    line-height: 1;
  }}
  /* Modal */
  .modal-overlay {{
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.75);
    z-index: 200;
    align-items: flex-start;
    justify-content: center;
    padding-top: 40px;
  }}
  .modal-overlay.open {{ display: flex; }}
  .modal-box {{
    background: #252525;
    border: 1px solid #555;
    border-radius: 10px;
    padding: 20px 24px;
    max-width: 720px;
    max-height: 80vh;
    overflow-y: auto;
    width: 92%;
    position: relative;
  }}
  .modal-title {{
    font-size: 17px;
    font-weight: bold;
    margin: 0 0 16px;
    color: #eee;
    border-bottom: 1px solid #444;
    padding-bottom: 10px;
  }}
  .modal-close {{
    position: absolute;
    top: 14px;
    right: 16px;
    background: none;
    border: none;
    color: #aaa;
    font-size: 22px;
    cursor: pointer;
    line-height: 1;
    padding: 0;
  }}
  .modal-close:hover {{ color: #fff; }}
  .glossary-group {{ margin-bottom: 20px; }}
  .glossary-group-title {{
    font-size: 12px;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #888;
    margin-bottom: 8px;
  }}
  .glossary-entry {{
    padding: 5px 0;
    border-bottom: 1px solid #333;
    font-size: 12px;
  }}
  .glossary-entry:last-child {{ border-bottom: none; }}
  .entry-main {{
    display: grid;
    grid-template-columns: 130px 1fr;
    gap: 6px 12px;
    align-items: start;
  }}
  .g-type {{
    font-family: monospace;
    font-size: 12px;
    background: #5a4a1a;
    border: 1px solid #aa8a3a;
    color: #ffd;
    border-radius: 3px;
    padding: 1px 5px;
    white-space: nowrap;
    justify-self: start;
    margin-top: 2px;
  }}
  .g-right {{ display: flex; flex-direction: column; gap: 2px; }}
  .g-full {{ font-family: monospace; font-size: 10px; color: #777; }}
  .g-desc {{ color: #ccc; line-height: 1.4; }}
  .detail-toggle {{
    font-size: 11px; color: #556; cursor: pointer; background: none;
    border: none; padding: 2px 0 0; text-align: left; margin-top: 4px;
    align-self: flex-start;
  }}
  .detail-toggle:hover {{ color: #aaf; }}
  .detail-panel {{
    display: none;
    padding: 10px 0 4px;
    border-top: 1px solid #2a2a2a;
    margin-top: 6px;
  }}
  .detail-panel.open {{ display: block; }}
  .detail-section {{ margin-bottom: 10px; }}
  .detail-section:last-child {{ margin-bottom: 0; }}
  .detail-section-title {{
    font-size: 10px; font-weight: bold; text-transform: uppercase;
    letter-spacing: 0.07em; color: #7af; margin-bottom: 4px;
  }}
  .detail-text {{ font-size: 12px; color: #bbb; line-height: 1.5; }}
  /* Table */
  .table-wrap {{
    overflow: auto;
    max-height: calc(100vh - 260px);
    border-radius: 6px;
    border: 1px solid #333;
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    font-size: 12px;
  }}
  thead {{
    position: sticky;
    top: 0;
    z-index: 10;
    background: #2a2a2a;
  }}
  th, td {{
    border: 1px solid #333;
    padding: 3px 6px;
    white-space: nowrap;
  }}
  th {{ font-weight: bold; text-align: center; }}
  .col-layer {{
    position: sticky;
    left: 0;
    background: #1e1e1e;
    z-index: 5;
    font-family: monospace;
    font-size: 11px;
    min-width: 120px;
    max-width: 200px;
    overflow: hidden;
    text-overflow: ellipsis;
    color: #ccc;
  }}
  thead .col-layer {{ z-index: 15; background: #2a2a2a; color: #eee; text-align: left; }}
  .col-pair {{ text-align: center; min-width: 80px; }}
  .sim-cell {{ text-align: center; font-family: monospace; font-size: 11px; cursor: default; }}
  .mean-row {{ font-weight: bold; }}
  .mean-row .col-pair {{ font-family: monospace; text-align: center; }}
  .mean-row .col-layer {{ color: #aaa; font-style: italic; font-size: 11px; }}
  tr.g0 .col-layer {{ background: #1e1e1e; }}
  tr.g1 .col-layer {{ background: #242424; }}
  tr.g0 td:not(.col-layer) {{ background: #1a1a1a; }}
  tr.g1 td:not(.col-layer) {{ background: #1f1f1f; }}
  .ln-filtered .blk {{ color: #7af; font-weight: bold; }}
  .legend {{
    margin-top: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    color: #888;
  }}
  .legend-bar {{
    width: 200px;
    height: 14px;
    border-radius: 3px;
    background: linear-gradient(to right, #ff4444, #ffffff, #4444ff);
    border: 1px solid #444;
  }}
  .metric-tab {{
    background: #3a3a3a;
    color: #eee;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 3px 9px;
    cursor: pointer;
    font-size: 12px;
    white-space: nowrap;
  }}
  .metric-tab:hover {{ background: #4a4a4a; border-color: #7a9aaa; }}
  .metric-tab.active {{ background: #1a3a5a; border-color: #4a8aaa; color: #adf; }}
</style>
</head>
<body>
<h1>Layer-wise Cosine Similarity</h1>
<p class="subtitle">Comparing: {subtitle}</p>

<div class="controls">
  <div class="controls-row">
    <div class="ctrl-group">
      <span class="row-label">Presets:</span>
      <div id="presetsBar"></div>
    </div>
    <div class="sep"></div>
    <div class="ctrl-group">
      <span class="ctrl-label">|T-stat| \u2265</span>
      <input type="range" id="tSlider" min="0" max="{t_max}" step="0.1" value="0">
      <span class="threshold-display" id="tDisplay">0.0</span>
    </div>
    <div class="ctrl-group">
      <span class="ctrl-label">|Cohen\u2019s d| \u2265</span>
      <input type="range" id="dSlider" min="0" max="{d_max}" step="0.01" value="0">
      <span class="threshold-display" id="dDisplay">0.00</span>
    </div>
    <button class="guide-btn" id="guideBtn">\U0001f4d6 Layer guide</button>
    <button class="rec-btn"  id="recBtn">\U0001f4a1 Recommendations</button>
  </div>
  <div class="controls-row">
    <span class="row-label">Layer type:</span>
    <button class="filter-btn active" id="filterAll">All</button>
    <div id="filterBar" style="display:flex;gap:4px;flex-wrap:wrap"></div>
  </div>
  <div class="controls-row">
    <span class="row-label">Metric:</span>
    <button class="metric-tab active" data-metric="data_cosine">Cosine</button>
    <button class="metric-tab" data-metric="data_pearson">Pearson</button>
    <button class="metric-tab" data-metric="data_overlap">Overlap</button>
    <span id="metricDesc" style="font-size:11px;color:#777;margin-left:8px;">direction of masked t-stats (−1 to +1)</span>
  </div>
</div>

<!-- Recommendations modal -->
<div class="modal-overlay" id="recModal">
  <div class="modal-box">
    <div class="modal-title">\U0001f4a1 Recommendations</div>
    <button class="modal-close" id="recClose">&times;</button>
    <div class="modal-tabs">
      <button class="modal-tab active" data-tab="interpret">Interpret</button>
      <button class="modal-tab" data-tab="whatnext">What to do next</button>
    </div>
    <div id="tab-interpret" class="tab-panel active"></div>
    <div id="tab-whatnext" class="tab-panel"></div>
  </div>
</div>

<!-- Guide modal -->
<div class="modal-overlay" id="guideModal">
  <div class="modal-box">
    <div class="modal-title">Layer Reference \u2014 Qwen3 Transformer</div>
    <button class="modal-close" id="guideClose">&times;</button>
    <div id="glossaryContent"></div>
  </div>
</div>

<div class="table-wrap">
  <table id="simTable">
    <thead>
      <tr>
        <th class="col-layer">Layer</th>
        {pair_headers}
      </tr>
      <tr class="mean-row">
        <td class="col-layer">Mean (visible non-zero)</td>
        {mean_cells}
      </tr>
    </thead>
    <tbody>
{rows_html}
    </tbody>
  </table>
</div>

<div class="legend">
  <span id="legendLeft">\u22121 (dissimilar)</span>
  <div class="legend-bar"></div>
  <span id="legendRight">+1 (identical direction)</span>
  <span style="margin-left:16px;color:#666">&mdash; 0 = no overlap or orthogonal after thresholding</span>
</div>

<div id="heatmapPopup" style="display:none;position:fixed;z-index:300;pointer-events:none;
  background:#1e1e1e;border:1px solid #555;border-radius:8px;padding:10px 14px;
  box-shadow:0 4px 24px rgba(0,0,0,.6);">
  <div id="heatmapTitle" style="font-size:11px;color:#aaa;margin-bottom:8px;font-family:monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:400px;"></div>
  <div style="display:flex;gap:12px;align-items:flex-start;">
    <div>
      <div id="heatmapLabelA" style="font-size:11px;color:#7af;margin-bottom:4px;text-align:center;"></div>
      <canvas id="heatmapA" style="image-rendering:pixelated;width:180px;height:180px;border:1px solid #333;display:block;"></canvas>
    </div>
    <div>
      <div id="heatmapLabelB" style="font-size:11px;color:#fa7;margin-bottom:4px;text-align:center;"></div>
      <canvas id="heatmapB" style="image-rendering:pixelated;width:180px;height:180px;border:1px solid #333;display:block;"></canvas>
    </div>
  </div>
  <div id="heatmapFooter" style="font-size:10px;color:#666;margin-top:6px;"></div>
</div>

<script>
const simData    = {json.dumps(payload)};
const PRESETS    = {json.dumps(presets)};
const LAYER_TYPES = {json.dumps(unique_types)};

const GLOSSARY_GROUPS = [
  {{
    title: "Special \u2014 appear once",
    entries: [
      {{ type: "embed.tokens", full: "model.embed_tokens",
        desc: "Token embedding table output. Converts token IDs into hidden-dim vectors. No learned transformation \u2014 pure lookup. Captures which tokens are present, nothing about their context." }},
      {{ type: "model.norm",   full: "model.norm",
        desc: "Final RMS LayerNorm before lm_head. Normalizes the residual stream at the very end. Its direction heavily shapes which logits get amplified." }},
      {{ type: "lm.head",      full: "lm_head",
        desc: "Linear projection from hidden_dim to vocabulary size (151,936 for Qwen3). Pre-softmax logits. High similarity here means the two languages drive the model toward the same output token distribution." }},
    ]
  }},
  {{
    title: "Layer norms \u2014 \u00d7N per transformer block",
    entries: [
      {{ type: "input.layernorm", full: "model.layers.N.input_layernorm",
        desc: "Pre-attention RMS norm. Applied to the residual stream before self-attention. Because RMS norm rescales to unit length, this captures the direction of the residual, not its magnitude. Differences here reflect how much the residual stream direction differs between conditions." }},
      {{ type: "attention.layernorm", full: "model.layers.N.post_attention_layernorm",
        desc: "Pre-MLP RMS norm. Applied after attention output has been added to the residual, before the MLP. Same interpretation as input.layernorm but at the post-attention stage." }},
    ]
  }},
  {{
    title: "Self-attention \u2014 \u00d7N per transformer block",
    entries: [
      {{ type: "q.proj", full: "model.layers.N.self_attn.q_proj",
        desc: "Query projection. Shape: (n_heads \u00d7 head_dim). Represents what this token is searching for in the context. High similarity means the model\u2019s attention queries are oriented in the same direction for both conditions." }},
      {{ type: "k.proj", full: "model.layers.N.self_attn.k_proj",
        desc: "Key projection. Shape: (n_kv_heads \u00d7 head_dim). What this token advertises to other tokens\u2019 queries. Qwen3 uses Grouped Query Attention (GQA) so fewer key heads than query heads." }},
      {{ type: "v.proj", full: "model.layers.N.self_attn.v_proj",
        desc: "Value projection. Shape: (n_kv_heads \u00d7 head_dim). The content retrieved when this token is attended to. What gets mixed into the query token\u2019s representation." }},
      {{ type: "o.proj", full: "model.layers.N.self_attn.o_proj",
        desc: "Attention output projection back to hidden_dim. The net vector that attention contributes to the residual stream. Reflects both what was attended to and how it gets written back." }},
      {{ type: "q.norm", full: "model.layers.N.self_attn.q_norm",
        desc: "Per-head RMS norm applied to queries (Qwen3-specific). Stabilizes attention score magnitudes, especially useful at high learning rates or with long contexts." }},
      {{ type: "k.norm", full: "model.layers.N.self_attn.k_norm",
        desc: "Per-head RMS norm applied to keys (Qwen3-specific). Mirrors q.norm \u2014 prevents attention logits from growing unboundedly with sequence length." }},
    ]
  }},
  {{
    title: "MLP / FFN \u2014 SwiGLU gated FFN, \u00d7N per block",
    entries: [
      {{ type: "gate.proj", full: "model.layers.N.mlp.gate_proj",
        desc: "Gate branch linear projection. Expanded to ~4\u00d7 hidden_dim. Raw pre-activation values; not yet passed through non-linearity. Less interpretable on its own." }},
      {{ type: "act.fn", full: "model.layers.N.mlp.act_fn",
        desc: "SiLU(gate_proj(x)) \u2014 the gated activated MLP neurons. Most interpretable layer: a neuron here is effectively \u2018on\u2019 only when the gate projection is positive. Best proxy for which MLP features fired. High cosine sim = the same intermediate features activate for both conditions." }},
      {{ type: "up.proj", full: "model.layers.N.mlp.up_proj",
        desc: "Content branch. Combined with the gate: act_fn(gate_proj(x)) \u2299 up_proj(x). Represents the \u2018what\u2019 that the MLP wants to write, before the gate decides whether to write it." }},
      {{ type: "down.proj", full: "model.layers.N.mlp.down_proj",
        desc: "Down projection back to hidden_dim. The net vector the MLP contributes to the residual stream. Directly comparable to o.proj: this is the MLP\u2019s \u2018write\u2019 operation." }},
    ]
  }},
];

const GLOSSARY_DETAILS = {{
  "embed.tokens": {{
    what: "Token embedding table. Maps each of the 151,936 Qwen3 vocabulary tokens to a hidden_dim vector (3584 for 8B, 5120 for 14B). The same token always produces the same vector regardless of context. The weight matrix is often weight-tied with lm_head.",
    why: "Similarity here mostly reflects vocabulary overlap between the two conditions. Arabic, Chinese, and European languages use entirely different tokens (different scripts and tokenisation schemes), so embed_tokens similarity is almost always low across languages \u2014 this is expected and does not indicate different reasoning circuits. Only compare embed_tokens similarity for conditions that share the same language.",
    caveat: "This layer does no computation \u2014 it is a pure lookup. Do not draw mechanistic conclusions from embed_tokens similarity. The captured activation corresponds to the last token position (the one the model is predicting from)."
  }},
  "model.norm": {{
    what: "Final RMS LayerNorm applied to the full residual stream just before lm_head. Formula: \u03b3 \u00d7 (x / RMS(x)), where \u03b3 is a learned per-dimension scale vector. This is the very last transformation before vocabulary projection.",
    why: "Because lm_head is a linear projection of this layer\u2019s output, similarity here directly predicts output-logit similarity. High similarity means the model has arrived at the same final residual direction for both conditions, regardless of how it got there. It is a useful aggregate signal: if model.norm similarity is high, the entire stack converges on the same answer.",
    caveat: "RMS norm erases magnitude information \u2014 only directional agreement is captured. Two conditions can have very different residual magnitudes but appear similar here if they point in the same direction. The \u03b3 weights modulate which dimensions are amplified and are shared across conditions."
  }},
  "lm.head": {{
    what: "Linear projection from hidden_dim to vocabulary size (151,936 tokens for Qwen3). These are the pre-softmax logits. The weight matrix is typically tied to embed_tokens. The argmax gives the model\u2019s predicted next token; the full vector encodes the distribution over all tokens.",
    why: "The most direct measure of output-level similarity. High similarity = the model is predicting the same token distribution for both conditions. Useful as a top-level sanity check: if lm_head similarity is high but earlier layers differ, both conditions reach the same answer via different circuits. If lm_head similarity is low, the conditions produce fundamentally different outputs.",
    caveat: "lm_head has 151k dimensions. The vast majority will have near-zero contrast between conditions; threshold filtering is critical here. At the Show All preset, lm_head similarity is often artificially high (dominated by shared low-signal dimensions). Use Exploratory or Strict for meaningful results."
  }},
  "input.layernorm": {{
    what: "Pre-attention RMS LayerNorm at each transformer block (model.layers.N.input_layernorm). Applied to the residual stream x before self-attention. Formula: \u03b3 \u00d7 (x / RMS(x)). This is a Pre-LN architecture \u2014 normalisation happens before, not after, each sublayer.",
    why: "Reflects the direction of the residual stream just before attention processes it at block N. Comparing input_layernorm similarity across blocks reveals at which depth the representations diverge between conditions. A sharp drop at block N means that block\u2019s attention or MLP is where the conditions begin to split.",
    caveat: "RMS norm projects to a unit sphere and discards magnitude information. Similarity here is purely directional. Best used comparatively across blocks or against attention_layernorm at the same block, not in isolation."
  }},
  "attention.layernorm": {{
    what: "Pre-MLP RMS LayerNorm (post_attention_layernorm in HuggingFace). Applied to the residual stream after the attention output has been added: residual = x + attn_out. Mathematically identical to input_layernorm but at a later stage in the same block.",
    why: "Represents the residual state between attention and MLP in the same block. Comparing input_layernorm vs attention_layernorm similarity at the same block isolates the source of divergence: if attention_layernorm \u2248 input_layernorm, attention is not causing the difference and the MLP is responsible. If attention_layernorm < input_layernorm, attention is where the conditions diverge.",
    caveat: "Only meaningful when compared to input_layernorm at the same block. In isolation it has the same interpretation caveats as input_layernorm: directional only, magnitude erased."
  }},
  "q.proj": {{
    what: "Linear projection from hidden_dim to n_heads \u00d7 head_dim query vectors. For Qwen3-8B: 32 heads \u00d7 128 dim = 4096 total dimensions. These vectors dot-product with keys to compute attention scores, determining which positions each head attends to.",
    why: "Query similarity reflects whether the model is searching for the same types of context under both conditions. High similarity at a given block means attention routing is similar for both conditions at that depth. However, attention patterns also depend on keys (from other positions), so q_proj alone does not fully determine where attention goes.",
    caveat: "In Qwen3, q_proj output passes through q_norm before attention scores are computed. Q_proj similarity therefore does not directly translate to attention score similarity \u2014 look at q_norm for the normalised picture. Also note GQA: 32 Q heads vs only 8 KV heads in Qwen3-8B."
  }},
  "k.proj": {{
    what: "Linear projection from hidden_dim to n_kv_heads \u00d7 head_dim key vectors. For Qwen3-8B: 8 KV heads \u00d7 128 dim = 1024 total dimensions (Grouped Query Attention \u2014 4\u00d7 fewer key heads than query heads). Keys determine which context positions advertise themselves as worth attending to.",
    why: "Key similarity reflects whether the same context positions stand out as important under both conditions. Combined with q_proj similarity, this gives a picture of whether attention routing is shared. Lower dimensionality than q_proj means fewer surviving dimensions after thresholding.",
    caveat: "GQA means k_proj has only 1024 dimensions (vs 4096 for q_proj), so thresholding removes a higher fraction of the vector. Interpret k_proj together with q_proj and q_norm/k_norm for the full attention-routing picture."
  }},
  "v.proj": {{
    what: "Linear projection from hidden_dim to n_kv_heads \u00d7 head_dim value vectors (same GQA structure as k_proj: 1024 dims for Qwen3-8B). Values are the content retrieved and mixed into the attending token\u2019s representation when a position is attended to.",
    why: "Value similarity reflects what information attention retrieves from context. High similarity = both conditions pull similar content from the context tokens. More semantically meaningful than q/k similarity for faithfulness tasks because it reflects what the model reads from the premise rather than just where it looks.",
    caveat: "v_proj captures the content side of attention (\u2018what\u2019); q/k capture the routing side (\u2018where\u2019). For faithfulness analysis, v_proj similarity is often more diagnostic than q_proj or k_proj because it directly reflects which information from the premise is being read."
  }},
  "o.proj": {{
    what: "Projects the concatenated multi-head attention output (n_heads \u00d7 head_dim = 4096 dims for Qwen3-8B) back to hidden_dim (3584). This vector is added to the residual stream as the attention sublayer\u2019s contribution: residual += o_proj(concat(head_outputs)).",
    why: "o_proj is the net write from attention to the residual stream. Directly analogous to down_proj for the MLP pathway. High similarity = attention contributes the same direction to the residual for both conditions, regardless of internal head-level details.",
    caveat: "o_proj similarity reflects both what was attended to AND how the head outputs were combined and projected. A difference here can arise from different attention patterns, different value content, or both."
  }},
  "q.norm": {{
    what: "Per-head RMS normalisation applied to query vectors after q_proj (Qwen3-specific; absent in Qwen2). Each of the 32 attention heads gets its queries normalised to unit norm independently. This makes attention scores invariant to query magnitude and stabilises training at high learning rates.",
    why: "q_norm similarity reflects whether the query directions (magnitude removed) are aligned between conditions. If q_norm similarity is high but q_proj similarity is low, the two conditions produce queries of different magnitudes but pointing in the same direction \u2014 attention patterns will still be similar. Interpret q_norm and k_norm together for the clearest picture of attention routing.",
    caveat: "q_norm and k_norm were introduced in Qwen3 for training stability. Read them as a pair: high q_norm + high k_norm strongly predicts similar attention patterns; high q_norm + low k_norm means the key side is driving any routing differences."
  }},
  "k.norm": {{
    what: "Per-head RMS normalisation applied to key vectors after k_proj (Qwen3-specific). Mirrors q_norm on the key side. Together, q_norm and k_norm normalise both sides of the attention dot product, making attention scores invariant to both query and key magnitudes.",
    why: "k_norm similarity complements q_norm. Together they determine how similar the effective attention logits are between conditions. If both are high, attention patterns are likely nearly identical. If only one is high, the divergence comes from that specific side.",
    caveat: "k_norm has only 1024 dimensions (8 KV heads \u00d7 128 for Qwen3-8B), much smaller than q_norm (4096). The smaller dimensionality makes it noisier at strict thresholds."
  }},
  "gate.proj": {{
    what: "One of two parallel linear projections in the SwiGLU FFN. Projects hidden_dim to intermediate_dim (~11008 for Qwen3-8B). The gate branch output goes through SiLU to become act_fn, which controls which intermediate neurons activate.",
    why: "gate_proj values before SiLU are less interpretable than act_fn because the non-linearity can invert their meaning near zero. Use act_fn for mechanistic conclusions. gate_proj similarity gives a pre-activation snapshot of the gating decision, useful when comparing against act_fn to understand how the non-linearity reshapes the pattern.",
    caveat: "A small change in gate_proj near zero can flip a neuron from inactive to active after SiLU. Gate_proj similarity does not translate linearly to act_fn similarity. Always prefer act_fn for circuit-level analysis."
  }},
  "act.fn": {{
    what: "SiLU(gate_proj(x)) \u2014 the gate projection after the SiLU activation function. SiLU(x) = x \u00b7 \u03c3(x): approximately 0 for negative x, approximately x for large positive x. This is the neuron on/off signal. The full MLP output is: down_proj(act_fn(gate_proj(x)) \u2299 up_proj(x)).",
    why: "The most interpretable MLP layer and the single best proxy for which features fire. A positive value here means the neuron is effectively on. High similarity = the same intermediate features activate for both conditions at this layer. Start here for any cross-lingual circuit analysis. If act_fn similarity stays high at the Strict preset, the shared circuit is robust and likely replicable.",
    caveat: "act_fn has intermediate_dim dimensions (~11008 for Qwen3-8B SwiGLU). At Strict thresholds, very few neurons survive. If similarity shows as 0.000 (dark cell), it may mean no neurons passed the threshold for one or both conditions \u2014 try the Exploratory preset."
  }},
  "up.proj": {{
    what: "The second parallel branch of the SwiGLU FFN. Projects hidden_dim to intermediate_dim. The complete MLP output is down_proj(act_fn(gate_proj(x)) \u2299 up_proj(x)), so up_proj provides the content that the gate selects.",
    why: "up_proj similarity reflects whether both conditions want to write similar content to the residual, independent of whether the gate fires. Comparing up_proj vs act_fn similarity reveals whether divergence comes from the gating decision (act_fn differs) or from what the MLP wants to write (up_proj differs).",
    caveat: "up_proj output is element-wise multiplied with act_fn before down_proj. High up_proj similarity does not mean the same neurons dominate the final output \u2014 the gate still determines which positions survive. For output-level conclusions, prefer down_proj."
  }},
  "down.proj": {{
    what: "Projects intermediate_dim back to hidden_dim. Takes (act_fn \u2299 up_proj) as input. This vector is added to the residual stream as the MLP sublayer\u2019s contribution, directly analogous to o_proj on the attention side.",
    why: "down_proj is the net MLP write to the residual stream. High similarity = the MLP contributes the same direction for both conditions. If down_proj similarity is high but act_fn similarity is low, different neurons are producing functionally equivalent outputs \u2014 a sign of superposition or distributed representations.",
    caveat: "down_proj maps through a full linear transformation, so different act_fn patterns can produce similar down_proj outputs if the weight matrix aligns them. High down_proj + low act_fn is an indicator of functional equivalence via different means, worth investigating further."
  }}
}};

// --- Build glossary modal ---
const glossaryContent = document.getElementById('glossaryContent');
GLOSSARY_GROUPS.forEach(group => {{
  const section = document.createElement('div');
  section.className = 'glossary-group';
  const title = document.createElement('div');
  title.className = 'glossary-group-title';
  title.textContent = group.title;
  section.appendChild(title);
  group.entries.forEach(e => {{
    const row = document.createElement('div');
    row.className = 'glossary-entry';
    const badge = document.createElement('span');
    badge.className = 'g-type';
    badge.textContent = '\u2026' + e.type;
    const right = document.createElement('div');
    right.className = 'g-right';
    const fullEl = document.createElement('div');
    fullEl.className = 'g-full';
    fullEl.textContent = e.full;
    const descEl = document.createElement('div');
    descEl.className = 'g-desc';
    descEl.textContent = e.desc;
    right.appendChild(fullEl);
    right.appendChild(descEl);
    const main = document.createElement('div');
    main.className = 'entry-main';
    main.appendChild(badge);
    main.appendChild(right);
    row.appendChild(main);
    const detail = GLOSSARY_DETAILS[e.type];
    if (detail) {{
      const toggle = document.createElement('button');
      toggle.className = 'detail-toggle';
      toggle.textContent = '\u25b6 More details';
      right.appendChild(toggle);
      const panel = document.createElement('div');
      panel.className = 'detail-panel';
      [['What it computes', detail.what], ['Why it matters', detail.why], ['Caveats', detail.caveat]].forEach(([hdr, txt]) => {{
        const sec = document.createElement('div');
        sec.className = 'detail-section';
        const h = document.createElement('div');
        h.className = 'detail-section-title';
        h.textContent = hdr;
        const p = document.createElement('div');
        p.className = 'detail-text';
        p.textContent = txt;
        sec.appendChild(h);
        sec.appendChild(p);
        panel.appendChild(sec);
      }});
      const toggleFn = () => {{
        const open = panel.classList.toggle('open');
        toggle.textContent = open ? '\u25bc Less' : '\u25b6 More details';
      }};
      toggle.addEventListener('click', evt => {{ evt.stopPropagation(); toggleFn(); }});
      row.style.cursor = 'pointer';
      row.addEventListener('click', toggleFn);
      row.appendChild(panel);
    }}
    section.appendChild(row);
  }});
  glossaryContent.appendChild(section);
}});

// --- Guide modal open/close ---
document.getElementById('guideBtn').addEventListener('click', () =>
  document.getElementById('guideModal').classList.add('open'));
document.getElementById('guideClose').addEventListener('click', () =>
  document.getElementById('guideModal').classList.remove('open'));
document.getElementById('guideModal').addEventListener('click', e => {{
  if (e.target === document.getElementById('guideModal'))
    document.getElementById('guideModal').classList.remove('open');
}});

// --- Recommendations modal content ---
const SIM_SCALE = [
  {{ value: '+1.0', bg: 'rgb(0,0,255)', fg: '#fff',
    desc: 'Identical direction — same neurons fire with same sign in both conditions. Circuits are fully shared.' }},
  {{ value: '+0.7', bg: 'rgb(77,77,255)', fg: '#fff',
    desc: 'Strong overlap — most significant neurons agree. Likely a shared mechanism with minor condition-specific variation.' }},
  {{ value: '~0.0', bg: 'rgb(255,255,255)', fg: '#111',
    desc: 'No overlap — different neurons selected by threshold, or selected neurons are orthogonal. Try a lower threshold.' }},
  {{ value: '−0.7', bg: 'rgb(255,77,77)', fg: '#111',
    desc: 'Opposite directions — same neurons fire but with reversed sign. The effect is present but inverted.' }},
  {{ value: '−1.0', bg: 'rgb(255,0,0)', fg: '#fff',
    desc: 'Completely opposite — circuits are actively anti-correlated between the two conditions.' }},
];

const LAYER_RECS = [
  {{ type: 'act.fn',
    rec: 'Best proxy for shared MLP circuits. High similarity = same intermediate features fire. Start here for faithfulness/fact-checking.' }},
  {{ type: 'lm.head',
    rec: 'Do both conditions predict the same tokens? High similarity = language-agnostic output distribution. Check this when ablation results already look similar.' }},
  {{ type: 'down.proj / o.proj',
    rec: '\u2018Write\u2019 operations — what MLP/attention contributes to the residual. High similarity = same information is written back, even if via different intermediate steps.' }},
  {{ type: 'q.proj / k.proj',
    rec: 'Routing — do both conditions attend to the same positions? Lower similarity is expected here; it reflects context, not the core reasoning mechanism.' }},
  {{ type: 'input.layernorm / attention.layernorm',
    rec: 'Least informative for mechanism analysis. Mostly reflects residual stream direction, not a specific computation. Use for sanity checks only.' }},
];

const PRESET_RECS = [
  {{ name: 'Show All',    desc: 'Baseline — all neurons contribute. Noisy: weak and random effects dominate, so similarity is inflated. Use only as a reference point.' }},
  {{ name: 'Exploratory', desc: 'Start here. Filters to neurons with clear statistical signal. Use to identify candidate layers before going stricter.' }},
  {{ name: 'Strict',      desc: 'Only the most robust neurons. If similarity stays high here, the shared circuit is strong and likely replicable across runs.' }},
];

const CONDITIONS = [
  {{
    title: 'High similarity in act.fn across languages',
    steps: [
      'The same MLP features drive the effect cross-lingually — this is the best evidence for a shared circuit.',
      'Good candidate for language-agnostic patching: use a single neuron set across all languages.',
      'Open interactive_stats.html for this run to see exactly which neurons are significant.',
      'Run the patching sweep with the shared neuron set and compare \u0394 across languages.',
    ]
  }},
  {{
    title: 'Low similarity everywhere (even at Show All)',
    steps: [
      'Each language uses different circuits for this task.',
      'Patch per-language separately rather than using a shared neuron set.',
      'Check that the tasks are truly comparable: same prompt structure, same label distribution.',
      'Consider whether the baseline model treats the languages differently (check lm.head similarity first).',
    ]
  }},
  {{
    title: 'High in lm.head but low in act.fn',
    steps: [
      'Languages converge on the same output distribution but through different intermediate circuits.',
      'The \u2018what\u2019 is shared but the \u2018how\u2019 differs \u2014 patching lm.head neurons may generalize; act.fn likely won\u2019t.',
      'Check whether the difference is early (embed, low layers) or late (high layers) using the layer filter.',
    ]
  }},
  {{
    title: 'High similarity only at Show All, drops at Exploratory/Strict',
    steps: [
      'Overlap is driven by weak/small effects, not the statistically robust neurons.',
      'Core circuits are likely language-specific.',
      'Focus analysis on layers that maintain similarity at Exploratory or higher.',
    ]
  }},
];

const WORKFLOW = [
  'Set <strong>Exploratory</strong> preset. Look at the Mean row to get a quick cross-language overview.',
  'Filter to <strong>act.fn</strong>. Find layers with consistently blue cells — these are the best patching candidates.',
  'Switch to <strong>Strict</strong> preset. Layers that stay blue here have the most robust shared circuits.',
  'For high-similarity layers, open <strong>interactive_stats.html</strong> for that run to inspect the full statistical map and identify specific neurons.',
  'Run the patching sweep using the shared neuron set. Compare \u0394 performance across languages to confirm cross-lingual generalization.',
  'If results are weak, fall back to <strong>per-language patching</strong> and compare scale factors.',
];

function buildRecModal() {{
  // -- Interpret tab --
  const interp = document.getElementById('tab-interpret');

  function addSection(container, title) {{
    const s = document.createElement('div');
    s.className = 'rec-section';
    const h = document.createElement('div');
    h.className = 'rec-section-title';
    h.textContent = title;
    s.appendChild(h);
    container.appendChild(s);
    return s;
  }}

  const s1 = addSection(interp, 'Reading similarity values');
  SIM_SCALE.forEach(e => {{
    const row = document.createElement('div');
    row.className = 'sim-row';
    const badge = document.createElement('span');
    badge.className = 'sim-badge';
    badge.style.cssText = `background:${{e.bg}};color:${{e.fg}}`;
    badge.textContent = e.value;
    const desc = document.createElement('span');
    desc.style.color = '#ccc';
    desc.textContent = e.desc;
    row.appendChild(badge);
    row.appendChild(desc);
    s1.appendChild(row);
  }});

  const s2 = addSection(interp, 'Which layers reveal what');
  LAYER_RECS.forEach(e => {{
    const row = document.createElement('div');
    row.className = 'layer-row2';
    const badge = document.createElement('span');
    badge.className = 'g-type';
    badge.style.whiteSpace = 'nowrap';
    badge.textContent = '\u2026' + e.type;
    const desc = document.createElement('span');
    desc.style.color = '#ccc';
    desc.textContent = e.rec;
    row.appendChild(badge);
    row.appendChild(desc);
    s2.appendChild(row);
  }});

  const s3 = addSection(interp, 'Threshold guidance');
  PRESET_RECS.forEach(e => {{
    const row = document.createElement('div');
    row.className = 'preset-row';
    const badge = document.createElement('span');
    badge.className = 'preset-badge';
    badge.textContent = e.name;
    const desc = document.createElement('span');
    desc.style.color = '#ccc';
    desc.textContent = e.desc;
    row.appendChild(badge);
    row.appendChild(desc);
    s3.appendChild(row);
  }});

  // -- What to do next tab --
  const whatnext = document.getElementById('tab-whatnext');

  const s4 = addSection(whatnext, 'Based on what you see');
  CONDITIONS.forEach(c => {{
    const block = document.createElement('div');
    block.className = 'condition-block';
    const title = document.createElement('div');
    title.className = 'condition-title';
    title.textContent = '\u2192 ' + c.title;
    const ul = document.createElement('ul');
    ul.className = 'condition-steps';
    c.steps.forEach(step => {{
      const li = document.createElement('li');
      li.textContent = step;
      ul.appendChild(li);
    }});
    block.appendChild(title);
    block.appendChild(ul);
    s4.appendChild(block);
  }});

  const s5 = addSection(whatnext, 'Recommended workflow');
  const ol = document.createElement('ol');
  ol.className = 'workflow-steps';
  WORKFLOW.forEach(step => {{
    const li = document.createElement('li');
    li.innerHTML = step;
    ol.appendChild(li);
  }});
  s5.appendChild(ol);
}}

buildRecModal();

// Recommendations modal open/close
document.getElementById('recBtn').addEventListener('click', () =>
  document.getElementById('recModal').classList.add('open'));
document.getElementById('recClose').addEventListener('click', () =>
  document.getElementById('recModal').classList.remove('open'));
document.getElementById('recModal').addEventListener('click', e => {{
  if (e.target === document.getElementById('recModal'))
    document.getElementById('recModal').classList.remove('open');
}});

// Tab switching
document.querySelectorAll('.modal-tab').forEach(tab => {{
  tab.addEventListener('click', () => {{
    const modal = tab.closest('.modal-box');
    modal.querySelectorAll('.modal-tab').forEach(t => t.classList.remove('active'));
    modal.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    modal.querySelector(`#tab-${{tab.dataset.tab}}`).classList.add('active');
  }});
}});

document.addEventListener('keydown', e => {{
  if (e.key === 'Escape') {{
    document.getElementById('guideModal').classList.remove('open');
    document.getElementById('recModal').classList.remove('open');
  }}
}});

// --- Utility ---
function findNearest(arr, val) {{
  return arr.reduce((best, v, i) =>
    Math.abs(v - val) < Math.abs(arr[best] - val) ? i : best, 0);
}}

function simToColor(sim) {{
  if (sim >= 0) {{
    const v = Math.round((1 - sim) * 255);
    return `rgb(${{v}},${{v}},255)`;
  }} else {{
    const v = Math.round((1 + sim) * 255);
    return `rgb(255,${{v}},${{v}})`;
  }}
}}

function textColor(sim) {{
  return Math.abs(sim) > 0.7 ? '#fff' : '#111';
}}

function overlapToColor(v) {{
  const c = Math.round((1 - v) * 255);
  return `rgb(${{c}},${{c}},255)`;
}}

let currentMetricKey = 'data_cosine';

const METRIC_DESCS = {{
  data_cosine:  'direction of masked t-stats (−1 to +1)',
  data_pearson: 'correlation on shared active neurons (−1 to +1)',
  data_overlap: 'Jaccard overlap of active neurons (0 to +1)',
}};

function isOverlapMetric() {{
  return currentMetricKey === 'data_overlap';
}}

function valueToColor(v) {{
  return isOverlapMetric() ? overlapToColor(v) : simToColor(v);
}}

function valueToText(v) {{
  return Math.abs(v) > 0.6 ? '#fff' : '#111';
}}

function updateLegend() {{
  const bar = document.querySelector('.legend-bar');
  if (isOverlapMetric()) {{
    bar.style.background = 'linear-gradient(to right, #ffffff, #4444ff)';
    document.getElementById('legendLeft').textContent = '0 (no overlap)';
    document.getElementById('legendRight').textContent = '+1 (full overlap)';
  }} else {{
    bar.style.background = 'linear-gradient(to right, #ff4444, #ffffff, #4444ff)';
    document.getElementById('legendLeft').textContent = '−1 (dissimilar)';
    document.getElementById('legendRight').textContent = '+1 (identical direction)';
  }}
}}

// Cache DOM references
const tSlider   = document.getElementById('tSlider');
const dSlider   = document.getElementById('dSlider');
const tDisplay  = document.getElementById('tDisplay');
const dDisplay  = document.getElementById('dDisplay');
const allRows   = Array.from(document.querySelectorAll('.layer-row'));
const simCells  = document.querySelectorAll('.sim-cell');
const meanCells = document.querySelectorAll('.mean-cell');

// --- Filter state ---
const activeFilters = new Set();

function applyFilters() {{
  const filtered = activeFilters.size > 0;
  allRows.forEach(row => {{
    const visible = !filtered || activeFilters.has(row.dataset.type);
    row.style.display = visible ? '' : 'none';
    // Toggle left column display mode
    row.querySelector('.ln-normal').style.display   = filtered ? 'none' : '';
    row.querySelector('.ln-filtered').style.display = filtered ? '' : 'none';
  }});
  updateMeans();
}}

// --- Similarity update ---
let currentPairData = null;

function updateCells(pairData) {{
  currentPairData = pairData;
  simCells.forEach(cell => {{
    const layerIdx = parseInt(cell.parentElement.dataset.layer);
    const pairIdx  = parseInt(cell.dataset.pair);
    const sim      = pairData[simData.pairs[pairIdx]][layerIdx];
    cell.style.backgroundColor = valueToColor(sim);
    cell.style.color = valueToText(sim);
    cell.textContent = sim.toFixed(3);
  }});
  updateMeans();
}}

function updateMeans() {{
  if (!currentPairData) return;
  meanCells.forEach(cell => {{
    const pairIdx  = parseInt(cell.dataset.pair);
    const allSims  = currentPairData[simData.pairs[pairIdx]];
    const visibleSims = allRows
      .filter(r => r.style.display !== 'none')
      .map(r => allSims[parseInt(r.dataset.layer)])
      .filter(s => s !== 0.0);
    const mean = visibleSims.length > 0
      ? visibleSims.reduce((a, b) => a + b, 0) / visibleSims.length : 0.0;
    cell.style.backgroundColor = valueToColor(mean);
    cell.style.color = valueToText(mean);
    cell.textContent = mean.toFixed(3);
  }});
}}

function update() {{
  const tIdx     = findNearest(simData.t_thresholds, parseFloat(tSlider.value));
  const dIdx     = findNearest(simData.d_thresholds, parseFloat(dSlider.value));
  const tNearest = simData.t_thresholds[tIdx];
  const dNearest = simData.d_thresholds[dIdx];

  tDisplay.textContent = tNearest.toFixed(1);
  dDisplay.textContent = dNearest.toFixed(2);

  updateCells(simData[currentMetricKey][`${{tIdx}}_${{dIdx}}`]);

  document.querySelectorAll('.preset-btn').forEach(btn => {{
    const p   = PRESETS[btn.dataset.preset];
    const pti = findNearest(simData.t_thresholds, p.t);
    const pdi = findNearest(simData.d_thresholds, p.cohens);
    btn.classList.toggle('active', pti === tIdx && pdi === dIdx);
  }});
}}

tSlider.addEventListener('input', update);
dSlider.addEventListener('input', update);

// --- Preset buttons ---
const presetsBar = document.getElementById('presetsBar');
Object.entries(PRESETS).forEach(([name, p]) => {{
  const btn = document.createElement('button');
  btn.className = 'preset-btn';
  btn.dataset.preset = name;
  btn.textContent = name;
  btn.addEventListener('click', () => {{
    tSlider.value = simData.t_thresholds[findNearest(simData.t_thresholds, p.t)];
    dSlider.value = simData.d_thresholds[findNearest(simData.d_thresholds, p.cohens)];
    update();
  }});
  presetsBar.appendChild(btn);
}});

// --- Layer type filter buttons ---
const filterBar = document.getElementById('filterBar');
const filterAll = document.getElementById('filterAll');

LAYER_TYPES.forEach(type => {{
  const btn = document.createElement('button');
  btn.className = 'filter-btn';
  btn.dataset.type = type;
  btn.textContent = '\u2026' + type;
  btn.addEventListener('click', () => {{
    if (activeFilters.has(type)) {{
      activeFilters.delete(type);
      btn.classList.remove('active');
    }} else {{
      activeFilters.add(type);
      btn.classList.add('active');
    }}
    filterAll.classList.toggle('active', activeFilters.size === 0);
    applyFilters();
  }});
  filterBar.appendChild(btn);
}});

filterAll.addEventListener('click', () => {{
  activeFilters.clear();
  document.querySelectorAll('.filter-btn:not(#filterAll)').forEach(b => b.classList.remove('active'));
  filterAll.classList.add('active');
  applyFilters();
}});

// --- Metric tabs ---
document.querySelectorAll('.metric-tab').forEach(tab => {{
  tab.addEventListener('click', () => {{
    document.querySelectorAll('.metric-tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    currentMetricKey = tab.dataset.metric;
    const desc = document.getElementById('metricDesc');
    if (desc) desc.textContent = METRIC_DESCS[currentMetricKey] || '';
    updateLegend();
    update();
  }});
}});

// --- Heatmap popup ---
const popup      = document.getElementById('heatmapPopup');
const popupTitle = document.getElementById('heatmapTitle');
const popupLblA  = document.getElementById('heatmapLabelA');
const popupLblB  = document.getElementById('heatmapLabelB');
const popupFoot  = document.getElementById('heatmapFooter');
const canvasA    = document.getElementById('heatmapA');
const canvasB    = document.getElementById('heatmapB');

function tToRdbu(t, tThresh) {{
  if (Math.abs(t) < tThresh) return [60, 60, 60, 255];
  const v = Math.max(-1, Math.min(1, t / 8));
  if (v >= 0) {{
    const c = Math.round((1 - v) * 255);
    return [c, c, 255, 255];
  }} else {{
    const c = Math.round((1 + v) * 255);
    return [255, c, c, 255];
  }}
}}

function drawHeatmap(canvas, arr, tThresh) {{
  if (!arr || !arr.length) {{ canvas.width = 1; canvas.height = 1; return; }}
  const n = arr.length;
  const W = Math.ceil(Math.sqrt(n));
  const H = Math.ceil(n / W);
  canvas.width  = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(W, H);
  for (let i = 0; i < W * H; i++) {{
    const t = i < n ? arr[i] : 0;
    const [r, g, b, a] = tToRdbu(t, tThresh);
    img.data[i * 4]     = r;
    img.data[i * 4 + 1] = g;
    img.data[i * 4 + 2] = b;
    img.data[i * 4 + 3] = a;
  }}
  ctx.putImageData(img, 0, 0);
}}

function positionPopup(evt) {{
  const pw = popup.offsetWidth  || 420;
  const ph = popup.offsetHeight || 260;
  let x = evt.clientX + 16;
  let y = evt.clientY + 16;
  if (x + pw > window.innerWidth  - 8) x = evt.clientX - pw - 16;
  if (y + ph > window.innerHeight - 8) y = evt.clientY - ph - 16;
  popup.style.left = x + 'px';
  popup.style.top  = y + 'px';
}}

function showHeatmap(cell, evt) {{
  const layerIdx  = parseInt(cell.parentElement.dataset.layer);
  const pairIdx   = parseInt(cell.dataset.pair);
  const pairKey   = simData.pairs[pairIdx];
  const layerName = simData.layers[layerIdx];
  const parts     = pairKey.split('↔');
  const labelA    = parts[0];
  const labelB    = parts[1];
  const tThresh   = parseFloat(tSlider.value);

  const vA = ((simData.vectors_t || {{}})[labelA] || {{}})[layerName] || [];
  const vB = ((simData.vectors_t || {{}})[labelB] || {{}})[layerName] || [];

  popupTitle.textContent = layerName + '  ·  sim = ' + cell.textContent;
  popupLblA.textContent  = labelA;
  popupLblB.textContent  = labelB;
  drawHeatmap(canvasA, vA, tThresh);
  drawHeatmap(canvasB, vB, tThresh);
  popupFoot.textContent  = `${{vA.length || vB.length}} dims shown  ·  gray = |t| < ${{tThresh.toFixed(1)}}`;

  popup.style.display = 'block';
  positionPopup(evt);
}}

simCells.forEach(cell => {{
  cell.addEventListener('mouseenter', evt => showHeatmap(cell, evt));
  cell.addEventListener('mousemove',  positionPopup);
  cell.addEventListener('mouseleave', () => {{ popup.style.display = 'none'; }});
}});

update();
</script>
</body>
</html>"""
