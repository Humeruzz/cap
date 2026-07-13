from __future__ import annotations

import json
from pathlib import Path


def write_manifest(
    output_path: Path,
    *,
    cap_version: str,
    command: list[str],
    model: str,
    data_spec: dict,
    prompt_template_dict: dict,
    seed: int,
    device: str,
) -> Path:
    manifest = {
        "cap_version": cap_version,
        "command": command,
        "model": model,
        "data": data_spec,
        "prompt_template": prompt_template_dict,
        "seed": seed,
        "device": device,
    }
    path = output_path / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def read_manifest(output_path: Path) -> dict:
    path = output_path / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No manifest.json found in {output_path}. Was this created by cap capture?"
        )
    return json.loads(path.read_text())
