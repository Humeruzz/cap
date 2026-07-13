from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer

from cap.data.loaders import LabeledDataset, TwoGroupDataset
from cap.data.prompt_templates import PROMPT_TYPES, PromptTemplate
from cap.utils.reproducibility import set_seed, resolve_device
from cap.cli._manifest import write_manifest
import cap


def capture(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="HuggingFace model name or local path."),
    output: Path = typer.Option(
        ..., "--output", help="Directory to write activations.h5, statistics.h5, manifest.json."
    ),
    # Single-file (label-split) input
    data: Optional[Path] = typer.Option(None, "--data", help="CSV file with text + label columns."),
    text_column: Optional[str] = typer.Option(
        None, "--text-column", help="Column containing text to feed to the model."
    ),
    label_column: Optional[str] = typer.Option(
        None, "--label-column", help="Binary label column (used with --data)."
    ),
    # Two-file input
    group_a: Optional[Path] = typer.Option(
        None, "--group-a", help="CSV file for group A (used with --group-b)."
    ),
    group_b: Optional[Path] = typer.Option(
        None, "--group-b", help="CSV file for group B (used with --group-a)."
    ),
    # Prompt
    type_: Optional[str] = typer.Option(
        None, "--type", help=f"Prompt template type: {list(PROMPT_TYPES.keys())}"
    ),
    prompt: Optional[str] = typer.Option(
        None, "--prompt", help="Custom prompt template string containing {text}."
    ),
    # Sampling
    n_samples: int = typer.Option(0, "--n-samples", help="Max rows per group (0 = all)."),
):
    seed = ctx.obj["seed"]
    device_arg = ctx.obj["device"]
    trust_remote_code = ctx.obj["trust_remote_code"]

    set_seed(seed)
    device = resolve_device(device_arg)

    # Resolve prompt template
    if prompt:
        template = PromptTemplate(prompt)
        template_type = "custom"
    elif type_:
        if type_ not in PROMPT_TYPES:
            typer.echo(
                f"Unknown --type '{type_}'. Choose from: {list(PROMPT_TYPES.keys())}", err=True
            )
            raise typer.Exit(code=1)
        template = PROMPT_TYPES[type_]
        template_type = type_
    else:
        typer.echo("Provide --type or --prompt.", err=True)
        raise typer.Exit(code=1)

    # Resolve dataset
    if data and group_a:
        typer.echo("Provide either --data or --group-a/--group-b, not both.", err=True)
        raise typer.Exit(code=1)

    if data:
        if not text_column or not label_column:
            typer.echo("--data requires --text-column and --label-column.", err=True)
            raise typer.Exit(code=1)
        dataset: LabeledDataset | TwoGroupDataset = LabeledDataset.from_csv(
            data, text_col=text_column, label_col=label_column, n_samples=n_samples, seed=seed
        )
        data_spec = {
            "path": str(data),
            "text_column": text_column,
            "label_column": label_column,
            "n_samples": n_samples,
        }
    elif group_a and group_b:
        if not text_column:
            typer.echo("--group-a/--group-b requires --text-column.", err=True)
            raise typer.Exit(code=1)
        dataset = TwoGroupDataset.from_csv_pair(
            group_a, group_b, text_col=text_column, n_samples=n_samples, seed=seed
        )
        data_spec = {
            "group_a": str(group_a),
            "group_b": str(group_b),
            "text_column": text_column,
            "n_samples": n_samples,
        }
    else:
        typer.echo("Provide --data or --group-a + --group-b.", err=True)
        raise typer.Exit(code=1)

    group_a_texts, group_b_texts = dataset.as_groups()
    typer.echo(f"Group A: {len(group_a_texts)} samples, Group B: {len(group_b_texts)} samples")

    output.mkdir(parents=True, exist_ok=True)

    # Apply prompt template to texts
    group_a_prompted = [template.apply(t) for t in group_a_texts]
    group_b_prompted = [template.apply(t) for t in group_b_texts]

    # Run capture
    from cap.experiments.capture import CaptureExperiment

    exp = CaptureExperiment(
        model_path=model, seed=seed, trust_remote_code=trust_remote_code, device=device
    )
    exp.run_contrast(
        prompts1=group_a_prompted,
        prompts2=group_b_prompted,
        label1="group_a",
        label2="group_b",
        output_path=output,
    )

    # Write manifest
    write_manifest(
        output,
        cap_version=cap.__version__,
        command=sys.argv,
        model=model,
        data_spec=data_spec,
        prompt_template_dict={"type": template_type, "template": template._template},
        seed=seed,
        device=device,
    )
    typer.echo(f"Manifest written to {output / 'manifest.json'}")
