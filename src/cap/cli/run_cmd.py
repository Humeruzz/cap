from __future__ import annotations

from pathlib import Path

import typer


def run(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model"),
    output: Path = typer.Option(..., "--output"),
    data: Path | None = typer.Option(None, "--data"),
    text_column: str | None = typer.Option(None, "--text-column"),
    label_column: str | None = typer.Option(None, "--label-column"),
    group_a: Path | None = typer.Option(None, "--group-a"),
    group_b: Path | None = typer.Option(None, "--group-b"),
    type_: str | None = typer.Option(None, "--type"),
    prompt: str | None = typer.Option(None, "--prompt"),
    n_samples: int = typer.Option(0, "--n-samples"),
    skip_viz: bool = typer.Option(False, "--skip-viz"),
    downsample: str = typer.Option(
        "32,64,128",
        "--downsample",
        help="Comma-separated heatmap resolutions for the stats HTML, e.g. 32,64,128.",
    ),
):
    # Delegate to each sub-command directly (not via shell subprocess).
    # cap capture already writes statistics.h5, so there is no separate stats step here;
    # use `cap stats` to recompute statistics with a different test on an existing run.
    from cap.cli._options import parse_resolutions
    from cap.cli.capture_cmd import capture as _capture

    # Validate --downsample up front so a typo fails fast, before the expensive capture.
    parse_resolutions(downsample)

    # capture/visualize are plain functions (not Typer sub-commands), so call them directly
    # with ctx — click's ctx.invoke would not fill the leading `ctx` positional. ctx.obj
    # (seed/device/trust_remote_code) is inherited from the root callback's context.
    typer.echo("[1/3] Capturing activations (writes activations.h5 + statistics.h5)...")
    _capture(
        ctx,
        model=model,
        output=output,
        data=data,
        text_column=text_column,
        label_column=label_column,
        group_a=group_a,
        group_b=group_b,
        type_=type_,
        prompt=prompt,
        n_samples=n_samples,
    )

    if not skip_viz:
        typer.echo("[2/3] Generating visualizations...")
        from cap.cli.visualize_cmd import visualize as _visualize

        _visualize(
            ctx,
            experiment=output,
            experiments=None,
            mode="stats",
            output=None,
            downsample=downsample,
        )

    typer.echo(
        "[3/3] Patch step: run cap patch --experiment <dir> separately with --evaluator or --benchmark."
    )
    typer.echo(f"Done. Results in {output}")
