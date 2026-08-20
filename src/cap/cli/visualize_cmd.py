from __future__ import annotations

from pathlib import Path

import typer

from cap.cli._manifest import read_manifest
from cap.cli._options import parse_experiment_paths, parse_resolutions


def visualize(
    ctx: typer.Context,
    experiment: Path | None = typer.Option(
        None, "--experiment", help="Single experiment dir (for --mode stats)."
    ),
    experiments: list[str] | None = typer.Option(
        None,
        "--experiments",
        help=(
            "Experiment dirs for --mode similarity. Comma-separate them (A,B) or repeat "
            "the flag (--experiments A --experiments B)."
        ),
    ),
    mode: str = typer.Option("stats", "--mode", help="'stats' or 'similarity'."),
    output: Path | None = typer.Option(
        None, "--output", help="Output directory for similarity HTML (--mode similarity only)."
    ),
    downsample: str = typer.Option(
        "32,64,128",
        "--downsample",
        help="Comma-separated heatmap resolutions for the stats HTML, e.g. 32,64,128.",
    ),
):
    if mode == "stats":
        if experiment is None:
            typer.echo("--mode stats requires --experiment.", err=True)
            raise typer.Exit(code=1)
        read_manifest(experiment)  # validates the dir was created by cap capture
        from cap.plot.stats_html import create_interactive_viewer
        from cap.utils.h5_utils import H5Store

        statistics = H5Store.load_statistics(experiment / "statistics.h5")
        if statistics is None:
            typer.echo(f"No statistics.h5 found in {experiment}.", err=True)
            raise typer.Exit(code=1)
        stats_for_html = {
            "t_stats": {layer: statistics[layer]["t_stats"] for layer in statistics},
            "p_values": {layer: statistics[layer]["p_corrected"] for layer in statistics},
            "cohens_d": {layer: statistics[layer]["cohens_d"] for layer in statistics},
            "mean_diff": {layer: statistics[layer]["mean_diff"] for layer in statistics},
            "pooled_std": {layer: statistics[layer]["pooled_std"] for layer in statistics},
        }
        html_path = create_interactive_viewer(
            stats_results=stats_for_html,
            output_path=experiment,
            downsample_resolutions=parse_resolutions(downsample),
        )
        typer.echo(f"Interactive HTML: {html_path}")

    elif mode == "similarity":
        exp_paths = parse_experiment_paths(experiments)
        if len(exp_paths) < 2:
            typer.echo(
                "--mode similarity requires at least two experiment dirs, e.g. "
                "--experiments A,B (or --experiments A --experiments B).",
                err=True,
            )
            raise typer.Exit(code=1)
        from cap.plot.similarity_html import create_similarity_viewer
        from cap.utils.h5_utils import H5Store

        out = output or exp_paths[0].parent / "similarity"
        labeled_stats = []
        for p in exp_paths:
            stats = H5Store.load_statistics(p / "statistics.h5")
            if stats is None:
                typer.echo(f"No statistics.h5 found in {p}.", err=True)
                raise typer.Exit(code=1)
            labeled_stats.append((p.name, stats))
        html_path = create_similarity_viewer(labeled_stats, out)
        typer.echo(f"Similarity HTML: {html_path}")
    else:
        typer.echo(f"Unknown --mode '{mode}'. Choose 'stats' or 'similarity'.", err=True)
        raise typer.Exit(code=1)
