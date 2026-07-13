from __future__ import annotations

from pathlib import Path

import typer

from cap.cli._manifest import read_manifest


def stats(
    ctx: typer.Context,
    experiment: Path = typer.Option(
        ..., "--experiment", help="Experiment directory written by cap capture."
    ),
):
    manifest = read_manifest(experiment)
    typer.echo(f"Computing statistics for {experiment} (model: {manifest['model']})")

    # Statistics are derived from the saved activations.h5 — no model load required.
    from cap.experiments.capture import CaptureExperiment

    CaptureExperiment.compute_statistics(experiment, model_name=manifest["model"])
    typer.echo(f"Statistics written to {experiment / 'statistics.h5'}")
