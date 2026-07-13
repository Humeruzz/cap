from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from cap.cli._manifest import read_manifest
from cap.utils.reproducibility import resolve_device


def patch(
    ctx: typer.Context,
    experiment: Path = typer.Option(
        ..., "--experiment", help="Experiment directory from cap capture."
    ),
    evaluator_path: Optional[str] = typer.Option(
        None, "--evaluator", help="Custom evaluator as 'module::ClassName'."
    ),
    benchmark: Optional[str] = typer.Option(
        None,
        "--benchmark",
        help="Built-in benchmark (see BUILTIN_EVALUATORS): gsm8k, mmlu, hellaswag.",
    ),
    d_threshold: float = typer.Option(1.0, "--d-threshold"),
    std_threshold: float = typer.Option(0.1, "--std-threshold"),
    scale: float = typer.Option(0.0, "--scale"),
    n_eval_samples: int = typer.Option(100, "--n-eval-samples"),
):
    manifest = read_manifest(experiment)

    from cap.experiments.patch import PatchingExperiment

    h5_path = experiment / "activations.h5"
    trust = ctx.obj["trust_remote_code"]
    device = resolve_device(ctx.obj["device"])

    patch_exp = PatchingExperiment(
        model_path=manifest["model"],
        h5_path=h5_path,
        trust_remote_code=trust,
        device=device,
    )

    # Both paths produce an EvaluatorProtocol — built-in benchmarks and custom evaluators alike.
    if evaluator_path:
        evaluator = _load_evaluator(evaluator_path)
        benchmarks = []  # custom metric names are unknown to the CLI; full results land in the JSON
    elif benchmark:
        from cap.core.evaluation import BUILTIN_EVALUATORS

        if benchmark not in BUILTIN_EVALUATORS:
            typer.echo(
                f"Unknown --benchmark '{benchmark}'. Choose from: {list(BUILTIN_EVALUATORS)}",
                err=True,
            )
            raise typer.Exit(code=1)
        evaluator = BUILTIN_EVALUATORS[benchmark]()
        benchmarks = [benchmark]  # _BenchmarkEvaluator returns {benchmark: score} → summary column
    else:
        typer.echo("Provide --evaluator or --benchmark.", err=True)
        raise typer.Exit(code=1)

    results = patch_exp.run(
        scale_factors=[scale],
        d_thresholds=[d_threshold],
        std_thresholds=[std_threshold],
        n_eval_samples=n_eval_samples,
        evaluator=evaluator,
        benchmarks=benchmarks,
    )
    typer.echo(f"Results: {results}")


def _load_evaluator(evaluator_path: str):
    """Load a custom evaluator from 'module::ClassName'."""
    import importlib

    if "::" not in evaluator_path:
        raise ValueError("--evaluator must be 'path.to.module::ClassName'")
    module_path, class_name = evaluator_path.split("::", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls()
