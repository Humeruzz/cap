import typer

app = typer.Typer(
    name="cap",
    help="Contrastive Activation Patching — multilingual faithfulness analysis for LLMs.",
    no_args_is_help=True,
)


def version_callback(value: bool) -> None:
    if value:
        import importlib.metadata

        try:
            v = importlib.metadata.version("cap")
        except importlib.metadata.PackageNotFoundError:
            from cap import __version__ as v
        typer.echo(f"cap {v}")
        raise typer.Exit()


# Universal flags attached to root app via callback
@app.callback()
def main(
    ctx: typer.Context,
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility."),
    device: str = typer.Option("auto", "--device", help="Device: cpu, cuda, mps, or auto."),
    trust_remote_code: bool = typer.Option(
        False,
        "--trust-remote-code",
        help="Allow model repos to execute custom code on download. Only enable for trusted sources.",
    ),
    version: bool | None = typer.Option(
        None, "--version", callback=version_callback, is_eager=True, help="Show version and exit."
    ),
):
    ctx.ensure_object(dict)
    ctx.obj["seed"] = seed
    ctx.obj["device"] = device
    ctx.obj["trust_remote_code"] = trust_remote_code


def _register_subcommands():
    # Each command is a single leaf registered directly on the root app so it is invoked
    # as `cap capture` (not `cap capture capture`). The universal --seed/--device/
    # --trust-remote-code flags live on the root callback above and reach each command via ctx.
    from cap.cli.capture_cmd import capture
    from cap.cli.patch_cmd import patch
    from cap.cli.run_cmd import run
    from cap.cli.stats_cmd import stats
    from cap.cli.visualize_cmd import visualize

    app.command("capture", help="Capture activations from two contrastive groups of prompts.")(
        capture
    )
    app.command("stats", help="Compute Welch t-test + FDR statistics on captured activations.")(
        stats
    )
    app.command("patch", help="Patch significant neurons and evaluate on benchmarks.")(patch)
    app.command("visualize", help="Visualize statistics or cross-language cosine similarity.")(
        visualize
    )
    app.command("run", help="Run the full capture → stats → patch → visualize pipeline.")(run)


_register_subcommands()
