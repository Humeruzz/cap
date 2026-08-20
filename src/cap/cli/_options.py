from __future__ import annotations

from pathlib import Path

import typer


def parse_resolutions(value: str) -> list[int]:
    """Parse a ``--downsample`` value into a list of integer resolutions.

    Accepts a single token with comma- or space-separated integers, e.g. ``"32,64,128"``
    or ``"32 64 128"`` (quote the latter in the shell). A repeatable option can't be typed
    as ``--downsample 32 64 128`` because the shell splits the tokens before the CLI sees
    them, so we take one string and split it here instead.
    """
    try:
        levels = [int(tok) for tok in value.replace(",", " ").split()]
    except ValueError as exc:
        raise typer.BadParameter(
            f"Expected comma-separated integers like '32,64,128', got {value!r}."
        ) from exc
    if not levels:
        raise typer.BadParameter("Provide at least one resolution, e.g. '32,64,128'.")
    return levels


def parse_experiment_paths(values: list[str] | None) -> list[Path]:
    """Parse ``--experiments`` values into a list of experiment directories.

    The option is repeatable *and* each value may hold comma-separated paths, so all of
    ``--experiments A --experiments B``, ``--experiments A,B``, and
    ``--experiments A,B --experiments C`` work. The comma form mirrors ``--downsample``,
    which is what users reach for first.

    Unlike :func:`parse_resolutions` this does *not* split on whitespace: spaces are legal
    in paths and far more common than commas. A path that genuinely contains a comma has to
    be passed via the repeatable form.
    """
    if not values:
        return []
    return [Path(tok.strip()) for value in values for tok in value.split(",") if tok.strip()]
