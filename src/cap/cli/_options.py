from __future__ import annotations

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
