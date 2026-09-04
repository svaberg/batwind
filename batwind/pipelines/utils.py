"""Shared lightweight helpers for pipeline modules."""

from __future__ import annotations

import logging
from pathlib import Path
import re

log = logging.getLogger(__name__)


def slug_key(text: str) -> str:
    """
    Create a filesystem-safe-ish slug from arbitrary text.
    """
    out = []
    for ch in str(text):
        if ch.isalnum():
            out.append(ch.lower())
        else:
            out.append("_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    out = slug.strip("_") or "item"
    log.debug("slug_key '%s' -> '%s'", text, out)
    return out


def output_prefix_from_input_file(input_file) -> str:
    """
    Build a quicklook output prefix from an input filename.
    Used by: `batwind/pipelines/slice.py`, `batwind/pipelines/shell.py`, `batwind/pipelines/volume.py`
    """
    path = Path(str(input_file))
    stem = path.name
    if path.suffix.lower() in {".plt", ".dat"}:
        stem = path.stem
    else:
        stem = Path(stem).stem
    out = slug_key(stem)
    log.debug("output_prefix_from_input_file '%s' -> '%s'", input_file, out)
    return out


def iteration_token_from_path(path: str | Path) -> str | None:
    """
    Extract one BATSRUS iteration token like ``n00109520`` from a path stem.
    """
    stem = Path(str(path)).stem
    match = re.search(r"(?:^|_)n(\d+)(?:[_\.]|$)", stem)
    if match is None:
        return None
    return f"n{match.group(1)}"


def annotate_iteration_axis(axis, path: str | Path) -> None:
    """
    Draw one BATSRUS iteration token inside an axes when the path encodes it.
    """
    token = iteration_token_from_path(path)
    if token is None:
        return
    axis.text(
        0.02,
        0.98,
        token,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "0.4", "alpha": 0.9, "pad": 3.0},
        zorder=1000,
    )
