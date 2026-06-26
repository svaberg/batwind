"""Per-file 2D slice pipeline for `batwind-pipe` (minimal, user-serviceable)."""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm

from batwind.constants import B_R_SYMLOG_LINTHRESH_T
from batwind.pipelines.utils import output_prefix_from_input_file
from batwind.smart_ds import SmartDs
from batwind.visualisation.slice import plot_xz_slice_tripcolor_with_cross_quantiles

log = logging.getLogger(__name__)
# Method for recording structured, machine-ingested pipeline payloads.
add_record = logging.getLogger(f"recorder.{__name__}").debug
BR_CMAP = "RdBu_r"


def process_plt_file(file_path: str | Path) -> None:
    """Process one `.plt` file into 2D rho/u/b/br slice PNGs."""
    # Start: resolve input/output paths and log the file being processed.
    stage_start = perf_counter()
    log.info("Resolving slice pipeline paths...")
    path = Path(file_path)
    output_dir = path.parent / "slice"
    log.info("%s", path.name)
    log.debug("Resolving slice pipeline paths complete in %.2f s.", perf_counter() - stage_start)

    # Start: load the dataset and attach the graph-backed derived fields it needs.
    stage_start = perf_counter()
    log.info("Loading and preparing slice dataset...")
    smart_ds = SmartDs.from_file(path, batsrus=True, spherical=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_prefix_from_input_file(path.name)
    log.debug("Loading and preparing slice dataset complete in %.2f s.", perf_counter() - stage_start)

    # Start: make, save, and record the density slice.
    stage_start = perf_counter()
    log.info("Computing density slice...")
    fig, _axes, _cbar = plot_xz_slice_tripcolor_with_cross_quantiles(
        smart_ds,
        var="Rho [kg/m^3]",
        tripcolor_kwargs={"shading": "flat", "norm": LogNorm()},
    )
    out_path = output_dir / f"{prefix}.slices.rho.png"
    fig.savefig(out_path)
    plt.close(fig)
    add_record("slice_rho_png %r", str(out_path.relative_to(path.parent)))
    log.debug("Computing density slice complete in %.2f s.", perf_counter() - stage_start)

    # Start: make, save, and record the speed slice.
    stage_start = perf_counter()
    log.info("Computing speed slice...")
    fig, _axes, _cbar = plot_xz_slice_tripcolor_with_cross_quantiles(
        smart_ds,
        var="U [m/s]",
        tripcolor_kwargs={"shading": "flat"},
    )
    out_path = output_dir / f"{prefix}.slices.u.png"
    fig.savefig(out_path)
    plt.close(fig)
    add_record("slice_u_png %r", str(out_path.relative_to(path.parent)))
    log.debug("Computing speed slice complete in %.2f s.", perf_counter() - stage_start)

    # Start: make, save, and record the magnetic-field slice.
    stage_start = perf_counter()
    log.info("Computing magnetic-field slice...")
    fig, _axes, _cbar = plot_xz_slice_tripcolor_with_cross_quantiles(
        smart_ds,
        var="B [T]",
        tripcolor_kwargs={"shading": "flat", "norm": LogNorm()},
    )
    out_path = output_dir / f"{prefix}.slices.b.png"
    fig.savefig(out_path)
    plt.close(fig)
    add_record("slice_b_png %r", str(out_path.relative_to(path.parent)))
    log.debug("Computing magnetic-field slice complete in %.2f s.", perf_counter() - stage_start)

    # Start: make, save, and record the radial magnetic-field slice.
    stage_start = perf_counter()
    log.info("Computing radial magnetic-field slice...")
    fig, _axes, _cbar = plot_xz_slice_tripcolor_with_cross_quantiles(
        smart_ds,
        var="B_r [T]",
        tripcolor_kwargs={
            "shading": "flat",
            "cmap": BR_CMAP,
            "norm": SymLogNorm(linthresh=B_R_SYMLOG_LINTHRESH_T, base=10),
        },
    )
    out_path = output_dir / f"{prefix}.slices.br.png"
    fig.savefig(out_path)
    plt.close(fig)
    add_record("slice_br_png %r", str(out_path.relative_to(path.parent)))
    log.debug("Computing radial magnetic-field slice complete in %.2f s.", perf_counter() - stage_start)
