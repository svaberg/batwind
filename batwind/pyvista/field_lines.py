from __future__ import annotations

from os import PathLike

import numpy as np
import pyvista as pv

from batwind.algorithms.sphere_sampling import fibonacci_sphere
from batwind.pyvista._scalar_bar import readable_scalar_bar_args
from batwind.pyvista.convert import DataLike, coerce_smart_ds, to_unstructured_grid
from batwind.pyvista.fields import radial_component, resolve_body_radius, resolve_magnetic_vector_si


_DEFAULT_N_SEEDS = 256
_SEED_RADIUS_SCALE = 1.02
_CLOSED_END_RADIUS_SCALE = 1.2
_DEFAULT_CLOSED_RADIUS_SCALE = 1.1
_LINE_WIDTH = 3.0
_LINE_CMAP = "RdBu_r"
_SCALAR_BAR_ARGS = readable_scalar_bar_args("B_r [T]")
_MAGNETIC_VECTOR_NAME = "B_vec [T]"


def build_magnetic_field_lines(
    data: DataLike,
    *,
    n_seeds: int = _DEFAULT_N_SEEDS,
) -> tuple[pv.UnstructuredGrid, pv.PolyData, pv.PolyData]:
    if int(n_seeds) <= 0:
        raise ValueError("n_seeds must be a positive integer")

    sds = coerce_smart_ds(data)
    magnetic = resolve_magnetic_vector_si(sds)

    grid = to_unstructured_grid(sds, point_data={_MAGNETIC_VECTOR_NAME: magnetic})
    grid.set_active_vectors(_MAGNETIC_VECTOR_NAME)

    body_radius = resolve_body_radius(sds)
    grid.add_field_data(np.array([body_radius], dtype=float), "RBODY [R]")

    seed_points = _SEED_RADIUS_SCALE * body_radius * np.asarray(
        fibonacci_sphere(int(n_seeds)),
        dtype=float,
    )
    source = pv.PolyData(seed_points)

    lines = grid.streamlines_from_source(
        source,
        vectors=_MAGNETIC_VECTOR_NAME,
        integration_direction="both",
        initial_step_length=0.2,
        min_step_length=0.01,
        max_step_length=1.0,
        max_steps=4000,
        max_length=100.0,
        terminal_speed=1e-16,
        compute_vorticity=False,
    )
    if lines.n_points == 0 or lines.n_cells == 0:
        raise ValueError("No magnetic field lines were traced from the stellar-surface seeds")

    lines.point_data["B_r [T]"] = radial_component(
        np.asarray(lines.point_data[_MAGNETIC_VECTOR_NAME], dtype=float),
        np.asarray(lines.points, dtype=float),
    )
    _annotate_field_line_topology(lines, body_radius)

    return grid, source, lines


def plot_magnetic_field_lines(
    data: DataLike,
    *,
    n_seeds: int = _DEFAULT_N_SEEDS,
    plot_radius: float,
    open_line_plot_radius: float,
    plotter: pv.Plotter | None = None,
    off_screen: bool = False,
    show: bool = True,
    screenshot: str | PathLike[str] | None = None,
):
    if plot_radius <= 0.0:
        raise ValueError("plot_radius must be positive")
    if open_line_plot_radius <= 0.0:
        raise ValueError("open_line_plot_radius must be positive")

    grid, source, lines = build_magnetic_field_lines(data, n_seeds=n_seeds)
    body_radius = float(np.asarray(grid.field_data["RBODY [R]"]).ravel()[0])

    visible_lines = _visible_lines(
        lines,
        plot_radius=float(plot_radius),
        open_line_plot_radius=min(float(plot_radius), float(open_line_plot_radius)),
    )
    star = _sample_stellar_surface(grid, body_radius)

    scale_values = np.concatenate(
        [
            np.asarray(visible_lines.point_data["B_r [T]"], dtype=float).ravel(),
            np.asarray(star.point_data["B_r [T]"], dtype=float).ravel(),
        ]
    )
    linthresh, scale_kwargs = _symlog_br_scale(scale_values)
    _set_symlog_br_scalar(visible_lines, linthresh)
    _set_symlog_br_scalar(star, linthresh)

    if plotter is None:
        plotter = pv.Plotter(off_screen=off_screen)

    plotter.add_mesh(
        star,
        scalars="symlog B_r [arb]",
        smooth_shading=True,
        show_scalar_bar=False,
        **scale_kwargs,
    )
    plotter.add_mesh(
        visible_lines,
        scalars="symlog B_r [arb]",
        line_width=_LINE_WIDTH,
        render_lines_as_tubes=True,
        scalar_bar_args=_SCALAR_BAR_ARGS,
        **scale_kwargs,
    )
    plotter.add_axes()
    plotter.add_title("Magnetic field lines colored by B_r (symlog)")

    if show or screenshot is not None:
        plotter.show(screenshot=None if screenshot is None else str(screenshot), auto_close=False)

    return plotter, visible_lines, source, grid


def open_flux_and_area_fractions(
    data: DataLike,
    *,
    open_radius: float,
    n_seeds: int = _DEFAULT_N_SEEDS,
    closed_radius: float | None = None,
) -> dict[str, float]:
    if open_radius <= 0.0:
        raise ValueError("open_radius must be positive")

    grid, source, lines = build_magnetic_field_lines(data, n_seeds=n_seeds)
    body_radius = float(np.asarray(grid.field_data["RBODY [R]"]).ravel()[0])
    if closed_radius is None:
        closed_radius = _DEFAULT_CLOSED_RADIUS_SCALE * body_radius
    if closed_radius <= 0.0:
        raise ValueError("closed_radius must be positive")

    seed_is_open, seed_is_closed, seed_is_undetermined = _seed_topology(
        source,
        lines,
        open_radius=float(open_radius),
        closed_radius=float(closed_radius),
    )
    footpoints = _sample_stellar_footpoints(grid, source, body_radius)
    abs_br = np.abs(np.asarray(footpoints.point_data["B_r [T]"], dtype=float))

    total_flux_weight = float(np.sum(abs_br))
    open_flux_weight = float(np.sum(abs_br[seed_is_open]))
    undetermined_flux_weight = float(np.sum(abs_br[seed_is_undetermined]))

    out = {
        "open_flux_fraction [none]": (
            float(open_flux_weight / total_flux_weight) if total_flux_weight > 0.0 else np.nan
        ),
        "open_area_fraction [none]": float(np.mean(seed_is_open)),
        "undetermined_flux_fraction [none]": (
            float(undetermined_flux_weight / total_flux_weight)
            if total_flux_weight > 0.0
            else np.nan
        ),
        "undetermined_area_fraction [none]": float(np.mean(seed_is_undetermined)),
        "open_count": int(np.count_nonzero(seed_is_open)),
        "closed_count": int(np.count_nonzero(seed_is_closed)),
        "undetermined_count": int(np.count_nonzero(seed_is_undetermined)),
        "n_seeds": int(source.n_points),
        "open_radius [R]": float(open_radius),
        "closed_radius [R]": float(closed_radius),
    }
    return out


def _annotate_field_line_topology(lines: pv.PolyData, body_radius: float) -> None:
    is_open = np.zeros(lines.n_cells, dtype=bool)
    end_radius = np.zeros(lines.n_cells, dtype=float)
    max_radius = np.zeros(lines.n_cells, dtype=float)
    closed_end_radius = _CLOSED_END_RADIUS_SCALE * body_radius

    cells = np.asarray(lines.lines)
    i = 0
    cell_id = 0
    while i < cells.size:
        n_points = int(cells[i])
        point_ids = cells[i + 1 : i + 1 + n_points]
        radii = np.linalg.norm(np.asarray(lines.points[point_ids], dtype=float), axis=1)
        end_radius[cell_id] = float(radii[-1])
        max_radius[cell_id] = float(np.max(radii))
        is_open[cell_id] = end_radius[cell_id] > closed_end_radius
        i += n_points + 1
        cell_id += 1

    lines.cell_data["field_line_is_open"] = is_open
    lines.cell_data["field_line_end_radius [R]"] = end_radius
    lines.cell_data["field_line_max_radius [R]"] = max_radius


def _seed_topology(
    source: pv.PolyData,
    lines: pv.PolyData,
    *,
    open_radius: float,
    closed_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "SeedIds" not in lines.cell_data:
        raise KeyError("Expected SeedIds in streamline cell data")

    seed_ids = np.asarray(lines.cell_data["SeedIds"], dtype=int)
    end_radius = np.asarray(lines.cell_data["field_line_end_radius [R]"], dtype=float)
    n_seeds = int(source.n_points)

    seed_is_open = np.zeros(n_seeds, dtype=bool)
    seed_is_closed = np.zeros(n_seeds, dtype=bool)
    seed_is_undetermined = np.zeros(n_seeds, dtype=bool)

    for seed_id in range(n_seeds):
        mask = seed_ids == seed_id
        seed_end_radius = end_radius[mask]
        if seed_end_radius.size == 0:
            seed_is_undetermined[seed_id] = True
        elif np.any(seed_end_radius > open_radius):
            seed_is_open[seed_id] = True
        elif np.all(seed_end_radius < closed_radius):
            seed_is_closed[seed_id] = True
        else:
            seed_is_undetermined[seed_id] = True

    return seed_is_open, seed_is_closed, seed_is_undetermined


def _clip_lines_to_radius(lines: pv.PolyData, radius: float) -> pv.PolyData:
    clipped = lines.clip_surface(pv.Sphere(radius=radius), invert=True)
    if clipped.n_points == 0 or clipped.n_cells == 0:
        raise ValueError(f"No magnetic field lines remained inside r={radius:g}")
    return clipped


def _visible_lines(lines: pv.PolyData, *, plot_radius: float, open_line_plot_radius: float):
    closed_lines = _clip_lines_to_radius(lines, plot_radius)
    closed_mask = ~np.asarray(closed_lines.cell_data["field_line_is_open"], dtype=bool)
    closed_visible = closed_lines.extract_cells(np.flatnonzero(closed_mask))

    open_lines = _clip_lines_to_radius(lines, open_line_plot_radius)
    open_mask = np.asarray(open_lines.cell_data["field_line_is_open"], dtype=bool)
    open_visible = open_lines.extract_cells(np.flatnonzero(open_mask))

    if closed_visible.n_cells == 0:
        return open_visible
    if open_visible.n_cells == 0:
        return closed_visible
    return closed_visible.merge(open_visible)


def _sample_stellar_surface(grid: pv.UnstructuredGrid, body_radius: float) -> pv.PolyData:
    star = pv.Sphere(radius=body_radius, theta_resolution=120, phi_resolution=120).sample(grid)
    star.point_data["B_r [T]"] = radial_component(
        np.asarray(star.point_data[_MAGNETIC_VECTOR_NAME], dtype=float),
        np.asarray(star.points, dtype=float),
    )
    return star


def _sample_stellar_footpoints(
    grid: pv.UnstructuredGrid,
    source: pv.PolyData,
    body_radius: float,
) -> pv.PolyData:
    directions = np.asarray(source.points, dtype=float)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    footpoints = pv.PolyData(body_radius * directions).sample(grid)
    footpoints.point_data["B_r [T]"] = radial_component(
        np.asarray(footpoints.point_data[_MAGNETIC_VECTOR_NAME], dtype=float),
        np.asarray(footpoints.points, dtype=float),
    )
    return footpoints


def _set_symlog_br_scalar(mesh, linthresh: float) -> None:
    mesh.point_data["symlog B_r [arb]"] = _symlog_transform(
        np.asarray(mesh.point_data["B_r [T]"], dtype=float),
        linthresh,
    )


def _symlog_br_scale(values) -> tuple[float, dict[str, object]]:
    linthresh = _choose_symlog_linthresh(values)
    vmax = _symlog_limit(values, linthresh)
    return linthresh, {
        "cmap": _LINE_CMAP,
        "clim": (-vmax, vmax),
        "annotations": _symlog_annotations(values, linthresh),
    }


def _choose_symlog_linthresh(values) -> float:
    finite = np.abs(np.asarray(values, dtype=float).ravel())
    positive = finite[(finite > 0.0) & np.isfinite(finite)]
    if positive.size == 0:
        return 1.0
    return float(np.quantile(positive, 0.1))


def _symlog_transform(values, linthresh: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.zeros_like(values)
    finite = np.isfinite(values)
    magnitude = np.abs(values[finite])
    transformed = np.where(
        magnitude <= linthresh,
        magnitude / linthresh,
        1.0 + np.log10(magnitude / linthresh),
    )
    out[finite] = np.sign(values[finite]) * transformed
    out[~finite] = np.nan
    return out


def _symlog_limit(values, linthresh: float) -> float:
    transformed = _symlog_transform(values, linthresh)
    finite = np.abs(transformed[np.isfinite(transformed)])
    return float(np.max(finite)) if finite.size else 1.0


def _symlog_annotations(values, linthresh: float) -> dict[float, str]:
    finite = np.abs(np.asarray(values, dtype=float).ravel())
    positive = finite[(finite > 0.0) & np.isfinite(finite)]
    if positive.size == 0:
        return {0.0: "0"}

    max_abs = float(np.max(positive))
    max_power = int(np.ceil(np.log10(max_abs / linthresh))) if max_abs > linthresh else 0

    annotations = {0.0: "0"}
    for power in range(max_power + 1):
        tick = linthresh * (10.0 ** power)
        for signed_tick in (-tick, tick):
            transformed = float(_symlog_transform(np.array([signed_tick]), linthresh)[0])
            annotations[transformed] = f"{signed_tick:.0e}"
    return annotations


__all__ = ["build_magnetic_field_lines", "open_flux_and_area_fractions", "plot_magnetic_field_lines"]
