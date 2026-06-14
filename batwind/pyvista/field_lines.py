from __future__ import annotations

from os import PathLike

import numpy as np
import pyvista as pv

from batwind.algorithms.sphere_sampling import fibonacci_sphere
from batwind.algorithms.sphere_sampling import PolarAzimuthalGrid
from batwind.pyvista._scalar_bar import readable_scalar_bar_args
from batwind.pyvista.convert import to_unstructured_grid
from batwind.pyvista.fields import radial_component, resolve_body_radius, resolve_magnetic_vector_si
from batwind.smart_ds import SmartDs


_DEFAULT_N_SEEDS = 256
_SEED_RADIUS_SCALE = 1.02
_CLOSED_END_RADIUS_SCALE = 1.2
_DEFAULT_CLOSED_RADIUS_SCALE = 1.1
_MAGNETIC_VECTOR_NAME = "B_vec [T]"


def build_magnetic_field_lines(
    smart_ds: SmartDs,
    *,
    n_seeds: int = _DEFAULT_N_SEEDS,
) -> tuple[pv.UnstructuredGrid, pv.PolyData, pv.PolyData]:
    if int(n_seeds) <= 0:
        raise ValueError("n_seeds must be a positive integer")

    grid, body_radius = magnetic_streamline_state(smart_ds)

    seed_points = _SEED_RADIUS_SCALE * body_radius * np.asarray(
        fibonacci_sphere(int(n_seeds)),
        dtype=float,
    )
    source = pv.PolyData(seed_points)
    lines = trace_magnetic_field_lines(grid, source, body_radius)

    return grid, source, lines


def field_line_max_radius_map(
    smart_ds: SmartDs,
    *,
    n_polar: int,
    n_azimuth: int,
) -> dict[str, np.ndarray]:
    polar, azimuth, solid_angle, radius_map = field_line_max_radius_grid(
        smart_ds,
        n_polar=n_polar,
        n_azimuth=n_azimuth,
    )
    return angular_radius_map_output(polar, azimuth, solid_angle, radius_map, "field_line_max_radius [R]")


def closed_field_line_max_radius_map(
    smart_ds: SmartDs,
    *,
    open_radius: float,
    n_polar: int,
    n_azimuth: int,
) -> dict[str, np.ndarray]:
    polar, azimuth, solid_angle, radius_map = closed_field_line_max_radius_grid(
        smart_ds,
        open_radius=open_radius,
        n_polar=n_polar,
        n_azimuth=n_azimuth,
    )
    return angular_radius_map_output(polar, azimuth, solid_angle, radius_map, "closed_field_line_max_radius [R]")


def angular_radius_map_output(
    polar: np.ndarray,
    azimuth: np.ndarray,
    solid_angle: np.ndarray,
    radius_map: np.ndarray,
    radius_name: str,
) -> dict[str, np.ndarray]:
    return {
        "polar [rad]": polar,
        "azimuth [rad]": azimuth,
        "cell_solid_angle [sr]": solid_angle,
        radius_name: radius_map,
    }


def field_line_max_radius_grid(
    smart_ds: SmartDs,
    *,
    n_polar: int,
    n_azimuth: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid, body_radius = magnetic_streamline_state(smart_ds)
    source, polar, azimuth, solid_angle = angular_seed_source(body_radius, n_polar=n_polar, n_azimuth=n_azimuth)
    lines = trace_magnetic_field_lines(grid, source, body_radius)
    seed_ids = np.asarray(lines.cell_data["SeedIds"], dtype=int)
    cell_max_radius = np.asarray(lines.cell_data["field_line_max_radius [R]"], dtype=float)
    max_radius_map = np.full(source.n_points, np.nan, dtype=float)
    for seed_id in range(int(source.n_points)):
        seed_max_radius = cell_max_radius[seed_ids == seed_id]
        if seed_max_radius.size > 0:
            max_radius_map[seed_id] = float(np.max(seed_max_radius))
    return polar, azimuth, solid_angle, max_radius_map.reshape(polar.shape)


def closed_field_line_max_radius_grid(
    smart_ds: SmartDs,
    *,
    open_radius: float,
    n_polar: int,
    n_azimuth: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if open_radius <= 0.0:
        raise ValueError("open_radius must be positive")

    grid, body_radius = magnetic_streamline_state(smart_ds)
    source, _polar_seed, _azimuth_seed, _solid_angle_seed = angular_seed_source(
        body_radius,
        n_polar=n_polar,
        n_azimuth=n_azimuth,
    )
    lines = trace_magnetic_field_lines(grid, source, body_radius)
    closed_radius = _DEFAULT_CLOSED_RADIUS_SCALE * body_radius
    if closed_radius <= 0.0:
        raise ValueError("closed_radius must be positive")

    _seed_is_open, seed_is_closed, _seed_is_undetermined = seed_topology(
        source,
        lines,
        open_radius=float(open_radius),
        closed_radius=float(closed_radius),
    )
    polar_edges = np.linspace(0.0, np.pi, int(n_polar) + 1)
    azimuth_edges = np.linspace(-np.pi, np.pi, int(n_azimuth) + 1)
    polar = 0.5 * (polar_edges[:-1] + polar_edges[1:])[:, None] * np.ones((1, int(n_azimuth)))
    azimuth = np.ones((int(n_polar), 1)) * 0.5 * (azimuth_edges[:-1] + azimuth_edges[1:])[None, :]
    solid_angle = (np.cos(polar_edges[:-1]) - np.cos(polar_edges[1:]))[:, None] * np.diff(azimuth_edges)[None, :]

    seed_ids = np.asarray(lines.cell_data["SeedIds"], dtype=int)
    closed_cell_ids = np.flatnonzero(np.asarray(seed_is_closed, dtype=bool)[seed_ids])
    if closed_cell_ids.size == 0:
        return polar, azimuth, solid_angle, np.full((n_polar, n_azimuth), np.nan, dtype=float)

    closed_lines = lines.extract_cells(closed_cell_ids)
    points = np.asarray(closed_lines.points, dtype=float)
    radii = np.linalg.norm(points, axis=1)
    valid = np.isfinite(radii) & (radii > 0.0)
    points = points[valid]
    radii = radii[valid]

    polar_ids = np.clip(
        np.searchsorted(
            polar_edges,
            np.arccos(np.clip(points[:, 2] / radii, -1.0, 1.0)),
            side="right",
        ) - 1,
        0,
        int(n_polar) - 1,
    )
    azimuth_ids = np.clip(
        np.searchsorted(
            azimuth_edges,
            np.arctan2(points[:, 1], points[:, 0]),
            side="right",
        ) - 1,
        0,
        int(n_azimuth) - 1,
    )

    closed_max_radius_map = np.full((int(n_polar), int(n_azimuth)), np.nan, dtype=float)
    for polar_id, azimuth_id, radius_value in zip(polar_ids, azimuth_ids, radii):
        current = closed_max_radius_map[polar_id, azimuth_id]
        if not np.isfinite(current) or radius_value > current:
            closed_max_radius_map[polar_id, azimuth_id] = float(radius_value)

    finite = np.isfinite(np.asarray(closed_max_radius_map, dtype=float))
    support = np.zeros_like(finite, dtype=int)
    for delta_polar in (-1, 0, 1):
        for delta_azimuth in (-1, 0, 1):
            support += np.roll(np.roll(finite, delta_polar, axis=0), delta_azimuth, axis=1)
    closed_max_radius_map[support < 3] = np.nan
    return polar, azimuth, solid_angle, closed_max_radius_map.reshape(polar.shape)


def build_field_line_max_radius_surface(
    smart_ds: SmartDs,
    *,
    n_polar: int,
    n_azimuth: int,
) -> pv.PolyData:
    """
    Build the radial surface defined by the field-line max-radius map.

    Each angular cell is seeded just above the stellar surface, the magnetic
    field line through that seed is traced in both directions, and the cell
    radius is set to the largest radius reached by that rooted line.
    """
    _polar, _azimuth, _solid_angle, radius_map = field_line_max_radius_grid(
        smart_ds,
        n_polar=n_polar,
        n_azimuth=n_azimuth,
    )
    return radius_map_surface(radius_map, radius_name="field_line_max_radius [R]")


def build_closed_field_line_max_radius_surface(
    smart_ds: SmartDs,
    *,
    open_radius: float,
    n_polar: int,
    n_azimuth: int,
) -> pv.PolyData:
    """
    Build the closed-line spatial envelope as a separatrix approximation.

    Closed field lines are traced from stellar-surface seeds, then all points on
    the closed subset are binned by spatial direction. The cell radius is the
    largest closed-line radius seen in that direction. Open or unsupported
    directions are omitted.
    """
    _polar, _azimuth, _solid_angle, radius_map = closed_field_line_max_radius_grid(
        smart_ds,
        open_radius=open_radius,
        n_polar=n_polar,
        n_azimuth=n_azimuth,
    )
    return radius_map_surface(radius_map, radius_name="closed_field_line_max_radius [R]")


def magnetic_streamline_state(smart_ds: SmartDs) -> tuple[pv.UnstructuredGrid, float]:
    magnetic = resolve_magnetic_vector_si(smart_ds)
    radial_magnetic_field = np.asarray(smart_ds["B_r [T]"], dtype=float)
    grid = to_unstructured_grid(
        smart_ds,
        point_data={
            _MAGNETIC_VECTOR_NAME: magnetic,
            "B_r [T]": radial_magnetic_field,
        },
    )
    grid.set_active_vectors(_MAGNETIC_VECTOR_NAME)

    body_radius = resolve_body_radius(smart_ds)
    grid.add_field_data(np.array([body_radius], dtype=float), "RBODY [R]")
    return grid, body_radius


def angular_seed_source(
    body_radius: float,
    *,
    n_polar: int,
    n_azimuth: int,
) -> tuple[pv.PolyData, np.ndarray, np.ndarray, np.ndarray]:
    if int(n_polar) <= 0:
        raise ValueError("n_polar must be a positive integer")
    if int(n_azimuth) <= 0:
        raise ValueError("n_azimuth must be a positive integer")

    angular_grid = PolarAzimuthalGrid(
        np.linspace(0.0, np.pi, int(n_polar) + 1),
        np.linspace(-np.pi, np.pi, int(n_azimuth) + 1),
    )
    polar = np.asarray(angular_grid.polar_centres, dtype=float).T
    azimuth = np.asarray(angular_grid.azimuthal_centres, dtype=float).T
    seed_points = (
        _SEED_RADIUS_SCALE
        * body_radius
        * angular_grid.centres_cartesian(radius=1.0).transpose(1, 0, 2).reshape(-1, 3)
    )
    return (
        pv.PolyData(seed_points),
        polar,
        azimuth,
        np.asarray(angular_grid.cell_solid_angle, dtype=float),
    )


def radius_map_surface(cell_radius: np.ndarray, *, radius_name: str) -> pv.PolyData:
    cell_radius = np.asarray(cell_radius, dtype=float)
    if not np.any(np.isfinite(cell_radius)):
        raise ValueError(f"No finite cells were available to build {radius_name}")

    n_polar, n_azimuth = cell_radius.shape
    point_radius = np.full((n_polar + 1, n_azimuth + 1), np.nan, dtype=float)

    top = cell_radius[0, np.isfinite(cell_radius[0])]
    if top.size > 0:
        point_radius[0, :] = float(np.mean(top))
    bottom = cell_radius[-1, np.isfinite(cell_radius[-1])]
    if bottom.size > 0:
        point_radius[-1, :] = float(np.mean(bottom))

    for i in range(1, n_polar):
        for j in range(n_azimuth):
            values = []
            for cell_i in (i - 1, i):
                for cell_j in ((j - 1) % n_azimuth, j):
                    value = cell_radius[cell_i, cell_j]
                    if np.isfinite(value):
                        values.append(float(value))
            if values:
                point_radius[i, j] = float(np.mean(values))

    point_radius[:, -1] = point_radius[:, 0]
    n_polar, n_azimuth = cell_radius.shape
    polar_edges = np.linspace(0.0, np.pi, n_polar + 1)
    azimuth_edges = np.linspace(-np.pi, np.pi, n_azimuth + 1)
    theta, phi = np.meshgrid(polar_edges, azimuth_edges, indexing="ij")
    sin_theta = np.sin(theta)
    x = point_radius * sin_theta * np.cos(phi)
    y = point_radius * sin_theta * np.sin(phi)
    z = point_radius * np.cos(theta)
    points = np.column_stack((x.ravel(order="C"), y.ravel(order="C"), z.ravel(order="C")))

    faces: list[int] = []
    cell_radius_list: list[float] = []
    n_points_az = n_azimuth + 1
    for i in range(n_polar):
        for j in range(n_azimuth):
            radius_value = float(cell_radius[i, j])
            corner_ids = np.array(
                [
                    i * n_points_az + j,
                    (i + 1) * n_points_az + j,
                    (i + 1) * n_points_az + (j + 1),
                    i * n_points_az + (j + 1),
                ],
                dtype=int,
            )
            if not np.isfinite(radius_value):
                continue
            if not np.all(np.isfinite(points[corner_ids])):
                continue
            faces.extend([4, int(corner_ids[0]), int(corner_ids[1]), int(corner_ids[2]), int(corner_ids[3])])
            cell_radius_list.append(radius_value)

    if not cell_radius_list:
        raise ValueError(f"No finite cells remained after building {radius_name}")

    surface = pv.PolyData(points, faces=np.asarray(faces, dtype=np.int64))
    surface.cell_data[radius_name] = np.asarray(cell_radius_list, dtype=float)
    surface.set_active_scalars(radius_name)
    return surface


def trace_magnetic_field_lines(
    grid: pv.UnstructuredGrid,
    source: pv.PolyData,
    body_radius: float,
) -> pv.PolyData:
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
    is_open = np.zeros(lines.n_cells, dtype=bool)
    end_radius = np.zeros(lines.n_cells, dtype=float)
    max_radius = np.zeros(lines.n_cells, dtype=float)
    closed_end_radius = _CLOSED_END_RADIUS_SCALE * body_radius

    cells = np.asarray(lines.lines)
    i = 0
    cell_id = 0
    while i < cells.size:
        n_points = int(cells[i])
        point_ids = cells[i + 1:i + 1 + n_points]
        radii = np.linalg.norm(np.asarray(lines.points[point_ids], dtype=float), axis=1)
        end_radius[cell_id] = float(radii[-1])
        max_radius[cell_id] = float(np.max(radii))
        is_open[cell_id] = end_radius[cell_id] > closed_end_radius
        i += n_points + 1
        cell_id += 1

    lines.cell_data["field_line_is_open"] = is_open
    lines.cell_data["field_line_end_radius [R]"] = end_radius
    lines.cell_data["field_line_max_radius [R]"] = max_radius
    return lines


def plot_magnetic_field_lines(
    smart_ds: SmartDs,
    *,
    n_seeds: int = _DEFAULT_N_SEEDS,
    plot_radius: float,
    open_line_plot_radius: float,
    off_screen: bool = False,
    screenshot: str | PathLike[str] | None = None,
) -> tuple[pv.Plotter, pv.PolyData]:
    if plot_radius <= 0.0:
        raise ValueError("plot_radius must be positive")
    if open_line_plot_radius <= 0.0:
        raise ValueError("open_line_plot_radius must be positive")

    grid, source, lines = build_magnetic_field_lines(smart_ds, n_seeds=n_seeds)
    body_radius = float(np.asarray(grid.field_data["RBODY [R]"]).ravel()[0])
    closed_lines = lines.clip_surface(pv.Sphere(radius=float(plot_radius)), invert=True)
    if closed_lines.n_points == 0 or closed_lines.n_cells == 0:
        raise ValueError(f"No magnetic field lines remained inside r={plot_radius:g}")
    closed_mask = ~np.asarray(closed_lines.cell_data["field_line_is_open"], dtype=bool)
    closed_visible = closed_lines.extract_cells(np.flatnonzero(closed_mask))

    open_radius = min(float(plot_radius), float(open_line_plot_radius))
    open_lines = lines.clip_surface(pv.Sphere(radius=open_radius), invert=True)
    if open_lines.n_points == 0 or open_lines.n_cells == 0:
        raise ValueError(f"No magnetic field lines remained inside r={open_radius:g}")
    open_mask = np.asarray(open_lines.cell_data["field_line_is_open"], dtype=bool)
    open_visible = open_lines.extract_cells(np.flatnonzero(open_mask))

    if closed_visible.n_cells == 0:
        visible_lines = open_visible
    elif open_visible.n_cells == 0:
        visible_lines = closed_visible
    else:
        visible_lines = closed_visible.merge(open_visible)

    star = pv.Sphere(radius=body_radius, theta_resolution=120, phi_resolution=120).sample(grid)

    scale_values = np.concatenate(
        [
            np.asarray(visible_lines.point_data["B_r [T]"], dtype=float).ravel(),
            np.asarray(star.point_data["B_r [T]"], dtype=float).ravel(),
        ]
    )
    finite = np.abs(np.asarray(scale_values, dtype=float).ravel())
    positive = finite[(finite > 0.0) & np.isfinite(finite)]
    linthresh = float(np.quantile(positive, 0.1)) if positive.size else 1.0

    def symlog_transform(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        out = np.zeros_like(values)
        finite_values = np.isfinite(values)
        magnitude = np.abs(values[finite_values])
        transformed = np.where(
            magnitude <= linthresh,
            magnitude / linthresh,
            1.0 + np.log10(magnitude / linthresh),
        )
        out[finite_values] = np.sign(values[finite_values]) * transformed
        out[~finite_values] = np.nan
        return out

    visible_lines.point_data["symlog B_r [arb]"] = symlog_transform(
        np.asarray(visible_lines.point_data["B_r [T]"], dtype=float)
    )
    star.point_data["symlog B_r [arb]"] = symlog_transform(
        np.asarray(star.point_data["B_r [T]"], dtype=float)
    )
    transformed_scale = np.asarray(visible_lines.point_data["symlog B_r [arb]"], dtype=float)
    transformed_scale = np.abs(transformed_scale[np.isfinite(transformed_scale)])
    transformed_star = np.asarray(star.point_data["symlog B_r [arb]"], dtype=float)
    transformed_star = np.abs(transformed_star[np.isfinite(transformed_star)])
    if transformed_scale.size or transformed_star.size:
        vmax = float(np.max(np.concatenate([transformed_scale, transformed_star])))
    else:
        vmax = 1.0
    annotations = {0.0: "0"}
    if positive.size:
        max_abs = float(np.max(positive))
        max_power = int(np.ceil(np.log10(max_abs / linthresh))) if max_abs > linthresh else 0
        for power in range(max_power + 1):
            tick = linthresh * (10.0 ** power)
            for signed_tick in (-tick, tick):
                transformed_tick = float(symlog_transform(np.array([signed_tick], dtype=float))[0])
                annotations[transformed_tick] = f"{signed_tick:.0e}"

    plotter = pv.Plotter(off_screen=off_screen)

    plotter.add_mesh(
        star,
        scalars="symlog B_r [arb]",
        smooth_shading=True,
        show_scalar_bar=False,
        cmap="RdBu_r",
        clim=(-vmax, vmax),
        annotations=annotations,
    )
    plotter.add_mesh(
        visible_lines,
        scalars="symlog B_r [arb]",
        line_width=3.0,
        render_lines_as_tubes=True,
        scalar_bar_args=readable_scalar_bar_args("B_r [T]"),
        cmap="RdBu_r",
        clim=(-vmax, vmax),
        annotations=annotations,
    )

    if screenshot is not None:
        plotter.show(screenshot=str(screenshot), auto_close=False)
    elif not off_screen:
        plotter.show(auto_close=False)

    return plotter, visible_lines


def open_flux_and_area_fractions(
    smart_ds: SmartDs,
    *,
    open_radius: float,
    n_seeds: int = _DEFAULT_N_SEEDS,
) -> tuple[float, float]:
    if open_radius <= 0.0:
        raise ValueError("open_radius must be positive")

    grid, source, lines = build_magnetic_field_lines(smart_ds, n_seeds=n_seeds)
    body_radius = float(np.asarray(grid.field_data["RBODY [R]"]).ravel()[0])
    closed_radius = _DEFAULT_CLOSED_RADIUS_SCALE * body_radius
    if closed_radius <= 0.0:
        raise ValueError("closed_radius must be positive")

    seed_is_open, _seed_is_closed, _seed_is_undetermined = seed_topology(
        source,
        lines,
        open_radius=float(open_radius),
        closed_radius=float(closed_radius),
    )
    directions = np.asarray(source.points, dtype=float)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    footpoints = pv.PolyData(body_radius * directions).sample(grid)
    abs_br = np.abs(np.asarray(footpoints.point_data["B_r [T]"], dtype=float))

    total_flux_weight = float(np.sum(abs_br))
    open_flux_weight = float(np.sum(abs_br[seed_is_open]))

    return (
        float(open_flux_weight / total_flux_weight) if total_flux_weight > 0.0 else 0.0,
        float(np.mean(seed_is_open)),
    )


def seed_topology(
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


__all__ = [
    "build_magnetic_field_lines",
    "build_closed_field_line_max_radius_surface",
    "build_field_line_max_radius_surface",
    "closed_field_line_max_radius_map",
    "field_line_max_radius_map",
    "open_flux_and_area_fractions",
    "plot_magnetic_field_lines",
]
