from __future__ import annotations

from os import PathLike

import numpy as np
import pyvista as pv

from batwind.pyvista._scalar_bar import readable_scalar_bar_args
from batwind.pyvista.convert import DataLike, coerce_smart_ds, to_unstructured_grid
from batwind.pyvista.fields import radial_component, resolve_density_si, resolve_magnetic_vector_si, resolve_wind_speed_si


_MU0 = 4.0e-7 * np.pi


def build_alfven_surface(
    data: DataLike,
    *,
    mach_level: float = 1.0,
) -> tuple[pv.UnstructuredGrid, pv.PolyData]:
    """
    Build the Alfvén surface ``M_A = mach_level`` and attach wind-speed coloring.
    """
    sds = coerce_smart_ds(data)
    state = _mhd_surface_state(sds)
    density = state["Rho [kg/m^3]"]
    wind_speed = state["U [m/s]"]
    magnetic_strength = state["B [T]"]
    alfven_speed = magnetic_strength / np.sqrt(_MU0 * density)
    with np.errstate(invalid="ignore", divide="ignore"):
        alfven_mach = np.divide(
            wind_speed,
            alfven_speed,
            out=np.full_like(wind_speed, np.nan, dtype=float),
            where=alfven_speed != 0.0,
        )

    grid = to_unstructured_grid(
        sds,
        point_data={
            "U [m/s]": wind_speed,
            "B [T]": magnetic_strength,
            "Rho [kg/m^3]": density,
            "B_r [T]": state["B_r [T]"],
            "c_A [m/s]": alfven_speed,
            "M_A [none]": alfven_mach,
        },
    )
    grid.set_active_scalars("M_A [none]")

    surface = grid.contour([float(mach_level)], scalars="M_A [none]")
    if surface.n_points == 0 or surface.n_cells == 0:
        finite = alfven_mach[np.isfinite(alfven_mach)]
        if finite.size:
            raise ValueError(
                f"No Alfvén surface found for M_A={mach_level:g}; "
                f"finite data range is [{finite.min():.3g}, {finite.max():.3g}]"
            )
        raise ValueError("No finite Alfvén Mach values were available to build a surface")

    surface = surface.compute_normals(point_normals=True, cell_normals=False, inplace=False)
    surface.set_active_scalars("U [m/s]")

    return grid, surface


def build_current_sheet_surface(
    data: DataLike,
    *,
    br_level: float = 0.0,
) -> tuple[pv.UnstructuredGrid, pv.PolyData]:
    """
    Build the current sheet ``B_r = br_level`` and attach MHD fields for coloring.
    """
    sds = coerce_smart_ds(data)
    state = _mhd_surface_state(sds)

    grid = to_unstructured_grid(
        sds,
        point_data={
            "U [m/s]": state["U [m/s]"],
            "B [T]": state["B [T]"],
            "Rho [kg/m^3]": state["Rho [kg/m^3]"],
            "B_r [T]": state["B_r [T]"],
        },
    )
    grid.set_active_scalars("B_r [T]")

    surface = grid.contour([float(br_level)], scalars="B_r [T]")
    if surface.n_points == 0 or surface.n_cells == 0:
        finite = state["B_r [T]"][np.isfinite(state["B_r [T]"])]
        if finite.size:
            raise ValueError(
                f"No current sheet found for B_r={br_level:g}; "
                f"finite data range is [{finite.min():.3g}, {finite.max():.3g}]"
            )
        raise ValueError("No finite radial magnetic-field values were available to build a current sheet")

    surface = surface.compute_normals(point_normals=True, cell_normals=False, inplace=False)
    surface.set_active_scalars("U [m/s]")

    return grid, surface


def current_sheet_orientation(
    data: DataLike,
    *,
    br_level: float = 0.0,
    rmin: float = 0.0,
    rmax: float = 30.0,
    max_points: int = 10000,
) -> dict[str, object]:
    """
    Fit the inner current sheet with an origin-passing plane and report its tilt.

    This follows the old batplotlib convention: fit a plane to ``B_r = 0`` points
    without centering them first, so the best-fit plane is constrained to pass
    through the stellar origin.
    """
    if rmin < 0.0:
        raise ValueError("rmin must be non-negative")
    if rmax <= rmin:
        raise ValueError("rmax must be greater than rmin")
    if int(max_points) <= 0:
        raise ValueError("max_points must be a positive integer")

    _grid, surface = build_current_sheet_surface(data, br_level=br_level)
    fit_points = _current_sheet_fit_points(
        surface,
        rmin=float(rmin),
        rmax=float(rmax),
        max_points=int(max_points),
    )
    normal = _fit_origin_plane_normal(fit_points)
    inclination = float(np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0))))
    return {
        "normal [none]": normal,
        "inclination [deg]": inclination,
        "point_count": int(fit_points.shape[0]),
        "rmin [R]": float(rmin),
        "rmax [R]": float(rmax),
    }


def alfven_surface_averages(
    data: DataLike,
    *,
    mach_level: float = 1.0,
) -> dict[str, float]:
    """
    Average Alfvén-surface radii over the projected stellar surface.

    This follows the old batplotlib method: take the `M_A = 1` surface, project it
    radially onto the unit sphere, and area-average the original spherical radius
    and cylindrical radius on that projected surface.
    """
    _grid, surface = build_alfven_surface(data, mach_level=mach_level)
    projected, radii, cyl_radii = _project_surface_to_unit_sphere(surface)
    areas = np.asarray(projected.cell_data["Area"], dtype=float)
    valid_radii = np.asarray(radii, dtype=float)
    valid_radii = valid_radii[np.isfinite(valid_radii) & (valid_radii > 0.0)]
    mean_radius = _projected_surface_average(projected, radii, radii, areas)
    mean_cyl_radius = _projected_surface_average(projected, cyl_radii, radii, areas)
    return {
        "average_alfven_radius [R]": float(mean_radius),
        "average_alfven_cyl_radius [R]": float(mean_cyl_radius),
        "min_alfven_radius [R]": float(np.min(valid_radii)),
        "max_alfven_radius [R]": float(np.max(valid_radii)),
        "cell_count": int(projected.n_cells),
    }


def plot_alfven_surface(
    data: DataLike,
    *,
    mach_level: float = 1.0,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    off_screen: bool = False,
    show: bool = True,
    screenshot: str | PathLike[str] | None = None,
) -> tuple[pv.Plotter, pv.PolyData, pv.UnstructuredGrid]:
    """
    Plot the Alfvén surface colored by wind speed.
    """
    grid, surface = build_alfven_surface(data, mach_level=mach_level)
    plotter = _plot_surface_with_wind_speed(
        grid,
        surface,
        title=f"Alfven surface (M_A={mach_level:g})",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        off_screen=off_screen,
        show=show,
        screenshot=screenshot,
    )
    return plotter, surface, grid


def plot_current_sheet_surface(
    data: DataLike,
    *,
    br_level: float = 0.0,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    off_screen: bool = False,
    show: bool = True,
    screenshot: str | PathLike[str] | None = None,
) -> tuple[pv.Plotter, pv.PolyData, pv.UnstructuredGrid]:
    """
    Plot the current sheet ``B_r = br_level`` colored by wind speed.
    """
    grid, surface = build_current_sheet_surface(data, br_level=br_level)
    plotter = _plot_surface_with_wind_speed(
        grid,
        surface,
        title=f"Current sheet (B_r={br_level:g})",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        off_screen=off_screen,
        show=show,
        screenshot=screenshot,
    )
    return plotter, surface, grid


def _mhd_surface_state(sds) -> dict[str, np.ndarray]:
    points = np.asarray(sds.points, dtype=float)[:, :3]
    magnetic_vector = resolve_magnetic_vector_si(sds)
    return {
        "U [m/s]": resolve_wind_speed_si(sds),
        "B [T]": np.linalg.norm(magnetic_vector, axis=1),
        "Rho [kg/m^3]": resolve_density_si(sds),
        "B_r [T]": radial_component(magnetic_vector, points),
    }


def _plot_surface_with_wind_speed(
    grid: pv.UnstructuredGrid,
    surface: pv.PolyData,
    *,
    title: str,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    off_screen: bool,
    show: bool,
    screenshot: str | PathLike[str] | None,
) -> pv.Plotter:
    plotter = pv.Plotter(off_screen=off_screen)
    plotter.add_mesh(grid.outline(), color="white", opacity=0.2, line_width=1.0)
    mesh_args = {
        "scalars": "U [m/s]",
        "cmap": cmap,
        "smooth_shading": True,
        "scalar_bar_args": readable_scalar_bar_args("U [m/s]"),
    }
    clim = _scalar_limits(vmin, vmax)
    if clim is not None:
        mesh_args["clim"] = clim
    plotter.add_mesh(surface, **mesh_args)
    plotter.add_axes()
    plotter.add_title(title)

    if show or screenshot is not None:
        plotter.show(screenshot=None if screenshot is None else str(screenshot), auto_close=False)

    return plotter


def _current_sheet_fit_points(
    surface: pv.PolyData,
    *,
    rmin: float,
    rmax: float,
    max_points: int,
) -> np.ndarray:
    points = np.asarray(surface.points, dtype=float)
    radii = np.linalg.norm(points, axis=1)
    fit_points = points[(rmin < radii) & (radii < rmax)]
    if fit_points.shape[0] < 3:
        raise ValueError(
            f"Current sheet plane fit requires at least 3 points in {rmin:g} < r < {rmax:g}; "
            f"found {fit_points.shape[0]}"
        )

    stride = int(np.ceil(fit_points.shape[0] / max_points))
    return fit_points[::stride]


def _fit_origin_plane_normal(points: np.ndarray) -> np.ndarray:
    # The old batplotlib fit used an uncentered SVD, which fits a plane through the origin.
    _u, _s, vh = np.linalg.svd(np.asarray(points, dtype=float), full_matrices=False)
    normal = np.asarray(vh[-1], dtype=float)
    normal /= np.linalg.norm(normal)
    if normal[2] < 0.0:
        normal = -normal
    return normal


def _project_surface_to_unit_sphere(surface: pv.PolyData) -> tuple[pv.PolyData, np.ndarray, np.ndarray]:
    projected = surface.triangulate(inplace=False).copy(deep=True)
    points = np.asarray(projected.points, dtype=float)
    radii = np.linalg.norm(points, axis=1)
    cyl_radii = np.linalg.norm(points[:, :2], axis=1)
    projected.points = np.divide(
        points,
        radii[:, None],
        out=np.zeros_like(points),
        where=radii[:, None] > 0.0,
    )
    projected = projected.compute_cell_sizes(length=False, area=True, volume=False)
    return projected, radii, cyl_radii


def _projected_surface_average(
    projected: pv.PolyData,
    point_values: np.ndarray,
    point_radii: np.ndarray,
    cell_areas: np.ndarray,
) -> float:
    faces = np.asarray(projected.faces, dtype=int).reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        raise ValueError("Expected triangulated Alfvén surface")
    triangle_ids = faces[:, 1:]
    cell_values = np.mean(np.asarray(point_values, dtype=float)[triangle_ids], axis=1)
    valid = (
        np.isfinite(cell_values)
        & np.isfinite(cell_areas)
        & (cell_areas > 0.0)
        & np.all(np.asarray(point_radii, dtype=float)[triangle_ids] > 0.0, axis=1)
    )
    if not np.any(valid):
        raise ValueError("No valid projected Alfvén-surface cells remained for averaging")
    return float(np.sum(cell_values[valid] * cell_areas[valid]) / np.sum(cell_areas[valid]))


def _scalar_limits(vmin: float | None, vmax: float | None) -> tuple[float, float] | None:
    if (vmin is None) ^ (vmax is None):
        raise ValueError("vmin and vmax must be provided together")
    if vmin is not None and vmax is not None:
        return float(vmin), float(vmax)
    return None


__all__ = [
    "alfven_surface_averages",
    "build_alfven_surface",
    "build_current_sheet_surface",
    "current_sheet_orientation",
    "plot_alfven_surface",
    "plot_current_sheet_surface",
]
