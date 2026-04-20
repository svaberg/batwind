from __future__ import annotations

from os import PathLike

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from batwind.algorithms.sphere_sampling import PolarAzimuthalGrid
from batwind.pyvista._scalar_bar import readable_scalar_bar_args
from batwind.pyvista.convert import to_unstructured_grid
from batwind.pyvista.fields import resolve_wind_speed_si
from batwind.smart_ds import SmartDs


def build_alfven_surface(
    smart_ds: SmartDs,
    *,
    mach_level: float = 1.0,
) -> tuple[pv.UnstructuredGrid, pv.PolyData]:
    """
    Build the Alfvén surface ``M_A = mach_level`` and attach wind-speed coloring.
    """
    wind_speed = resolve_wind_speed_si(smart_ds)
    alfven_mach = np.asarray(smart_ds["M_A [none]"], dtype=float)

    grid = to_unstructured_grid(
        smart_ds,
        point_data={
            "U [m/s]": wind_speed,
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
    smart_ds: SmartDs,
    *,
    br_level: float = 0.0,
) -> tuple[pv.UnstructuredGrid, pv.PolyData]:
    """
    Build the current sheet ``B_r = br_level`` and attach MHD fields for coloring.
    """
    wind_speed = resolve_wind_speed_si(smart_ds)
    radial_magnetic_field = np.asarray(smart_ds["B_r [T]"], dtype=float)

    grid = to_unstructured_grid(
        smart_ds,
        point_data={
            "U [m/s]": wind_speed,
            "B_r [T]": radial_magnetic_field,
        },
    )
    grid.set_active_scalars("B_r [T]")

    surface = grid.contour([float(br_level)], scalars="B_r [T]")
    if surface.n_points == 0 or surface.n_cells == 0:
        finite = radial_magnetic_field[np.isfinite(radial_magnetic_field)]
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
    smart_ds: SmartDs,
    *,
    br_level: float = 0.0,
    rmin: float = 0.0,
    rmax: float = 30.0,
    max_points: int = 10000,
) -> float:
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

    _grid, surface = build_current_sheet_surface(smart_ds, br_level=br_level)
    points = np.asarray(surface.points, dtype=float)
    radii = np.linalg.norm(points, axis=1)
    fit_points = points[(float(rmin) < radii) & (radii < float(rmax))]
    if fit_points.shape[0] < 3:
        raise ValueError(
            f"Current sheet plane fit requires at least 3 points in {rmin:g} < r < {rmax:g}; "
            f"found {fit_points.shape[0]}"
        )
    stride = int(np.ceil(fit_points.shape[0] / int(max_points)))
    fit_points = fit_points[::stride]
    # Old batplotlib used an uncentered SVD, which fits a plane through the origin.
    _u, _s, vh = np.linalg.svd(np.asarray(fit_points, dtype=float), full_matrices=False)
    normal = np.asarray(vh[-1], dtype=float)
    normal /= np.linalg.norm(normal)
    if normal[2] < 0.0:
        normal = -normal
    return float(np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0))))


def alfven_surface_averages(
    smart_ds: SmartDs,
    *,
    mach_level: float = 1.0,
) -> tuple[float, float]:
    """
    Average Alfvén-surface radii over the projected stellar surface.

    This follows the old batplotlib method: take the `M_A = 1` surface, project it
    radially onto the unit sphere, and area-average the original spherical radius
    and cylindrical radius on that projected surface.
    """
    _grid, surface = build_alfven_surface(smart_ds, mach_level=mach_level)
    projected, radii, cyl_radii = project_surface_to_unit_sphere(surface)
    areas = np.asarray(projected.cell_data["Area"], dtype=float)
    mean_radius = projected_surface_average(projected, radii, radii, areas)
    mean_cyl_radius = projected_surface_average(projected, cyl_radii, radii, areas)
    return float(mean_radius), float(mean_cyl_radius)


def alfven_surface_radius_map(
    smart_ds: SmartDs,
    *,
    n_polar: int,
    n_azimuth: int,
    mach_level: float = 1.0,
) -> dict[str, np.ndarray]:
    """
    Sample the projected Alfvén surface on a regular ``(polar, azimuth)`` grid.
    """
    if int(n_polar) <= 0:
        raise ValueError("n_polar must be a positive integer")
    if int(n_azimuth) <= 0:
        raise ValueError("n_azimuth must be a positive integer")

    _grid, surface = build_alfven_surface(smart_ds, mach_level=mach_level)
    projected, radii, _cyl_radii = project_surface_to_unit_sphere(surface)
    projected.point_data["alfven_radius [R]"] = radii

    angular_grid = PolarAzimuthalGrid(
        np.linspace(0.0, np.pi, int(n_polar) + 1),
        np.linspace(-np.pi, np.pi, int(n_azimuth) + 1),
    )
    polar = np.asarray(angular_grid.polar_centres, dtype=float).T
    azimuth = np.asarray(angular_grid.azimuthal_centres, dtype=float).T
    sample_points = angular_grid.centres_cartesian(radius=1.0).transpose(1, 0, 2).reshape(-1, 3)
    point_tree = cKDTree(np.asarray(projected.points, dtype=float))
    _distance, point_ids = point_tree.query(sample_points)
    radius_map = np.asarray(radii, dtype=float)[np.asarray(point_ids, dtype=int)]

    return {
        "polar [rad]": polar,
        "azimuth [rad]": azimuth,
        "cell_solid_angle [sr]": np.asarray(angular_grid.cell_solid_angle, dtype=float),
        "alfven_radius [R]": radius_map.reshape(polar.shape),
    }


def plot_alfven_surface(
    smart_ds: SmartDs,
    *,
    mach_level: float = 1.0,
    vmin: float | None = None,
    vmax: float | None = None,
    off_screen: bool = False,
    screenshot: str | PathLike[str] | None = None,
) -> tuple[pv.Plotter, pv.PolyData]:
    """
    Plot the Alfvén surface colored by wind speed.
    """
    _grid, surface = build_alfven_surface(smart_ds, mach_level=mach_level)
    plotter = surface_plotter_with_wind_speed(
        surface,
        vmin=vmin,
        vmax=vmax,
        off_screen=off_screen,
        screenshot=screenshot,
    )
    return plotter, surface


def plot_current_sheet_surface(
    smart_ds: SmartDs,
    *,
    br_level: float = 0.0,
    vmin: float | None = None,
    vmax: float | None = None,
    off_screen: bool = False,
    screenshot: str | PathLike[str] | None = None,
) -> tuple[pv.Plotter, pv.PolyData]:
    """
    Plot the current sheet ``B_r = br_level`` colored by wind speed.
    """
    _grid, surface = build_current_sheet_surface(smart_ds, br_level=br_level)
    plotter = surface_plotter_with_wind_speed(
        surface,
        vmin=vmin,
        vmax=vmax,
        off_screen=off_screen,
        screenshot=screenshot,
    )
    return plotter, surface


def surface_plotter_with_wind_speed(
    surface: pv.PolyData,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    off_screen: bool = False,
    screenshot: str | PathLike[str] | None = None,
) -> pv.Plotter:
    plotter = pv.Plotter(off_screen=off_screen)
    mesh_args = {
        "scalars": "U [m/s]",
        "cmap": "viridis",
        "smooth_shading": True,
        "scalar_bar_args": readable_scalar_bar_args("U [m/s]"),
    }
    if (vmin is None) ^ (vmax is None):
        raise ValueError("vmin and vmax must be provided together")
    if vmin is not None and vmax is not None:
        mesh_args["clim"] = (float(vmin), float(vmax))
    plotter.add_mesh(surface, **mesh_args)

    if screenshot is not None:
        plotter.show(screenshot=str(screenshot), auto_close=False)
    elif not off_screen:
        plotter.show(auto_close=False)
    return plotter


def project_surface_to_unit_sphere(surface: pv.PolyData) -> tuple[pv.PolyData, np.ndarray, np.ndarray]:
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


def projected_surface_average(
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


__all__ = [
    "alfven_surface_radius_map",
    "alfven_surface_averages",
    "build_alfven_surface",
    "build_current_sheet_surface",
    "current_sheet_orientation",
    "plot_alfven_surface",
    "plot_current_sheet_surface",
]
