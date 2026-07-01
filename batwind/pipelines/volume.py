"""Per-file 3D volume pipeline for `batwind-pipe`."""

from __future__ import annotations

import logging
from math import isfinite
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import pyvista as pv
from matplotlib import ticker
from matplotlib.colors import LogNorm
from batcamp import camera_rays
from batcamp import Octree
from batcamp import OctreeInterpolator
from batcamp import OctreeRayTracer

from batwind.algorithms.octree_integration import cumulative_radius_exact_rpa
from batwind.algorithms.octree_integration import radial_emission_profile_exact_rpa
from batwind.analysis.shells import integrate_shell_scalar
from batwind.analysis.shells import sample_spherical_shells_fibonacci
from batwind.constants import DEFAULT_QUICKLOOK_RADII_R
from batwind.physics.emission import DEFAULT_RESPONSE_FUNCTION_PATH
from batwind.physics.emission import band_emissivity_from_response_table_si
from batwind.physics.emission import point_unblocked_solid_angle_sr
from batwind.pipelines.utils import output_prefix_from_input_file
from batwind.pyvista.field_lines import (
    build_closed_field_line_max_radius_surface,
    build_field_line_max_radius_surface,
    build_magnetic_field_lines,
    field_line_max_radius_map,
    open_flux_and_area_fractions,
    project_field_lines_to_view_plane,
    visible_magnetic_field_lines,
)
from batwind.pyvista.fields import resolve_body_radius
from batwind.pyvista.isosurfaces import (
    alfven_surface_averages,
    alfven_surface_radius_map,
    current_sheet_orientation,
)
from batwind.pyvista.viewport import plot_pyvista_viewport
from batwind.smart_ds import SmartDs

log = logging.getLogger(__name__)
add_record = logging.getLogger(f"recorder.{__name__}").debug

LOS_GRID_N = 512
LOS_EXAMPLE_GRID_N = 192
LOS_EXAMPLE_SIDE_LENGTH_R = 2.0
FIELDLINE_FRACTION_N_SEEDS = 1000
ANGULAR_MAP_N_POLAR = 18
ANGULAR_MAP_N_AZIMUTH = 36
SURFACE_RENDER_N_POLAR = 36
SURFACE_RENDER_N_AZIMUTH = 72
SURFACE_VIEWPORT_FIGSIZE = (7.0, 7.0)
SURFACE_VIEWPORT_DPI = 180
SURFACE_VIEWPORT_RENDER_SIZE = (1400, 1400)
CORONAL_EMISSION_BANDS = {
    "hard": {
        "response_components": ("Hard_line", "Hard_cont"),
        "display_label": "Hard X-ray",
    },
    "rosat": {
        "response_components": ("ROSAT_line", "ROSAT_cont"),
        "display_label": "ROSAT",
    },
    "euv": {
        "response_components": ("EUV_line", "EUV_cont"),
        "display_label": "EUV",
    },
}
CORONAL_EMISSION_BAND_NAMES = tuple(CORONAL_EMISSION_BANDS)
CORONAL_EMISSION_BAND_LABELS = {
    band_name: rf"{CORONAL_EMISSION_BANDS[band_name]['display_label']} band intensity [W m$^{{-2}}$ sr$^{{-1}}$]"
    for band_name in CORONAL_EMISSION_BAND_NAMES
}
CORONAL_EMISSION_IMAGE_INTENSITY_UNIT = r"W m$^{-2}$ sr$^{-1}$"
CORONAL_EMISSION_RADIANT_INTENSITY_UNIT = r"W sr$^{-1}$"
CORONAL_EMISSION_LUMINOSITY_UNIT = r"W"
CORONAL_EMISSION_EMISSIVITY_UNIT = r"W m$^{-3}$ sr$^{-1}$"
CORONAL_EMISSION_SINGLE_DIRECTION_VIEW_AXIS = "+Y"
CORONAL_EMISSION_EXAMPLE_VIEW_AXIS = "+Y"
CORONAL_EMISSION_TOTALS_IMAGE_N = 512
FIELD_LINE_OVERLAY_N_SEEDS = 256


def build_los_geometry(smart_ds: SmartDs) -> tuple[Octree, OctreeRayTracer, tuple[float, float, float, float, float, float]]:
    """
    Build the shared octree LOS geometry state.
    """
    x = np.asarray(smart_ds["X [R]"], dtype=float)
    y = np.asarray(smart_ds["Y [R]"], dtype=float)
    z = np.asarray(smart_ds["Z [R]"], dtype=float)
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    z_min = float(np.nanmin(z))
    z_max = float(np.nanmax(z))
    tree = Octree.from_ds(smart_ds.raw)
    tracer = OctreeRayTracer(tree)
    return tree, tracer, (x_min, x_max, y_min, y_max, z_min, z_max)


def build_los_interpolator(tree: Octree, point_values: np.ndarray) -> OctreeInterpolator:
    """
    Build an octree interpolator from one point-valued scalar field.
    """
    return OctreeInterpolator(tree, np.asarray(point_values, dtype=float))


def integrate_image_radiant_intensity(
    image: np.ndarray,
    extent: tuple[float, float, float, float],
    body_radius_m: float,
) -> float:
    """
    Integrate one LOS intensity image over projected image-plane area.

    Units:
    - image intensity: ``W m^-2 sr^-1``
    - projected area: ``m^2``
    - returned radiant intensity: ``W sr^-1``
    """
    x_min, x_max, y_min, y_max = extent
    pixel_area_m2 = (
        (float(x_max - x_min) * body_radius_m) * (float(y_max - y_min) * body_radius_m) / float(image.size)
    )
    return float(np.nansum(np.asarray(image, dtype=float)) * pixel_area_m2)


def plot_coronal_emission_radial_summary(
    radial_png_path: Path,
    profiles: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    stats: dict[str, dict[str, float]],
) -> None:
    """
    Plot per-band radial emission and cumulative-fraction summaries.
    """
    band_names = tuple(profiles)
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.5), constrained_layout=True, sharex=True)
    colors = {band_name: f"C{band_id}" for band_id, band_name in enumerate(band_names)}
    for band_name, (radius_r, shell_emission, cumulative_fraction) in profiles.items():
        positive_shell_emission = np.where(np.asarray(shell_emission, dtype=float) > 0.0, shell_emission, np.nan)
        axes[0].plot(
            radius_r,
            positive_shell_emission,
            color=colors[band_name],
            label=(
                f"{band_name}: "
                f"r90={stats[band_name]['r90_r']:.2f} R*, "
                f"r99={stats[band_name]['r99_r']:.2f} R*"
            ),
        )
        axes[1].plot(radius_r, cumulative_fraction, color=colors[band_name], label=band_name)
        axes[1].axvline(stats[band_name]["r90_r"], color=colors[band_name], linestyle="--", alpha=0.5)
        axes[1].axvline(stats[band_name]["r99_r"], color=colors[band_name], linestyle=":", alpha=0.7)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[1].set_xscale("log")
    axes[0].set_ylabel(f"Shell luminosity [{CORONAL_EMISSION_LUMINOSITY_UNIT}]")
    axes[1].set_ylabel("Cumulative fraction [-]")
    axes[1].set_xlabel(r"Radius [$R_\star$]")
    axes[0].set_title("Coronal Band Emission by Radius")
    axes[1].set_title("Cumulative Emission Fraction")
    axes[1].axhline(0.90, color="0.3", linestyle="--", linewidth=0.8)
    axes[1].axhline(0.99, color="0.3", linestyle=":", linewidth=0.8)
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.savefig(radial_png_path)
    plt.close(fig)


def plot_coronal_emission_unit_summary(
    summary_png_path: Path,
    stats: dict[str, dict[str, float]],
    *,
    view_axis: str,
) -> None:
    """
    Plot a unit-carrying summary of the X-ray totals.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.8), constrained_layout=True)
    ax.axis("off")
    lines = [
        "Coronal emission unit trace",
        r"$G(T)$: response table values [W m$^3$ sr$^{-1}$]",
        r"$N_e$: electron density [m$^{-3}$]",
        rf"$\epsilon = N_e^2 G(T)$ [{CORONAL_EMISSION_EMISSIVITY_UNIT}]",
        rf"$I_{{\mathrm{{dir}}}} = \int \epsilon \, dl$ [{CORONAL_EMISSION_IMAGE_INTENSITY_UNIT}]",
        rf"$J_{{\mathrm{{dir,{view_axis}}}}} = \int I_{{\mathrm{{dir}}}} \, dA$ [{CORONAL_EMISSION_RADIANT_INTENSITY_UNIT}]",
        rf"$L_{{\Omega}} = \int \epsilon \, \Omega_{{\mathrm{{unblocked}}}} \, dV$ [{CORONAL_EMISSION_LUMINOSITY_UNIT}]",
        rf"$L_{{4\pi}} = \int \epsilon \, 4\pi \, dV$ [{CORONAL_EMISSION_LUMINOSITY_UNIT}]",
        "",
        "Band totals",
    ]
    for band_name in stats:
        band_stats = stats[band_name]
        lines.extend(
            [
                (
                    f"{band_name}: "
                    f"J_dir={band_stats['directional_radiant_intensity']:.3e}, "
                    f"L_Ω={band_stats['unblocked_total']:.3e}, "
                    f"L_4π={band_stats['four_pi_total']:.3e}"
                ),
                (
                    f"      r90={band_stats['r90_r']:.2f} R*, "
                    f"r99={band_stats['r99_r']:.2f} R*, "
                    f"L_Ω/L_4π={band_stats['unblocked_over_four_pi']:.4f}"
                ),
            ]
        )
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10)
    fig.savefig(summary_png_path)
    plt.close(fig)


def render_rho2_los_image(
    tracer: OctreeRayTracer,
    interp: OctreeInterpolator,
    bounds_r: tuple[float, float, float, float, float, float],
    *,
    path_length_scale: float,
    image_n: int,
    view_axis: str,
    width: float | None = None,
    height: float | None = None,
) -> tuple[np.ndarray, tuple[float, float, float, float], np.ndarray]:
    """
    Render one LOS image through the octree with an explicit path-length scale.
    """
    x_min, x_max, y_min, y_max, z_min, z_max = bounds_r
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    z_center = 0.5 * (z_min + z_max)
    x_span = x_max - x_min
    z_span = z_max - z_min
    x_pad = max(1.0e-6 * x_span, 1.0e-6)
    z_pad = max(1.0e-6 * z_span, 1.0e-6)
    if view_axis == "+Z":
        width = (x_max - x_min) if width is None else float(width)
        height = (y_max - y_min) if height is None else float(height)
        origins, directions = camera_rays(
            origin=(x_center, y_center, z_min - z_pad),
            target=(x_center, y_center, z_max + z_pad),
            up=(0.0, 1.0, 0.0),
            nx=image_n,
            ny=image_n,
            width=width,
            height=height,
            projection="parallel",
        )
        extent = (x_center - 0.5 * width, x_center + 0.5 * width, y_center - 0.5 * height, y_center + 0.5 * height)
    elif view_axis == "+X":
        width = (y_max - y_min) if width is None else float(width)
        height = (z_max - z_min) if height is None else float(height)
        origins, directions = camera_rays(
            origin=(x_min - x_pad, y_center, z_center),
            target=(x_max + x_pad, y_center, z_center),
            up=(0.0, 0.0, 1.0),
            nx=image_n,
            ny=image_n,
            width=width,
            height=height,
            projection="parallel",
        )
        extent = (y_center - 0.5 * width, y_center + 0.5 * width, z_center - 0.5 * height, z_center + 0.5 * height)
    elif view_axis == "+Y":
        width = (x_max - x_min) if width is None else float(width)
        height = (z_max - z_min) if height is None else float(height)
        origins, directions = camera_rays(
            origin=(x_center, y_min - x_pad, z_center),
            target=(x_center, y_max + x_pad, z_center),
            up=(0.0, 0.0, 1.0),
            nx=image_n,
            ny=image_n,
            width=width,
            height=height,
            projection="parallel",
        )
        extent = (x_center - 0.5 * width, x_center + 0.5 * width, z_center - 0.5 * height, z_center + 0.5 * height)
    else:
        raise ValueError(f"Unsupported LOS view_axis {view_axis!r}")

    image_r_units, counts = tracer.trilinear_image(interp, origins, directions)
    image_scaled = np.asarray(image_r_units, dtype=float) * float(path_length_scale)
    return image_scaled, extent, counts


def save_los_colormesh_npz(
    npz_path: Path,
    image: np.ndarray,
    extent: tuple[float, float, float, float],
    counts: np.ndarray,
    *,
    view_axis: str,
) -> None:
    """
    Save one LOS image as a reusable colormesh product.
    """
    x_min, x_max, y_min, y_max = extent
    y_n, x_n = image.shape
    x = np.linspace(x_min, x_max, x_n)
    y = np.linspace(y_min, y_max, y_n)
    if view_axis == "+Z":
        xlabel = "X [R]"
        ylabel = "Y [R]"
        title = r"LOS $\int \rho^2\,dl$"
    elif view_axis == "+X":
        xlabel = "Y [R]"
        ylabel = "Z [R]"
        title = r"Side LOS $\int \rho^2\,dl$"
    elif view_axis == "+Y":
        xlabel = "X [R]"
        ylabel = "Z [R]"
        title = r"Example LOS $\int \rho^2\,dl$"
    else:
        raise ValueError(f"Unsupported LOS view_axis {view_axis!r}")
    np.savez_compressed(
        npz_path,
        x=x,
        y=y,
        image=np.asarray(image, dtype=float),
        counts=np.asarray(counts),
        view_axis=np.asarray(view_axis),
        xlabel=np.asarray(xlabel),
        ylabel=np.asarray(ylabel),
        title=np.asarray(title),
        colorbar_label=np.asarray(r"$\int \rho^2\,dl$ [kg$^2$/m$^5$]"),
        unit=np.asarray("kg^2/m^5"),
    )


def save_example_los_colormesh_npz(
    source_npz_path: Path,
    example_npz_path: Path,
    *,
    side_length_r: float,
    colorbar_label: str,
    unit: str,
) -> None:
    """
    Save a cropped example-panel LOS colormesh product.
    """
    with np.load(source_npz_path, allow_pickle=False) as data:
        x = np.asarray(data["x"], dtype=float)
        y = np.asarray(data["y"], dtype=float)
        image = np.asarray(data["image"], dtype=float)
        counts = np.asarray(data["counts"])
        view_axis = str(data["view_axis"])
    x_mask = np.abs(x) <= side_length_r
    y_mask = np.abs(y) <= side_length_r
    x_crop = x[x_mask]
    y_crop = y[y_mask]
    image_crop = image[np.ix_(y_mask, x_mask)]
    counts_crop = counts[np.ix_(y_mask, x_mask)]
    np.savez_compressed(
        example_npz_path,
        x=x_crop,
        y=y_crop,
        image=np.asarray(image_crop, dtype=float),
        counts=counts_crop,
        view_axis=np.asarray(view_axis),
        xlabel=np.asarray(r"$x$ $(R_\star)$"),
        ylabel=np.asarray(r"$z$ $(R_\star)$"),
        colorbar_label=np.asarray(colorbar_label),
        unit=np.asarray(unit),
        side_length_r=np.asarray(float(side_length_r)),
    )


def overlay_sphere_graticule(
    ax: plt.Axes,
    *,
    radius: float = 1.0,
    central_lon_deg: float = 0.0,
    central_lat_deg: float = 40.0,
    color: str = "black",
    linestyle: str = "dotted",
    linewidth: float = 0.5,
) -> None:
    """
    Overlay a simple orthographic globe graticule in data coordinates.
    """
    lon0 = np.deg2rad(central_lon_deg)
    lat0 = np.deg2rad(central_lat_deg)

    def project(lon_deg: np.ndarray, lat_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lon = np.deg2rad(lon_deg)
        lat = np.deg2rad(lat_deg)
        cos_c = np.sin(lat0) * np.sin(lat) + np.cos(lat0) * np.cos(lat) * np.cos(lon - lon0)
        x = radius * np.cos(lat) * np.sin(lon - lon0)
        y = radius * (np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(lon - lon0))
        return x, y, cos_c >= 0.0

    theta = np.linspace(0.0, 2.0 * np.pi, 721)
    ax.plot(radius * np.cos(theta), radius * np.sin(theta), color=color, linestyle=linestyle, linewidth=linewidth)

    longitudes = np.arange(-180.0, 180.0, 45.0)
    latitudes = np.arange(-60.0, 61.0, 30.0)
    lat_line = np.linspace(-90.0, 90.0, 721)
    lon_line = np.linspace(-180.0, 180.0, 721)

    for lon_deg in longitudes:
        lon = np.full_like(lat_line, lon_deg)
        x, y, visible = project(lon, lat_line)
        ax.plot(x[visible], y[visible], color=color, linestyle=linestyle, linewidth=linewidth)

    for lat_deg in latitudes:
        lat = np.full_like(lon_line, lat_deg)
        x, y, visible = project(lon_line, lat)
        ax.plot(x[visible], y[visible], color=color, linestyle=linestyle, linewidth=linewidth)


def plot_los_colormesh_npz(npz_path: Path, png_path: Path) -> None:
    """
    Plot one saved LOS colormesh product.
    """
    with np.load(npz_path, allow_pickle=False) as data:
        x = np.asarray(data["x"], dtype=float)
        y = np.asarray(data["y"], dtype=float)
        image = np.asarray(data["image"], dtype=float)
        xlabel = str(data["xlabel"])
        ylabel = str(data["ylabel"])
        title = str(data["title"])
        colorbar_label = str(data["colorbar_label"])
    x_mesh, y_mesh = np.meshgrid(x, y)
    x_span = float(x[-1] - x[0]) if x.size else 0.0
    y_span = float(y[-1] - y[0]) if y.size else 0.0
    positive = image[np.isfinite(image) & (image > 0.0)]
    norm = LogNorm(vmin=float(np.nanmin(positive)), vmax=float(np.nanmax(positive))) if positive.size else None
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    mesh = ax.pcolormesh(
        x_mesh,
        y_mesh,
        image,
        cmap="viridis",
        norm=norm,
        shading="gouraud",
        rasterized=True,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")
    if x_span <= 10.0:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    if y_span <= 10.0:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax.grid(False)
    fig.colorbar(mesh, ax=ax, label=colorbar_label)
    fig.savefig(png_path)
    plt.close(fig)


def load_example_los_colormesh_npz(npz_path: Path) -> dict[str, object]:
    with np.load(npz_path, allow_pickle=False) as data:
        return {
            "x": np.asarray(data["x"], dtype=float),
            "y": np.asarray(data["y"], dtype=float),
            "image": np.asarray(data["image"], dtype=float),
            "xlabel": str(data["xlabel"]),
            "ylabel": str(data["ylabel"]),
            "colorbar_label": str(data["colorbar_label"]),
            "side_length_r": float(data["side_length_r"]),
            "view_axis": str(data["view_axis"]),
        }


def draw_example_los_colormesh(ax: plt.Axes, example_data: dict[str, object]):
    x = np.asarray(example_data["x"], dtype=float)
    y = np.asarray(example_data["y"], dtype=float)
    image = np.asarray(example_data["image"], dtype=float)
    positive = image[np.isfinite(image) & (image > 0.0)]
    norm = LogNorm(vmin=float(np.nanmin(positive)), vmax=float(np.nanmax(positive))) if positive.size else None
    mesh = ax.pcolormesh(x, y, image, cmap="viridis", norm=norm, shading="nearest", rasterized=True)
    side_length_r = float(example_data["side_length_r"])
    ax.set_xlim(-side_length_r, side_length_r)
    ax.set_ylim(-side_length_r, side_length_r)
    ax.set_xlabel(str(example_data["xlabel"]))
    ax.set_ylabel(str(example_data["ylabel"]))
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax.grid(False)
    overlay_sphere_graticule(ax, color="0.2", linewidth=0.35)
    return mesh


def draw_projected_field_lines(
    ax: plt.Axes,
    field_lines,
    *,
    view_axis: str,
) -> None:
    projected_lines = project_field_lines_to_view_plane(field_lines, view_axis=view_axis)
    line_effects = [
        path_effects.Stroke(linewidth=2.6, foreground="black"),
        path_effects.Normal(),
    ]
    closed_x, closed_y = projected_lines["closed"]
    if closed_x.size > 0:
        closed_line, = ax.plot(closed_x, closed_y, color="white", linewidth=1.2, zorder=4)
        closed_line.set_path_effects(line_effects)
    open_x, open_y = projected_lines["open"]
    if open_x.size > 0:
        open_line, = ax.plot(
            open_x,
            open_y,
            color="white",
            linewidth=1.0,
            linestyle="--",
            zorder=4,
        )
        open_line.set_path_effects(line_effects)


def plot_example_los_colormesh_npz(npz_path: Path, png_path: Path) -> None:
    """
    Plot one cropped example-panel LOS colormesh product in the old notebook style.
    """
    example_data = load_example_los_colormesh_npz(npz_path)
    fig = plt.figure(figsize=(4.2, 4.3), constrained_layout=True)
    ax = fig.add_subplot(111)
    mesh = draw_example_los_colormesh(ax, example_data)
    colorbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.02, fraction=0.06, location="top")
    colorbar.set_label(str(example_data["colorbar_label"]))
    fig.savefig(png_path)
    plt.close(fig)


def plot_example_los_colormesh_field_lines_npz(
    npz_path: Path,
    field_lines,
    png_path: Path,
    *,
    title: str,
) -> None:
    """
    Plot one cropped synthetic-image panel with projected magnetic field lines on top.
    """
    example_data = load_example_los_colormesh_npz(npz_path)
    fig = plt.figure(figsize=(4.2, 4.3), constrained_layout=True)
    ax = fig.add_subplot(111)
    mesh = draw_example_los_colormesh(ax, example_data)
    draw_projected_field_lines(ax, field_lines, view_axis=str(example_data["view_axis"]))
    ax.set_title(title)
    colorbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.02, fraction=0.06, location="top")
    colorbar.set_label(str(example_data["colorbar_label"]))
    fig.savefig(png_path)
    plt.close(fig)


def sample_shell_grid(
    smart_ds: SmartDs,
    radii: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, bool]:
    stage_start = perf_counter()
    log.info("Sampling shell grid once for all diagnostics...")
    energy_source = "E [J/m^3]"
    fields = [
        "Rho [kg/m^3]",
        "U_x [m/s]",
        "U_y [m/s]",
        "U_z [m/s]",
        "B_x [T]",
        "B_y [T]",
        "B_z [T]",
    ]
    has_energy_source = energy_source in smart_ds
    if has_energy_source:
        fields.append(energy_source)
    body_radius = float(smart_ds["RBODY [m]"])
    shells = sample_spherical_shells_fibonacci(
        smart_ds,
        radii,
        fields=fields,
        n_points=24 * 48,
        method="octree",
        length_unit_to_m=body_radius,
    )
    shell_radii = np.nanmean(np.asarray(shells["R [R]"]).reshape(len(radii), -1), axis=1)
    log.debug("Sampling shell grid once for all diagnostics complete in %.2f s.", perf_counter() - stage_start)
    return shells, shell_radii, has_energy_source


def record_wind_mass_loss(ax, shell_radii: np.ndarray, shells: dict[str, np.ndarray]) -> None:
    stage_start = perf_counter()
    log.info("Computing wind mass loss...")
    mass_loss_values, mass_loss_coverage = integrate_shell_scalar(
        np.asarray(shells["mass_flux [kg/m^2/s]"]),
        np.asarray(shells["dA [m^2]"]),
    )
    ax.plot(shell_radii - 1.0, mass_loss_values, ".-", color="C0")
    ax.set_title("Wind Mass Loss")
    ax.set_ylabel("Mass flux [kg/s]")
    ax.set_xlabel("Height [R]")
    ax.grid(True, alpha=0.3)
    add_record("radius_R %r", shell_radii)
    add_record("mass_loss_kg_s %r", mass_loss_values)
    add_record("mass_loss_coverage %r", mass_loss_coverage)
    for radius_value, mass_loss_value in zip(shell_radii, mass_loss_values, strict=True):
        if isfinite(radius_value) and isfinite(mass_loss_value):
            add_record("mass_loss_radius_R %r", float(radius_value))
            add_record("mass_loss_value_kg_s %r", float(mass_loss_value))
    log.debug("Computing wind mass loss complete in %.2f s.", perf_counter() - stage_start)


def record_wind_torque(ax, shell_radii: np.ndarray, shells: dict[str, np.ndarray]) -> None:
    stage_start = perf_counter()
    log.info("Computing wind torque...")
    shell_area = np.asarray(shells["dA [m^2]"])
    magnetic_torque, torque_coverage_mag = integrate_shell_scalar(
        np.asarray(shells["magnetic_torque_density [N/m]"]),
        shell_area,
    )
    dynamic_torque, torque_coverage_dyn = integrate_shell_scalar(
        np.asarray(shells["dynamic_torque_density [N/m]"]),
        shell_area,
    )
    total_torque = magnetic_torque + dynamic_torque
    torque_coverage = np.minimum(torque_coverage_mag, torque_coverage_dyn)
    ax.plot(shell_radii - 1.0, total_torque, ".-", color="C1")
    ax.set_title("Wind Torque")
    ax.set_ylabel("Torque [Nm]")
    ax.set_xlabel("Height [R]")
    ax.grid(True, alpha=0.3)
    add_record("magnetic_torque_nm %r", magnetic_torque)
    add_record("dynamic_torque_nm %r", dynamic_torque)
    add_record("total_torque_nm %r", total_torque)
    add_record("total_torque_coverage %r", torque_coverage)
    for radius_value, torque_value in zip(shell_radii, total_torque, strict=True):
        if isfinite(radius_value) and isfinite(torque_value):
            add_record("total_torque_radius_R %r", float(radius_value))
            add_record("total_torque_value_nm %r", float(torque_value))
    log.debug("Computing wind torque complete in %.2f s.", perf_counter() - stage_start)


def record_open_magnetic_flux(ax, shell_radii: np.ndarray, shells: dict[str, np.ndarray]) -> None:
    stage_start = perf_counter()
    log.info("Computing open magnetic flux...")
    open_flux_values, open_flux_coverage = integrate_shell_scalar(
        np.abs(np.asarray(shells["B_r [T]"])),
        np.asarray(shells["dA [m^2]"]),
    )
    ax.plot(shell_radii - 1.0, open_flux_values, ".-", color="C2")
    ax.set_title("Open Magnetic Flux")
    ax.set_ylabel("Open flux [Wb]")
    ax.set_xlabel("Height [R]")
    ax.grid(True, alpha=0.3)
    add_record("open_flux_wb %r", open_flux_values)
    add_record("open_flux_coverage %r", open_flux_coverage)
    for radius_value, open_flux_value in zip(shell_radii, open_flux_values, strict=True):
        if isfinite(radius_value) and isfinite(open_flux_value):
            add_record("open_flux_radius_R %r", float(radius_value))
            add_record("open_flux_value_wb %r", float(open_flux_value))
    log.debug("Computing open magnetic flux complete in %.2f s.", perf_counter() - stage_start)


def record_energy_flux(
    ax,
    shell_radii: np.ndarray,
    shells: dict[str, np.ndarray],
    has_energy_source: bool,
) -> None:
    stage_start = perf_counter()
    log.info("Computing energy flux...")
    if has_energy_source:
        energy_flux_values, energy_flux_coverage = integrate_shell_scalar(
            np.asarray(shells["energy_flux [W/m^2]"]),
            np.asarray(shells["dA [m^2]"]),
        )
        ax.plot(shell_radii - 1.0, energy_flux_values, ".-", color="C3")
        ax.set_title("Energy Flux")
        ax.set_ylabel("Energy flux [W]")
        ax.set_xlabel("Height [R]")
        ax.grid(True, alpha=0.3)
        add_record("energy_flux_w %r", energy_flux_values)
        add_record("energy_flux_coverage %r", energy_flux_coverage)
        for radius_value, energy_flux_value in zip(shell_radii, energy_flux_values, strict=True):
            if isfinite(radius_value) and isfinite(energy_flux_value):
                add_record("energy_flux_radius_R %r", float(radius_value))
                add_record("energy_flux_value_w %r", float(energy_flux_value))
    else:
        ax.set_title("Energy Flux")
        ax.text(0.5, 0.5, "E [J/m^3] unavailable", ha="center", va="center")
        ax.set_axis_off()
    log.debug("Computing energy flux complete in %.2f s.", perf_counter() - stage_start)


def record_3d_quantities(smart_ds: SmartDs) -> float:
    stage_start = perf_counter()
    log.info("Computing 3D topology and surface quantities...")
    alfven_radius = alfven_surface_radius_map(
        smart_ds,
        n_polar=ANGULAR_MAP_N_POLAR,
        n_azimuth=ANGULAR_MAP_N_AZIMUTH,
    )
    add_record("polar_map_rad %r", np.asarray(alfven_radius["polar [rad]"], dtype=float))
    add_record("azimuth_map_rad %r", np.asarray(alfven_radius["azimuth [rad]"], dtype=float))
    add_record("angular_cell_solid_angle_sr %r", np.asarray(alfven_radius["cell_solid_angle [sr]"], dtype=float))
    alfven_radius_map_r = np.asarray(alfven_radius["alfven_radius [R]"], dtype=float)
    add_record("alfven_radius_map_R %r", alfven_radius_map_r)
    max_alfven_radius = float(np.nanmax(alfven_radius_map_r))

    average_alfven_radius, average_alfven_cyl_radius = alfven_surface_averages(smart_ds)
    add_record("average_alfven_radius_R %r", average_alfven_radius)
    add_record("average_alfven_cyl_radius_R %r", average_alfven_cyl_radius)

    field_line_radius = field_line_max_radius_map(
        smart_ds,
        n_polar=ANGULAR_MAP_N_POLAR,
        n_azimuth=ANGULAR_MAP_N_AZIMUTH,
    )
    add_record("field_line_max_radius_map_R %r", np.asarray(field_line_radius["field_line_max_radius [R]"], dtype=float))
    add_record("current_sheet_inclination_deg %r", current_sheet_orientation(smart_ds))

    open_flux_fraction, open_area_fraction = open_flux_and_area_fractions(
        smart_ds,
        open_radius=max_alfven_radius,
        n_seeds=FIELDLINE_FRACTION_N_SEEDS,
    )
    add_record("open_flux_fraction %r", open_flux_fraction)
    add_record("open_area_fraction %r", open_area_fraction)
    log.debug("Computing 3D topology and surface quantities complete in %.2f s.", perf_counter() - stage_start)
    return max_alfven_radius


def save_field_line_surface_plots(
    smart_ds: SmartDs,
    output_dir: Path,
    prefix: str,
    parent_dir: Path,
    max_alfven_radius: float,
) -> None:
    stage_start = perf_counter()
    log.info("Saving 3D field-line surface plots...")
    body_radius = resolve_body_radius(smart_ds)
    surfaces = [
        (
            build_field_line_max_radius_surface(
                smart_ds,
                n_polar=SURFACE_RENDER_N_POLAR,
                n_azimuth=SURFACE_RENDER_N_AZIMUTH,
            ),
            output_dir / f"{prefix}.field_line_max_radius_surface.png",
            "volume_field_line_max_radius_surface_png",
        ),
        (
            build_closed_field_line_max_radius_surface(
                smart_ds,
                open_radius=max_alfven_radius,
                n_polar=SURFACE_RENDER_N_POLAR,
                n_azimuth=SURFACE_RENDER_N_AZIMUTH,
            ),
            output_dir / f"{prefix}.closed_field_line_envelope.png",
            "volume_closed_field_line_envelope_png",
        ),
    ]
    for surface, output_png, record_name in surfaces:
        plotter = pv.Plotter(off_screen=True, window_size=(1800, 1800))
        surface_fig = None
        try:
            plotter.set_background("white")
            plotter.add_mesh(
                pv.Sphere(radius=body_radius, theta_resolution=120, phi_resolution=120),
                color=(0.85, 0.85, 0.85),
                smooth_shading=True,
                show_scalar_bar=False,
            )
            actor = plotter.add_mesh(
                surface,
                scalars=surface.active_scalars_name,
                cmap="viridis",
                opacity=0.85,
                smooth_shading=True,
                show_edges=False,
                show_scalar_bar=False,
            )
            surface_fig, surface_ax = plt.subplots(
                figsize=SURFACE_VIEWPORT_FIGSIZE,
                dpi=SURFACE_VIEWPORT_DPI,
                constrained_layout=True,
            )
            surface_radius = float(np.nanmax(np.linalg.norm(np.asarray(surface.points, dtype=float), axis=1)))
            surface_fig, surface_ax, _colorbar, _image = plot_pyvista_viewport(
                plotter,
                fig=surface_fig,
                ax=surface_ax,
                colorbar_actor=actor,
                colorbar_label=surface.active_scalars_name,
                view="isometric",
                view_center=(0.0, 0.0, 0.0),
                parallel_scale=1.1 * surface_radius,
                render_size=SURFACE_VIEWPORT_RENDER_SIZE,
            )
            surface_ax.grid(True, color="0.9", linewidth=0.6)
            surface_ax.axhline(0.0, color="0.82", linewidth=0.8)
            surface_ax.axvline(0.0, color="0.82", linewidth=0.8)
            surface_fig.savefig(output_png, dpi=SURFACE_VIEWPORT_DPI)
            add_record(record_name + " %r", str(output_png.relative_to(parent_dir)))
        finally:
            if surface_fig is not None:
                plt.close(surface_fig)
            plotter.close()
    log.debug("Saving 3D field-line surface plots complete in %.2f s.", perf_counter() - stage_start)


def save_los_images(
    smart_ds: SmartDs,
    output_dir: Path,
    prefix: str,
    parent_dir: Path,
    max_alfven_radius: float,
) -> None:
    stage_start = perf_counter()
    log.info("Rendering LOS rho^2 images...")
    body_radius_m = float(smart_ds["RBODY [m]"])
    image_n = LOS_GRID_N
    tree, tracer, bounds_r = build_los_geometry(smart_ds)
    interp = build_los_interpolator(tree, np.asarray(smart_ds["Rho [kg/m^3]"], dtype=float) ** 2)
    rho_sq_los, los_extent, los_counts = render_rho2_los_image(
        tracer,
        interp,
        bounds_r,
        path_length_scale=body_radius_m,
        image_n=image_n,
        view_axis="+Z",
    )
    rho_sq_los_side, los_side_extent, los_side_counts = render_rho2_los_image(
        tracer,
        interp,
        bounds_r,
        path_length_scale=body_radius_m,
        image_n=image_n,
        view_axis="+X",
    )
    rho_sq_los_example, los_example_extent, los_example_counts = render_rho2_los_image(
        tracer,
        interp,
        bounds_r,
        path_length_scale=body_radius_m,
        image_n=LOS_EXAMPLE_GRID_N,
        view_axis=CORONAL_EMISSION_EXAMPLE_VIEW_AXIS,
        width=2.0 * LOS_EXAMPLE_SIDE_LENGTH_R,
        height=2.0 * LOS_EXAMPLE_SIDE_LENGTH_R,
    )
    los_npz = output_dir / f"{prefix}.rho2_los.npz"
    los_png = output_dir / f"{prefix}.rho2_los.png"
    save_los_colormesh_npz(
        los_npz,
        rho_sq_los,
        los_extent,
        los_counts,
        view_axis="+Z",
    )
    plot_los_colormesh_npz(los_npz, los_png)
    los_side_npz = output_dir / f"{prefix}.rho2_los_side.npz"
    los_side_png = output_dir / f"{prefix}.rho2_los_side.png"
    save_los_colormesh_npz(
        los_side_npz,
        rho_sq_los_side,
        los_side_extent,
        los_side_counts,
        view_axis="+X",
    )
    plot_los_colormesh_npz(los_side_npz, los_side_png)
    los_example_npz_full = output_dir / f"{prefix}.rho2_los_example_full.npz"
    save_los_colormesh_npz(
        los_example_npz_full,
        rho_sq_los_example,
        los_example_extent,
        los_example_counts,
        view_axis=CORONAL_EMISSION_EXAMPLE_VIEW_AXIS,
    )
    los_example_npz = output_dir / f"{prefix}.rho2_los_example.npz"
    save_example_los_colormesh_npz(
        los_example_npz_full,
        los_example_npz,
        side_length_r=LOS_EXAMPLE_SIDE_LENGTH_R,
        colorbar_label="Proxy LOS intensity",
        unit="kg^2/m^5",
    )
    los_example_png = output_dir / f"{prefix}.rho2_los_example.png"
    plot_example_los_colormesh_npz(los_example_npz, los_example_png)

    response_path = DEFAULT_RESPONSE_FUNCTION_PATH
    overlay_plot_radius = float(np.hypot(LOS_EXAMPLE_SIDE_LENGTH_R, LOS_EXAMPLE_SIDE_LENGTH_R))
    _field_line_grid, _field_line_source, traced_field_lines = build_magnetic_field_lines(
        smart_ds,
        n_seeds=FIELD_LINE_OVERLAY_N_SEEDS,
    )
    overlay_field_lines = visible_magnetic_field_lines(
        traced_field_lines,
        plot_radius=overlay_plot_radius,
        open_line_plot_radius=max_alfven_radius,
    )
    point_unblocked_solid_angle = point_unblocked_solid_angle_sr(smart_ds)
    raw_band_emissivities = {
        band_name: band_emissivity_from_response_table_si(
            smart_ds,
            CORONAL_EMISSION_BANDS[band_name]["response_components"],
            response_path=response_path,
        )
        for band_name in CORONAL_EMISSION_BAND_NAMES
    }
    coronal_emission_band_stats = {}
    coronal_emission_band_profiles = {}
    for band_name, emissivity in raw_band_emissivities.items():
        band_interp = build_los_interpolator(tree, emissivity)
        band_image, band_extent, band_counts = render_rho2_los_image(
            tracer,
            band_interp,
            bounds_r,
            path_length_scale=body_radius_m,
            image_n=LOS_EXAMPLE_GRID_N,
            view_axis=CORONAL_EMISSION_EXAMPLE_VIEW_AXIS,
            width=2.0 * LOS_EXAMPLE_SIDE_LENGTH_R,
            height=2.0 * LOS_EXAMPLE_SIDE_LENGTH_R,
        )
        band_npz_full = output_dir / f"{prefix}.{band_name}_los_example_full.npz"
        save_los_colormesh_npz(
            band_npz_full,
            band_image,
            band_extent,
            band_counts,
            view_axis=CORONAL_EMISSION_EXAMPLE_VIEW_AXIS,
        )
        band_npz = output_dir / f"{prefix}.{band_name}_los_example.npz"
        save_example_los_colormesh_npz(
            band_npz_full,
            band_npz,
            side_length_r=LOS_EXAMPLE_SIDE_LENGTH_R,
            colorbar_label=CORONAL_EMISSION_BAND_LABELS[band_name],
            unit="W/m^2/sr",
        )
        band_png = output_dir / f"{prefix}.{band_name}_los_example.png"
        plot_example_los_colormesh_npz(band_npz, band_png)
        if band_name == "hard":
            band_field_lines_png = output_dir / f"{prefix}.{band_name}_los_example_field_lines.png"
            plot_example_los_colormesh_field_lines_npz(
                band_npz,
                overlay_field_lines,
                band_field_lines_png,
                title="Hard X-ray with magnetic field lines",
            )
            add_record(
                f"volume_{band_name}_los_example_field_lines_png %r",
                str(band_field_lines_png.relative_to(parent_dir)),
            )
        add_record(f"volume_{band_name}_los_example_npz %r", str(band_npz.relative_to(parent_dir)))
        add_record(f"volume_{band_name}_los_example_png %r", str(band_png.relative_to(parent_dir)))
        add_record(f"volume_{band_name}_los_example_response %r", str(response_path))

        directional_image, directional_extent, _ = render_rho2_los_image(
            tracer,
            band_interp,
            bounds_r,
            path_length_scale=body_radius_m,
            image_n=CORONAL_EMISSION_TOTALS_IMAGE_N,
            view_axis=CORONAL_EMISSION_SINGLE_DIRECTION_VIEW_AXIS,
        )
        directional_radiant_intensity = integrate_image_radiant_intensity(directional_image, directional_extent, body_radius_m)
        point_unblocked_luminosity_density = raw_band_emissivities[band_name] * point_unblocked_solid_angle
        point_four_pi_luminosity_density = raw_band_emissivities[band_name] * (4.0 * np.pi)
        radius_r, unblocked_shell_total, unblocked_cumulative_fraction = radial_emission_profile_exact_rpa(
            tree,
            point_unblocked_luminosity_density,
            length_scale=body_radius_m,
        )
        unblocked_total = float(np.sum(unblocked_shell_total))
        four_pi_total = float(
            np.sum(
                radial_emission_profile_exact_rpa(
                    tree,
                    point_four_pi_luminosity_density,
                    length_scale=body_radius_m,
                )[1]
            )
        )
        r90_r = cumulative_radius_exact_rpa(
            tree,
            point_unblocked_luminosity_density,
            0.90,
            length_scale=body_radius_m,
        )
        r99_r = cumulative_radius_exact_rpa(
            tree,
            point_unblocked_luminosity_density,
            0.99,
            length_scale=body_radius_m,
        )
        coronal_emission_band_stats[band_name] = {
            "directional_radiant_intensity": directional_radiant_intensity,
            "unblocked_total": unblocked_total,
            "four_pi_total": four_pi_total,
            "unblocked_over_four_pi": unblocked_total / four_pi_total if four_pi_total > 0.0 else float("nan"),
            "r90_r": r90_r,
            "r99_r": r99_r,
        }
        coronal_emission_band_profiles[band_name] = (radius_r, unblocked_shell_total, unblocked_cumulative_fraction)
        add_record(f"volume_{band_name}_directional_radiant_intensity_w_sr %r", directional_radiant_intensity)
        add_record(f"volume_{band_name}_unblocked_luminosity_w %r", unblocked_total)
        add_record(f"volume_{band_name}_four_pi_luminosity_w %r", four_pi_total)
        add_record(f"volume_{band_name}_r90_R %r", r90_r)
        add_record(f"volume_{band_name}_r99_R %r", r99_r)
        add_record(f"volume_{band_name}_image_intensity_unit %r", CORONAL_EMISSION_IMAGE_INTENSITY_UNIT)
        add_record(f"volume_{band_name}_radiant_intensity_unit %r", CORONAL_EMISSION_RADIANT_INTENSITY_UNIT)
        add_record(f"volume_{band_name}_emissivity_unit %r", CORONAL_EMISSION_EMISSIVITY_UNIT)
        add_record(f"volume_{band_name}_luminosity_unit %r", CORONAL_EMISSION_LUMINOSITY_UNIT)
    coronal_emission_summary_npz = output_dir / f"{prefix}.coronal_emission_summary.npz"
    np.savez_compressed(
        coronal_emission_summary_npz,
        bands=np.asarray(CORONAL_EMISSION_BAND_NAMES),
        directional_radiant_intensity_w_sr=np.asarray(
            [coronal_emission_band_stats[name]["directional_radiant_intensity"] for name in CORONAL_EMISSION_BAND_NAMES]
        ),
        unblocked_luminosity_w=np.asarray(
            [coronal_emission_band_stats[name]["unblocked_total"] for name in CORONAL_EMISSION_BAND_NAMES]
        ),
        four_pi_luminosity_w=np.asarray(
            [coronal_emission_band_stats[name]["four_pi_total"] for name in CORONAL_EMISSION_BAND_NAMES]
        ),
        r90_r=np.asarray([coronal_emission_band_stats[name]["r90_r"] for name in CORONAL_EMISSION_BAND_NAMES]),
        r99_r=np.asarray([coronal_emission_band_stats[name]["r99_r"] for name in CORONAL_EMISSION_BAND_NAMES]),
        image_intensity_unit=np.asarray(CORONAL_EMISSION_IMAGE_INTENSITY_UNIT),
        directional_radiant_intensity_unit=np.asarray(CORONAL_EMISSION_RADIANT_INTENSITY_UNIT),
        luminosity_unit=np.asarray(CORONAL_EMISSION_LUMINOSITY_UNIT),
        emissivity_unit=np.asarray(CORONAL_EMISSION_EMISSIVITY_UNIT),
        radius_unit=np.asarray(r"$R_\star$"),
        view_axis=np.asarray(CORONAL_EMISSION_SINGLE_DIRECTION_VIEW_AXIS),
    )
    coronal_emission_radial_png = output_dir / f"{prefix}.coronal_emission_radial_summary.png"
    plot_coronal_emission_radial_summary(
        coronal_emission_radial_png,
        coronal_emission_band_profiles,
        coronal_emission_band_stats,
    )
    coronal_emission_units_png = output_dir / f"{prefix}.coronal_emission_unit_summary.png"
    plot_coronal_emission_unit_summary(
        coronal_emission_units_png,
        coronal_emission_band_stats,
        view_axis=CORONAL_EMISSION_SINGLE_DIRECTION_VIEW_AXIS,
    )
    add_record("volume_coronal_emission_summary_npz %r", str(coronal_emission_summary_npz.relative_to(parent_dir)))
    add_record("volume_coronal_emission_radial_summary_png %r", str(coronal_emission_radial_png.relative_to(parent_dir)))
    add_record("volume_coronal_emission_unit_summary_png %r", str(coronal_emission_units_png.relative_to(parent_dir)))
    add_record("volume_rho2_los_npz %r", str(los_npz.relative_to(parent_dir)))
    add_record("volume_rho2_los_png %r", str(los_png.relative_to(parent_dir)))
    add_record("volume_rho2_los_side_npz %r", str(los_side_npz.relative_to(parent_dir)))
    add_record("volume_rho2_los_side_png %r", str(los_side_png.relative_to(parent_dir)))
    add_record("volume_rho2_los_example_npz %r", str(los_example_npz.relative_to(parent_dir)))
    add_record("volume_rho2_los_example_png %r", str(los_example_png.relative_to(parent_dir)))
    add_record("volume_rho2_los_image_n %r", image_n)
    add_record("volume_rho2_los_view_axis %r", "+Z")
    add_record("volume_rho2_los_side_view_axis %r", "+X")
    add_record("volume_rho2_los_example_view_axis %r", "+Y")
    add_record("volume_rho2_los_unit %r", "kg^2/m^5")
    add_record("volume_rho2_los_nonempty_rays %r", int(np.count_nonzero(np.asarray(los_counts) > 0)))
    add_record("volume_rho2_los_side_nonempty_rays %r", int(np.count_nonzero(np.asarray(los_side_counts) > 0)))
    add_record("volume_rho2_los_example_nonempty_rays %r", int(np.count_nonzero(np.asarray(los_example_counts) > 0)))
    log.debug("Rendering LOS rho^2 images complete in %.2f s.", perf_counter() - stage_start)


def process_plt_file(file_path: str | Path) -> None:
    """Process one 3D `.plt` file into recorded diagnostics and plot artifacts."""
    stage_start = perf_counter()
    log.info("Resolving volume pipeline paths...")
    path = Path(file_path)
    output_dir = path.parent / "volume"
    prefix = output_prefix_from_input_file(path.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("%s", path.name)
    log.debug("Resolving volume pipeline paths complete in %.2f s.", perf_counter() - stage_start)

    stage_start = perf_counter()
    log.info("Loading volume dataset...")
    smart_ds = SmartDs.from_file(path, batsrus=True, spherical=True)
    log.debug("Loading volume dataset complete in %.2f s.", perf_counter() - stage_start)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    shells, shell_radii, has_energy_source = sample_shell_grid(smart_ds, DEFAULT_QUICKLOOK_RADII_R)

    record_wind_mass_loss(axes[0, 0], shell_radii, shells)
    record_wind_torque(axes[0, 1], shell_radii, shells)
    record_open_magnetic_flux(axes[1, 0], shell_radii, shells)
    record_energy_flux(axes[1, 1], shell_radii, shells, has_energy_source)

    max_alfven_radius = record_3d_quantities(smart_ds)
    save_field_line_surface_plots(smart_ds, output_dir, prefix, path.parent, max_alfven_radius)
    save_los_images(smart_ds, output_dir, prefix, path.parent, max_alfven_radius)

    stage_start = perf_counter()
    log.info("Saving volume figure...")
    shell_png = output_dir / f"{prefix}.shells.png"
    fig.savefig(shell_png)
    plt.close(fig)
    add_record("volume_shell_png %r", str(shell_png.relative_to(path.parent)))
    log.debug("Saving volume figure complete in %.2f s.", perf_counter() - stage_start)
