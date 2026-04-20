"""Per-file 3D volume pipeline for `batwind-pipe`."""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.colors import LogNorm

from batwind.analysis.shells import integrate_shell_scalar
from batwind.analysis.shells import sample_spherical_shells_fibonacci
from batwind.constants import DEFAULT_QUICKLOOK_RADII_R
from batwind.data.field_names import DEFAULT_XYZ_NAMES
from batwind.pipelines.utils import output_prefix_from_input_file
from batwind.pyvista.field_lines import (
    build_closed_field_line_max_radius_surface,
    build_field_line_max_radius_surface,
    field_line_max_radius_map,
    open_flux_and_area_fractions,
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
FIELDLINE_FRACTION_N_SEEDS = 1000
ANGULAR_MAP_N_POLAR = 18
ANGULAR_MAP_N_AZIMUTH = 36
SURFACE_RENDER_N_POLAR = 36
SURFACE_RENDER_N_AZIMUTH = 72
SURFACE_VIEWPORT_FIGSIZE = (7.0, 7.0)
SURFACE_VIEWPORT_DPI = 180
SURFACE_VIEWPORT_RENDER_SIZE = (1400, 1400)


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


def save_fake_los_image(
    smart_ds: SmartDs,
    output_dir: Path,
    prefix: str,
    parent_dir: Path,
) -> None:
    stage_start = perf_counter()
    log.info("Computing fake LOS rho^2 image...")
    x = np.asarray(smart_ds["X [R]"], dtype=float)
    y = np.asarray(smart_ds["Y [R]"], dtype=float)
    z = np.asarray(smart_ds["Z [R]"], dtype=float)
    x_lin = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), LOS_GRID_N)
    y_lin = np.linspace(float(np.nanmin(y)), float(np.nanmax(y)), LOS_GRID_N)
    z_lin = np.linspace(float(np.nanmin(z)), float(np.nanmax(z)), LOS_GRID_N)
    grid_x, grid_y, grid_z = np.meshgrid(x_lin, y_lin, z_lin, indexing="ij")
    cube_points = np.stack([grid_x, grid_y, grid_z], axis=-1)
    los_cube = smart_ds.resample(
        cube_points,
        coordinate_fields=DEFAULT_XYZ_NAMES,
        fields=smart_ds.source_fields(("Rho [kg/m^3]",)),
        method="octree",
    )
    rho_sq_los = np.nansum(np.asarray(los_cube["Rho [kg/m^3]"], dtype=float) ** 2, axis=2)
    positive = rho_sq_los[np.isfinite(rho_sq_los) & (rho_sq_los > 0.0)]
    los_fig, los_ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    los_norm = LogNorm(vmin=float(np.nanmin(positive)), vmax=float(np.nanmax(positive))) if positive.size else None
    image = los_ax.imshow(
        rho_sq_los.T,
        origin="lower",
        extent=(x_lin[0], x_lin[-1], y_lin[0], y_lin[-1]),
        cmap="magma",
        norm=los_norm,
        aspect="equal",
    )
    los_ax.set_title(r"Fake LOS $\rho^2$")
    los_ax.set_xlabel("X [R]")
    los_ax.set_ylabel("Y [R]")
    los_fig.colorbar(image, ax=los_ax, label=r"$\sum \rho^2$")
    los_png = output_dir / f"{prefix}.rho2_los.png"
    los_fig.savefig(los_png)
    plt.close(los_fig)
    add_record("volume_rho2_los_png %r", str(los_png.relative_to(parent_dir)))
    add_record("volume_rho2_los_grid_n %r", LOS_GRID_N)
    log.debug("Computing fake LOS rho^2 image complete in %.2f s.", perf_counter() - stage_start)


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
    save_fake_los_image(smart_ds, output_dir, prefix, path.parent)

    stage_start = perf_counter()
    log.info("Saving volume figure...")
    shell_png = output_dir / f"{prefix}.shells.png"
    fig.savefig(shell_png)
    plt.close(fig)
    add_record("volume_shell_png %r", str(shell_png.relative_to(path.parent)))
    log.debug("Saving volume figure complete in %.2f s.", perf_counter() - stage_start)
