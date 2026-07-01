"""Per-file shell pipeline for `batwind-pipe` (minimal, user-serviceable)."""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from batwind.pipelines.utils import output_prefix_from_input_file
from batwind.smart_ds import SmartDs

log = logging.getLogger(__name__)
# Method for recording structured, machine-ingested pipeline payloads.
add_record = logging.getLogger(f"recorder.{__name__}").debug


def raw_field_available(smart_ds: SmartDs, name: str) -> bool:
    """Return whether one field is present in the raw dataset."""
    raw = getattr(smart_ds, "raw", None)
    variables = getattr(raw, "variables", None)
    if variables is not None:
        return name in variables
    try:
        smart_ds[name]
    except (IndexError, KeyError):
        return False
    return True


def shell_cell_values(
    values,
    *,
    shell_mask,
    lon_deg,
    lat_deg,
    lon_nodes,
    lat_nodes,
):
    """Convert nodal shell values on corner nodes into explicit cell values."""
    node_values = np.full((lat_nodes.size, lon_nodes.size), np.nan)
    lat_index = np.searchsorted(lat_nodes, np.round(lat_deg[shell_mask], 10))
    lon_index = np.searchsorted(lon_nodes, np.round(lon_deg[shell_mask], 10))
    node_values[lat_index, lon_index] = values[shell_mask]
    return 0.25 * (
        node_values[:-1, :-1]
        + node_values[1:, :-1]
        + node_values[:-1, 1:]
        + node_values[1:, 1:]
    )


def load_shell_coordinates(smart_ds: SmartDs):
    """Load native shell longitude/latitude coordinates and sorted node axes."""
    lon_all = np.ravel(smart_ds["Lon [deg]"])
    lat_all = np.ravel(smart_ds["Lat [deg]"])

    lon_nodes = np.unique(np.round(lon_all, 10))
    lat_nodes = np.unique(np.round(lat_all, 10))
    return lon_all, lat_all, lon_nodes, lat_nodes


def load_shell_grid(smart_ds: SmartDs):
    """Load native shell coordinates and precompute masks and cell areas."""
    r_all = np.ravel(smart_ds["R [R]"])
    star_radius_m = float(smart_ds["RBODY [m]"])
    lon_all, lat_all, lon_nodes, lat_nodes = load_shell_coordinates(smart_ds)
    shell_radii_r = [float(radius_r) for radius_r in np.unique(np.round(r_all, 10))]

    solid_angle = (
        np.sin(np.deg2rad(lat_nodes[1:]))[:, None] - np.sin(np.deg2rad(lat_nodes[:-1]))[:, None]
    ) * np.deg2rad(np.diff(lon_nodes))[None, :]

    shell_masks = []
    shell_areas_m2 = []
    for radius_r in shell_radii_r:
        shell_mask = np.isclose(r_all, radius_r, rtol=0.0, atol=1.0e-10)
        shell_masks.append(shell_mask)
        shell_areas_m2.append((float(radius_r) * star_radius_m) ** 2 * solid_angle)

    height_r = [radius_r - 1.0 for radius_r in shell_radii_r]
    return lon_all, lat_all, shell_radii_r, lon_nodes, lat_nodes, shell_masks, shell_areas_m2, height_r


def shell_surface_map(
    values,
    *,
    lon_deg,
    lat_deg,
    lon_nodes,
    lat_nodes,
):
    """Build one lon/lat cell map from a single-shell nodal field."""
    values = np.ravel(np.asarray(values, dtype=float))
    shell_mask = np.ones(values.shape, dtype=bool)
    return shell_cell_values(
        values,
        shell_mask=shell_mask,
        lon_deg=lon_deg,
        lat_deg=lat_deg,
        lon_nodes=lon_nodes,
        lat_nodes=lat_nodes,
    )


def shell_map_and_profile(
    values,
    *,
    shell_masks,
    shell_areas_m2,
    lon_deg,
    lat_deg,
    lon_nodes,
    lat_nodes,
):
    """Build outer-shell cell map and integrated radial profile for one field."""
    integrated_values = []
    for shell_mask, area in zip(shell_masks, shell_areas_m2):
        shell_cells = shell_cell_values(
            values,
            shell_mask=shell_mask,
            lon_deg=lon_deg,
            lat_deg=lat_deg,
            lon_nodes=lon_nodes,
            lat_nodes=lat_nodes,
        )
        integrated_values.append(float(np.sum(shell_cells * area)))

    outer_map = shell_cell_values(
        values,
        shell_mask=shell_masks[-1],
        lon_deg=lon_deg,
        lat_deg=lat_deg,
        lon_nodes=lon_nodes,
        lat_nodes=lat_nodes,
    )
    return outer_map, integrated_values


def plot_shell_component_stack_png(
    component_maps: tuple[tuple[np.ndarray, str, str], ...],
    *,
    lon_nodes,
    lat_nodes,
    output_path: Path,
) -> None:
    """Plot stacked magnetic shell maps to one PNG."""
    figure, axes = plt.subplots(len(component_maps), 1, figsize=(7, 12), constrained_layout=True, sharex=True)
    axes = np.atleast_1d(axes)
    for axis, (map_values, title, colorbar_label) in zip(axes, component_maps, strict=True):
        plot_kwargs = {"shading": "flat", "cmap": "RdBu_r"}
        limit = float(np.nanmax(np.abs(np.asarray(map_values, dtype=float))))
        if np.isfinite(limit) and limit > 0.0:
            plot_kwargs |= {"vmin": -limit, "vmax": limit}
        image = axis.pcolormesh(lon_nodes, lat_nodes, map_values, **plot_kwargs)
        axis.set_ylabel("Latitude [deg]")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, label=colorbar_label)
    axes[-1].set_xlabel("Longitude [deg]")
    figure.savefig(output_path)
    plt.close(figure)


def process_magnetic_shell_file(
    smart_ds: SmartDs,
    *,
    path: Path,
    output_dir: Path,
    prefix: str,
) -> None:
    """Plot native magnetic shell components over longitude and latitude."""
    stage_start = perf_counter()
    log.info("Computing shell magnetic component maps...")
    lon_all, lat_all, lon_nodes, lat_nodes = load_shell_coordinates(smart_ds)
    component_specs = (
        ("B_r [T]", "Radial Magnetic Field", "B_r [T]"),
        ("bphi [T]", "Azimuthal Magnetic Field", "B_phi [T]"),
        ("btheta [T]", "Meridional Magnetic Field", "B_theta [T]"),
    )
    component_maps = []
    for field_name, title, colorbar_label in component_specs:
        component_maps.append((
            shell_surface_map(
            smart_ds[field_name],
            lon_deg=lon_all,
            lat_deg=lat_all,
            lon_nodes=lon_nodes,
            lat_nodes=lat_nodes,
            ),
            title,
            colorbar_label,
        ))
    output_path = output_dir / f"{prefix}.magnetic_components.png"
    plot_shell_component_stack_png(
        tuple(component_maps),
        lon_nodes=lon_nodes,
        lat_nodes=lat_nodes,
        output_path=output_path,
    )
    add_record("shell_magnetic_components_png %r", str(output_path.relative_to(path.parent)))
    log.debug("Computing shell magnetic component maps complete in %.2f s.", perf_counter() - stage_start)


def process_plt_file(file_path: str | Path) -> None:
    """Process one shell-like file into maps, profiles, and recorded diagnostics."""
    # Start: resolve input/output paths and log file.
    stage_start = perf_counter()
    log.info("Resolving shell pipeline paths...")
    path = Path(file_path)
    output_dir = path.parent / "shell"
    prefix = output_prefix_from_input_file(path.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("%s", path.name)
    log.debug("Resolving shell pipeline paths complete in %.2f s.", perf_counter() - stage_start)

    # Start: load dataset, attach the graph-backed fields, and build shell geometry.
    stage_start = perf_counter()
    log.info("Loading shell dataset and preparing native shell grid...")
    smart_ds = SmartDs.from_file(path, batsrus=True, spherical=True)
    if not raw_field_available(smart_ds, "R [R]"):
        process_magnetic_shell_file(
            smart_ds,
            path=path,
            output_dir=output_dir,
            prefix=prefix,
        )
        log.debug("Loading shell dataset and preparing native shell grid complete in %.2f s.", perf_counter() - stage_start)
        return
    lon_all, lat_all, shell_radii_r, lon_nodes, lat_nodes, shell_masks, shell_areas_m2, height_r = load_shell_grid(smart_ds)
    log.debug("Loading shell dataset and preparing native shell grid complete in %.2f s.", perf_counter() - stage_start)

    # Start: compute, plot, and record shell wind mass flux.
    stage_start = perf_counter()
    log.info("Computing shell wind mass flux...")
    mass_flux = np.ravel(smart_ds["mass_flux [kg/m^2/s]"])
    mass_flux_map, mass_loss_kg_s = shell_map_and_profile(
        mass_flux,
        shell_masks=shell_masks,
        shell_areas_m2=shell_areas_m2,
        lon_deg=lon_all,
        lat_deg=lat_all,
        lon_nodes=lon_nodes,
        lat_nodes=lat_nodes,
    )

    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    image = axis.pcolormesh(lon_nodes, lat_nodes, mass_flux_map, shading="flat", cmap="viridis")
    axis.set_xlabel("Longitude [deg]")
    axis.set_ylabel("Latitude [deg]")
    axis.set_title("Wind Mass Flux")
    figure.colorbar(image, ax=axis, label="Mass flux [kg/m^2/s]")
    mass_flux_png = output_dir / f"{prefix}.mass_flux_map.png"
    figure.savefig(mass_flux_png)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    axis.plot(height_r, mass_loss_kg_s, ".-", color="C0")
    axis.set_xlabel("Height [R]")
    axis.set_ylabel("Mass loss [kg/s]")
    axis.set_title("Wind Mass Loss")
    axis.grid(True, alpha=0.25)
    mass_loss_png = output_dir / f"{prefix}.mass_loss_profile.png"
    figure.savefig(mass_loss_png)
    plt.close(figure)

    add_record("shell_mass_flux_map_png %r", str(mass_flux_png.relative_to(path.parent)))
    add_record("shell_mass_loss_profile_png %r", str(mass_loss_png.relative_to(path.parent)))
    add_record("shell_radius_R %r", shell_radii_r)
    add_record("shell_mass_loss_kg_s %r", mass_loss_kg_s)
    add_record("shell_mass_loss_value_kg_s %r", mass_loss_kg_s[-1])
    log.debug("Computing shell wind mass flux complete in %.2f s.", perf_counter() - stage_start)

    # Start: compute, plot, and record shell angular momentum flux.
    stage_start = perf_counter()
    log.info("Computing shell angular momentum flux...")
    torque_density = np.ravel(smart_ds["total_torque_density [N/m]"])
    torque_map, total_torque_nm = shell_map_and_profile(
        torque_density,
        shell_masks=shell_masks,
        shell_areas_m2=shell_areas_m2,
        lon_deg=lon_all,
        lat_deg=lat_all,
        lon_nodes=lon_nodes,
        lat_nodes=lat_nodes,
    )

    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    image = axis.pcolormesh(lon_nodes, lat_nodes, torque_map, shading="flat", cmap="cividis")
    axis.set_xlabel("Longitude [deg]")
    axis.set_ylabel("Latitude [deg]")
    axis.set_title("Angular Momentum Flux")
    figure.colorbar(image, ax=axis, label="Torque density [N/m]")
    torque_map_png = output_dir / f"{prefix}.torque_map.png"
    figure.savefig(torque_map_png)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    axis.plot(height_r, total_torque_nm, ".-", color="C1")
    axis.set_xlabel("Height [R]")
    axis.set_ylabel("Torque [Nm]")
    axis.set_title("Wind Torque")
    axis.grid(True, alpha=0.25)
    torque_profile_png = output_dir / f"{prefix}.torque_profile.png"
    figure.savefig(torque_profile_png)
    plt.close(figure)

    add_record("shell_torque_map_png %r", str(torque_map_png.relative_to(path.parent)))
    add_record("shell_torque_profile_png %r", str(torque_profile_png.relative_to(path.parent)))
    add_record("shell_total_torque_nm %r", total_torque_nm)
    add_record("shell_total_torque_value_nm %r", total_torque_nm[-1])
    log.debug("Computing shell angular momentum flux complete in %.2f s.", perf_counter() - stage_start)

    # Start: compute, plot, and record shell energy flux.
    stage_start = perf_counter()
    log.info("Computing shell energy flux...")
    energy_flux = np.ravel(smart_ds["energy_flux [W/m^2]"])
    energy_map, energy_flow_w = shell_map_and_profile(
        energy_flux,
        shell_masks=shell_masks,
        shell_areas_m2=shell_areas_m2,
        lon_deg=lon_all,
        lat_deg=lat_all,
        lon_nodes=lon_nodes,
        lat_nodes=lat_nodes,
    )

    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    image = axis.pcolormesh(lon_nodes, lat_nodes, energy_map, shading="flat", cmap="plasma")
    axis.set_xlabel("Longitude [deg]")
    axis.set_ylabel("Latitude [deg]")
    axis.set_title("Energy Flux")
    figure.colorbar(image, ax=axis, label="Energy flux [W/m^2]")
    energy_map_png = output_dir / f"{prefix}.energy_flux_map.png"
    figure.savefig(energy_map_png)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    axis.plot(height_r, energy_flow_w, ".-", color="C2")
    axis.set_xlabel("Height [R]")
    axis.set_ylabel("Energy flow [W]")
    axis.set_title("Shell Energy Flow")
    axis.grid(True, alpha=0.25)
    energy_profile_png = output_dir / f"{prefix}.energy_flow_profile.png"
    figure.savefig(energy_profile_png)
    plt.close(figure)

    add_record("shell_energy_flux_map_png %r", str(energy_map_png.relative_to(path.parent)))
    add_record("shell_energy_flow_profile_png %r", str(energy_profile_png.relative_to(path.parent)))
    add_record("shell_energy_flow_w %r", energy_flow_w)
    add_record("shell_energy_flow_value_w %r", energy_flow_w[-1])
    log.debug("Computing shell energy flux complete in %.2f s.", perf_counter() - stage_start)

    # Start: compute, plot, and record shell open magnetic flux.
    stage_start = perf_counter()
    log.info("Computing shell open magnetic flux...")
    open_flux_density = np.abs(np.ravel(smart_ds["B_r [T]"]))
    open_flux_map, open_flux_wb = shell_map_and_profile(
        open_flux_density,
        shell_masks=shell_masks,
        shell_areas_m2=shell_areas_m2,
        lon_deg=lon_all,
        lat_deg=lat_all,
        lon_nodes=lon_nodes,
        lat_nodes=lat_nodes,
    )

    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    image = axis.pcolormesh(lon_nodes, lat_nodes, open_flux_map, shading="flat", cmap="magma")
    axis.set_xlabel("Longitude [deg]")
    axis.set_ylabel("Latitude [deg]")
    axis.set_title("Open Magnetic Flux Density")
    figure.colorbar(image, ax=axis, label="|B_r| [T]")
    open_flux_map_png = output_dir / f"{prefix}.open_flux_map.png"
    figure.savefig(open_flux_map_png)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    axis.plot(height_r, open_flux_wb, ".-", color="C3")
    axis.set_xlabel("Height [R]")
    axis.set_ylabel("Open flux [Wb]")
    axis.set_title("Open Magnetic Flux")
    axis.grid(True, alpha=0.25)
    open_flux_profile_png = output_dir / f"{prefix}.open_flux_profile.png"
    figure.savefig(open_flux_profile_png)
    plt.close(figure)

    add_record("shell_open_flux_map_png %r", str(open_flux_map_png.relative_to(path.parent)))
    add_record("shell_open_flux_profile_png %r", str(open_flux_profile_png.relative_to(path.parent)))
    add_record("shell_open_flux_wb %r", open_flux_wb)
    add_record("shell_open_flux_value_wb %r", open_flux_wb[-1])
    log.debug("Computing shell open magnetic flux complete in %.2f s.", perf_counter() - stage_start)
