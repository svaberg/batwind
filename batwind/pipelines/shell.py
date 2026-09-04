"""Per-file shell pipeline for `batwind-pipe` (minimal, user-serviceable)."""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from batwind.pipelines.utils import annotate_iteration_axis
from batwind.pipelines.utils import output_prefix_from_input_file
from batwind.smart_ds import SmartDs

log = logging.getLogger(__name__)
# Method for recording structured, machine-ingested pipeline payloads.
add_record = logging.getLogger(f"recorder.{__name__}").debug


def load_shell_axis_indices(smart_ds: SmartDs, name: str) -> tuple[np.ndarray, int]:
    """Load one shell logical index axis and require contiguous 1-based values."""
    axis = np.ravel(np.asarray(smart_ds[name], dtype=int))
    size = int(np.max(axis))
    expected = np.arange(1, size + 1, dtype=int)
    if not np.array_equal(np.unique(axis), expected):
        raise ValueError(f"Expected contiguous 1-based shell {name} indices")
    return axis, size


def unique_values_in_order(values) -> np.ndarray:
    """Return unique values in first-appearance order."""
    values = np.ravel(np.asarray(values, dtype=float))
    unique_values, first_indices = np.unique(values, return_index=True)
    return unique_values[np.argsort(first_indices)]


def is_unit_radius_shell(smart_ds: SmartDs, *, atol: float = 1.0e-10) -> bool:
    """Return whether one shell dataset lies entirely on the unit-radius surface."""
    radius = np.ravel(np.asarray(smart_ds["R [R]"], dtype=float))
    return bool(np.allclose(radius, 1.0, rtol=0.0, atol=float(atol)))


def shell_cell_values(node_values):
    """Convert one structured nodal shell field into explicit cell values."""
    node_values = np.asarray(node_values, dtype=float)
    return 0.25 * (
        node_values[:-1, :-1]
        + node_values[1:, :-1]
        + node_values[:-1, 1:]
        + node_values[1:, 1:]
    )


def load_magnetic_shell_grid(smart_ds: SmartDs):
    """Load one magnetic shell grid and its longitude/latitude nodes."""
    lon_all = np.asarray(smart_ds["Lon [deg]"], dtype=float).reshape(-1)
    lat_all = np.asarray(smart_ds["Lat [deg]"], dtype=float).reshape(-1)
    variables = set(smart_ds.raw.variables)
    if {"I", "J", "K"} <= variables:
        i_all, n_radius = load_shell_axis_indices(smart_ds, "I")
        j_all, n_lon = load_shell_axis_indices(smart_ds, "J")
        k_all, n_lat = load_shell_axis_indices(smart_ds, "K")
        if n_radius != 1:
            raise ValueError("Expected magnetic shell to contain exactly one radial index")
        expected_i = np.ones(n_lon * n_lat, dtype=int)
        expected_j = np.tile(np.arange(1, n_lon + 1, dtype=int), n_lat)
        expected_k = np.repeat(np.arange(1, n_lat + 1, dtype=int), n_lon)
        if not np.array_equal(i_all, expected_i) or not np.array_equal(j_all, expected_j) or not np.array_equal(k_all, expected_k):
            raise ValueError("Expected magnetic shell rows ordered with I fixed, J fastest, and K slowest")
    elif {"I", "J"} <= variables:
        i_all, n_lon = load_shell_axis_indices(smart_ds, "I")
        j_all, n_lat = load_shell_axis_indices(smart_ds, "J")
        expected_i = np.tile(np.arange(1, n_lon + 1, dtype=int), n_lat)
        expected_j = np.repeat(np.arange(1, n_lat + 1, dtype=int), n_lon)
        if not np.array_equal(i_all, expected_i) or not np.array_equal(j_all, expected_j):
            raise ValueError("Expected magnetic shell rows ordered with I fastest and J slowest")
    else:
        log.info("Reading shell longitude/latitude layout from longitude/latitude...")
        n_lon = unique_values_in_order(lon_all).size
        n_lat = unique_values_in_order(lat_all).size

    grid_shape = (n_lat, n_lon)
    lon_grid = lon_all.reshape(grid_shape)
    lat_grid = lat_all.reshape(grid_shape)
    lon_nodes = lon_grid[0, :].astype(float)
    lat_nodes = lat_grid[:, 0].astype(float)
    if not np.allclose(lon_grid, lon_nodes[None, :], rtol=0.0, atol=1.0e-10):
        raise ValueError("Expected magnetic shell longitude to vary only along grid columns")
    if not np.allclose(lat_grid, lat_nodes[:, None], rtol=0.0, atol=1.0e-10):
        raise ValueError("Expected magnetic shell latitude to vary only along grid rows")
    return grid_shape, lon_nodes, lat_nodes


def load_shell_grid(smart_ds: SmartDs):
    """Load one shell grid and precompute per-shell cell areas."""
    r_all = np.asarray(smart_ds["R [R]"], dtype=float).reshape(-1)
    lon_all = np.asarray(smart_ds["Lon [deg]"], dtype=float).reshape(-1)
    lat_all = np.asarray(smart_ds["Lat [deg]"], dtype=float).reshape(-1)
    if {"I", "J", "K"} <= set(smart_ds.raw.variables):
        i_all, n_radius = load_shell_axis_indices(smart_ds, "I")
        j_all, n_lon = load_shell_axis_indices(smart_ds, "J")
        k_all, n_lat = load_shell_axis_indices(smart_ds, "K")
        expected_i = np.tile(np.arange(1, n_radius + 1, dtype=int), n_lon * n_lat)
        expected_j = np.tile(np.repeat(np.arange(1, n_lon + 1, dtype=int), n_radius), n_lat)
        expected_k = np.repeat(np.arange(1, n_lat + 1, dtype=int), n_radius * n_lon)
        if not np.array_equal(i_all, expected_i) or not np.array_equal(j_all, expected_j) or not np.array_equal(k_all, expected_k):
            raise ValueError("Expected shell rows ordered with I fastest, then J, then K")
    else:
        log.info("Reading shell grid layout from radius/longitude/latitude...")
        n_radius = unique_values_in_order(r_all).size
        n_lon = unique_values_in_order(lon_all).size
        n_lat = unique_values_in_order(lat_all).size

    grid_shape = (n_lat, n_lon, n_radius)
    r_grid = r_all.reshape(grid_shape)
    lon_grid = lon_all.reshape(grid_shape)
    lat_grid = lat_all.reshape(grid_shape)
    shell_radii_r = r_grid[0, 0, :].astype(float)
    lon_nodes = lon_grid[0, :, 0].astype(float)
    lat_nodes = lat_grid[:, 0, 0].astype(float)
    if not np.allclose(r_grid, shell_radii_r[None, None, :], rtol=0.0, atol=1.0e-10):
        raise ValueError("Expected shell radius to vary only along the radial index")
    if not np.allclose(lon_grid, lon_nodes[None, :, None], rtol=0.0, atol=1.0e-10):
        raise ValueError("Expected shell longitude to vary only along the longitude index")
    if not np.allclose(lat_grid, lat_nodes[:, None, None], rtol=0.0, atol=1.0e-10):
        raise ValueError("Expected shell latitude to vary only along the latitude index")

    star_radius_m = float(np.asarray(smart_ds["RBODY [m]"], dtype=float).reshape(-1)[0])

    solid_angle = (
        np.sin(np.deg2rad(lat_nodes[1:]))[:, None] - np.sin(np.deg2rad(lat_nodes[:-1]))[:, None]
    ) * np.deg2rad(np.diff(lon_nodes))[None, :]

    shell_areas_m2 = [float(radius_r) ** 2 * star_radius_m ** 2 * solid_angle for radius_r in shell_radii_r]
    shell_radii_r = [float(radius_r) for radius_r in shell_radii_r]
    return grid_shape, shell_radii_r, lon_nodes, lat_nodes, shell_areas_m2


def shell_map_and_profile(
    values,
    *,
    grid_shape,
    shell_areas_m2,
):
    """Build outer-shell cell map and integrated radial profile for one field."""
    node_values = np.asarray(values, dtype=float).reshape(grid_shape)
    integrated_values = []
    for radius_index, area in enumerate(shell_areas_m2):
        shell_cells = shell_cell_values(node_values[:, :, radius_index])
        integrated_values.append(float(np.sum(shell_cells * area)))

    outer_map = shell_cell_values(node_values[:, :, -1])
    return outer_map, integrated_values


def plot_shell_component_stack_png(
    component_maps: tuple[tuple[np.ndarray, str, str], ...],
    *,
    source_path: Path,
    lon_nodes,
    lat_nodes,
    output_path: Path,
) -> None:
    """Plot stacked magnetic shell maps to one PNG."""
    figure, axes = plt.subplots(len(component_maps), 1, figsize=(7, 12), constrained_layout=True, sharex=True)
    axes = np.atleast_1d(axes)
    annotate_iteration_axis(axes[0], source_path)
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
    """Plot magnetic shell components over longitude and latitude."""
    stage_start = perf_counter()
    log.info("Computing shell magnetic component maps...")
    grid_shape, lon_nodes, lat_nodes = load_magnetic_shell_grid(smart_ds)
    component_specs = (
        ("B_r [T]", "Radial Magnetic Field", "B_r [T]"),
        ("bphi [T]", "Azimuthal Magnetic Field", "B_phi [T]"),
        ("btheta [T]", "Meridional Magnetic Field", "B_theta [T]"),
    )
    component_maps = []
    for field_name, title, colorbar_label in component_specs:
        component_maps.append((
            shell_cell_values(np.asarray(smart_ds[field_name], dtype=float).reshape(grid_shape)),
            title,
            colorbar_label,
        ))
    output_path = output_dir / f"{prefix}.magnetic_components.png"
    plot_shell_component_stack_png(
        tuple(component_maps),
        source_path=path,
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
    log.info("Loading shell dataset...")
    smart_ds = SmartDs.from_file(path, batsrus=True, spherical=True)
    if is_unit_radius_shell(smart_ds):
        log.info("Using shell longitude/latitude grid...")
        process_magnetic_shell_file(
            smart_ds,
            path=path,
            output_dir=output_dir,
            prefix=prefix,
        )
        log.debug("Loading shell dataset complete in %.2f s.", perf_counter() - stage_start)
        return
    log.info("Using shell radius/longitude/latitude grid...")
    grid_shape, shell_radii_r, lon_nodes, lat_nodes, shell_areas_m2 = load_shell_grid(smart_ds)
    height_r = [radius_r - 1.0 for radius_r in shell_radii_r]
    log.debug("Loading shell dataset complete in %.2f s.", perf_counter() - stage_start)

    # Start: compute, plot, and record shell wind mass flux.
    stage_start = perf_counter()
    log.info("Computing shell wind mass flux...")
    mass_flux = np.ravel(smart_ds["mass_flux [kg/m^2/s]"])
    mass_flux_map, mass_loss_kg_s = shell_map_and_profile(
        mass_flux,
        grid_shape=grid_shape,
        shell_areas_m2=shell_areas_m2,
    )

    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    image = axis.pcolormesh(lon_nodes, lat_nodes, mass_flux_map, shading="flat", cmap="viridis")
    axis.set_xlabel("Longitude [deg]")
    axis.set_ylabel("Latitude [deg]")
    axis.set_title("Wind Mass Flux")
    annotate_iteration_axis(axis, path)
    figure.colorbar(image, ax=axis, label="Mass flux [kg/m^2/s]")
    mass_flux_png = output_dir / f"{prefix}.mass_flux_map.png"
    figure.savefig(mass_flux_png)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    axis.plot(height_r, mass_loss_kg_s, ".-", color="C0")
    axis.set_xlabel("Height [R]")
    axis.set_ylabel("Mass loss [kg/s]")
    axis.set_title("Wind Mass Loss")
    annotate_iteration_axis(axis, path)
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
        grid_shape=grid_shape,
        shell_areas_m2=shell_areas_m2,
    )

    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    image = axis.pcolormesh(lon_nodes, lat_nodes, torque_map, shading="flat", cmap="cividis")
    axis.set_xlabel("Longitude [deg]")
    axis.set_ylabel("Latitude [deg]")
    axis.set_title("Angular Momentum Flux")
    annotate_iteration_axis(axis, path)
    figure.colorbar(image, ax=axis, label="Torque density [N/m]")
    torque_map_png = output_dir / f"{prefix}.torque_map.png"
    figure.savefig(torque_map_png)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    axis.plot(height_r, total_torque_nm, ".-", color="C1")
    axis.set_xlabel("Height [R]")
    axis.set_ylabel("Torque [Nm]")
    axis.set_title("Wind Torque")
    annotate_iteration_axis(axis, path)
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
        grid_shape=grid_shape,
        shell_areas_m2=shell_areas_m2,
    )

    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    image = axis.pcolormesh(lon_nodes, lat_nodes, energy_map, shading="flat", cmap="plasma")
    axis.set_xlabel("Longitude [deg]")
    axis.set_ylabel("Latitude [deg]")
    axis.set_title("Energy Flux")
    annotate_iteration_axis(axis, path)
    figure.colorbar(image, ax=axis, label="Energy flux [W/m^2]")
    energy_map_png = output_dir / f"{prefix}.energy_flux_map.png"
    figure.savefig(energy_map_png)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    axis.plot(height_r, energy_flow_w, ".-", color="C2")
    axis.set_xlabel("Height [R]")
    axis.set_ylabel("Energy flow [W]")
    axis.set_title("Shell Energy Flow")
    annotate_iteration_axis(axis, path)
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
        grid_shape=grid_shape,
        shell_areas_m2=shell_areas_m2,
    )

    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    image = axis.pcolormesh(lon_nodes, lat_nodes, open_flux_map, shading="flat", cmap="magma")
    axis.set_xlabel("Longitude [deg]")
    axis.set_ylabel("Latitude [deg]")
    axis.set_title("Open Magnetic Flux Density")
    annotate_iteration_axis(axis, path)
    figure.colorbar(image, ax=axis, label="|B_r| [T]")
    open_flux_map_png = output_dir / f"{prefix}.open_flux_map.png"
    figure.savefig(open_flux_map_png)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    axis.plot(height_r, open_flux_wb, ".-", color="C3")
    axis.set_xlabel("Height [R]")
    axis.set_ylabel("Open flux [Wb]")
    axis.set_title("Open Magnetic Flux")
    annotate_iteration_axis(axis, path)
    axis.grid(True, alpha=0.25)
    open_flux_profile_png = output_dir / f"{prefix}.open_flux_profile.png"
    figure.savefig(open_flux_profile_png)
    plt.close(figure)

    add_record("shell_open_flux_map_png %r", str(open_flux_map_png.relative_to(path.parent)))
    add_record("shell_open_flux_profile_png %r", str(open_flux_profile_png.relative_to(path.parent)))
    add_record("shell_open_flux_wb %r", open_flux_wb)
    add_record("shell_open_flux_value_wb %r", open_flux_wb[-1])
    log.debug("Computing shell open magnetic flux complete in %.2f s.", perf_counter() - stage_start)
