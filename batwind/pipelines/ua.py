"""Per-file UA/MGITM quicklook pipeline for `batwind-pipe`."""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
import numpy as np

from batwind.data.ua_gitm import read_ua_gitm_bin
from batwind.pipelines.utils import output_prefix_from_input_file
from batwind.recipes.ua import build_ua_graph
from batwind.smart_ds import SmartDs

log = logging.getLogger(__name__)
add_record = logging.getLogger(f"recorder.{__name__}").debug
UA_EXAGGERATED_SHELL_THICKNESS = 0.35
UA_EARTH_RBODY_M = 6372.0e3
UA_SYMLOG_LINTHRESH_PERCENTILE = 20.0
UA_SHELL_FLUX_FIELDS = (
    ("CO2", "CO2 [1/m^3]", "CO2 [kg/m^3]", "Vn_up_CO2 [m/s]"),
    ("CO", "CO [1/m^3]", "CO [kg/m^3]", "Vn_up_CO [m/s]"),
    ("O", "O [1/m^3]", "O [kg/m^3]", "Vn_up_O [m/s]"),
    ("N2", "N2 [1/m^3]", "N2 [kg/m^3]", "Vn_up_N2 [m/s]"),
    ("O2", "O2 [1/m^3]", "O2 [kg/m^3]", "Vn_up_O2 [m/s]"),
    ("Ar", "Ar [1/m^3]", "Ar [kg/m^3]", "Vn_up_Ar [m/s]"),
    ("He", "He [1/m^3]", "He [kg/m^3]", "Vn_up_He [m/s]"),
    ("N", "N [1/m^3]", "N [kg/m^3]", "Vn_up_N [m/s]"),
    ("H", "H [1/m^3]", "H [kg/m^3]", "Vn_up [m/s]"),
    ("N(2D)", "N(2D) [1/m^3]", "N(2D) [kg/m^3]", "Vn_up [m/s]"),
    ("NO", "NO [1/m^3]", "NO [kg/m^3]", "Vn_up [m/s]"),
    ("C", "C [1/m^3]", "C [kg/m^3]", "Vn_up [m/s]"),
    ("O+", "O+ [1/m^3]", "O+ [kg/m^3]", "Vi_up [m/s]"),
    ("O2+", "O2+ [1/m^3]", "O2+ [kg/m^3]", "Vi_up [m/s]"),
    ("CO2+", "CO2+ [1/m^3]", "CO2+ [kg/m^3]", "Vi_up [m/s]"),
    ("N2+", "N2+ [1/m^3]", "N2+ [kg/m^3]", "Vi_up [m/s]"),
    ("NO+", "NO+ [1/m^3]", "NO+ [kg/m^3]", "Vi_up [m/s]"),
    ("CO+", "CO+ [1/m^3]", "CO+ [kg/m^3]", "Vi_up [m/s]"),
    ("C+", "C+ [1/m^3]", "C+ [kg/m^3]", "Vi_up [m/s]"),
)


def _positive_for_log(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.where(arr > 0.0, arr, np.nan)
    if np.all(np.isnan(out)):
        raise ValueError("Requested log-scaled UA slice has no positive values")
    return out


def _cell_edges_1d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError(f"Expected 1D coordinate centers with at least two points, got shape {arr.shape}")
    mids = 0.5 * (arr[:-1] + arr[1:])
    first = arr[0] - 0.5 * (arr[1] - arr[0])
    last = arr[-1] + 0.5 * (arr[-1] - arr[-2])
    return np.concatenate(([first], mids, [last]))


def _physical_latitude_mask(latitude_deg: np.ndarray) -> np.ndarray:
    lat = np.asarray(latitude_deg, dtype=float)
    if lat.ndim != 1:
        raise ValueError(f"Expected 1D latitude axis, got shape {lat.shape}")
    mask = (lat >= -90.0) & (lat <= 90.0)
    if np.count_nonzero(mask) < 2:
        raise ValueError("UA latitude axis does not contain at least two physical latitude rows")
    return mask


def _physical_longitude_mask(longitude_deg: np.ndarray) -> np.ndarray:
    lon = np.asarray(longitude_deg, dtype=float)
    if lon.ndim != 1:
        raise ValueError(f"Expected 1D longitude axis, got shape {lon.shape}")
    mask = (lon >= 0.0) & (lon < 360.0)
    if np.count_nonzero(mask) < 2:
        raise ValueError("UA longitude axis does not contain at least two physical longitude rows")
    return mask


def _physical_altitude_mask(altitude_m: np.ndarray) -> np.ndarray:
    alt = np.asarray(altitude_m, dtype=float)
    if alt.ndim != 1:
        raise ValueError(f"Expected 1D altitude axis, got shape {alt.shape}")
    mask = alt >= 0.0
    if np.count_nonzero(mask) < 1:
        raise ValueError("UA altitude axis does not contain physical altitude shells")
    return mask


def _opposite_longitude_index(longitude_deg: np.ndarray, i_lon: int) -> int:
    lon = np.mod(np.asarray(longitude_deg, dtype=float), 360.0)
    if lon.ndim != 1:
        raise ValueError(f"Expected 1D longitude axis, got shape {lon.shape}")
    target = (float(lon[i_lon]) + 180.0) % 360.0
    delta = np.abs(((lon - target + 180.0) % 360.0) - 180.0)
    return int(np.argmin(delta))


def _shell_area_grid_m2(longitude_deg: np.ndarray, latitude_deg: np.ndarray, radius_m: np.ndarray) -> np.ndarray:
    lon_edges = np.deg2rad(_cell_edges_1d(longitude_deg))
    lat_edges = np.deg2rad(_cell_edges_1d(latitude_deg))
    d_lon = np.diff(lon_edges)
    d_sin_lat = np.diff(np.sin(lat_edges))
    return (
        (np.asarray(radius_m, dtype=float) ** 2)[None, None, :]
        * d_lon[:, None, None]
        * d_sin_lat[None, :, None]
    )


def _available_shell_flux_series(
    smart_ds: SmartDs,
    area_grid_m2: np.ndarray,
    lon_mask: np.ndarray,
    lat_mask: np.ndarray,
    alt_mask: np.ndarray,
):
    species_names: list[str] = []
    number_flux_rows: list[np.ndarray] = []
    mass_flux_rows: list[np.ndarray] = []
    number_flux_density_rows: list[np.ndarray] = []
    mass_flux_density_rows: list[np.ndarray] = []
    for species_name, number_density_field, mass_density_field, velocity_field in UA_SHELL_FLUX_FIELDS:
        try:
            number_density = np.asarray(smart_ds[number_density_field], dtype=float)[lon_mask, :, :][
                :, lat_mask, :
            ][:, :, alt_mask]
            mass_density = np.asarray(smart_ds[mass_density_field], dtype=float)[lon_mask, :, :][
                :, lat_mask, :
            ][:, :, alt_mask]
            velocity = np.asarray(smart_ds[velocity_field], dtype=float)[lon_mask, :, :][:, lat_mask, :][
                :, :, alt_mask
            ]
        except IndexError:
            continue
        number_flux_density = number_density * velocity
        mass_flux_density = mass_density * velocity
        species_names.append(species_name)
        number_flux_rows.append(np.sum(number_flux_density * area_grid_m2, axis=(0, 1)))
        mass_flux_rows.append(np.sum(mass_flux_density * area_grid_m2, axis=(0, 1)))
        number_flux_density_rows.append(number_flux_density)
        mass_flux_density_rows.append(mass_flux_density)
    if not species_names:
        raise ValueError("No UA shell-flux species could be constructed from the available fields")
    return (
        np.array(species_names, dtype="U"),
        np.vstack(number_flux_rows),
        np.vstack(mass_flux_rows),
        np.stack(number_flux_density_rows, axis=0),
        np.stack(mass_flux_density_rows, axis=0),
    )


def _plot_shell_flux_profile(
    altitude_m: np.ndarray,
    species_names: np.ndarray,
    species_flux: np.ndarray,
    total_flux: np.ndarray,
    *,
    ylabel: str,
    title: str,
    png_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 6.0), constrained_layout=True)
    for i_species, species_name in enumerate(species_names):
        values = np.asarray(species_flux[i_species], dtype=float)
        ax.plot(altitude_m, np.where(values > 0.0, values, np.nan), linewidth=1.0, alpha=0.8, label=f"{species_name} out")
        ax.plot(altitude_m, np.where(values < 0.0, -values, np.nan), linewidth=1.0, alpha=0.8, linestyle="--", label=f"{species_name} in")
    total_values = np.asarray(total_flux, dtype=float)
    ax.plot(altitude_m, np.where(total_values > 0.0, total_values, np.nan), color="k", linewidth=2.5, label="total out")
    ax.plot(
        altitude_m,
        np.where(total_values < 0.0, -total_values, np.nan),
        color="k",
        linewidth=2.5,
        linestyle="--",
        label="total in",
    )
    ax.set_xlabel("Altitude [m]")
    ax.set_ylabel(f"|{ylabel}|")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize="small")
    fig.savefig(png_path)
    plt.close(fig)


def _symlog_norm(values: np.ndarray) -> SymLogNorm:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("Requested shell flux map has no finite values")
    vmax = float(np.max(np.abs(finite)))
    positive = np.abs(finite[np.abs(finite) > 0.0])
    if positive.size == 0:
        linthresh = 1.0
        vmax = 1.0
    else:
        linthresh = float(np.percentile(positive, UA_SYMLOG_LINTHRESH_PERCENTILE))
    return SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax)


def _plot_shell_flux_map(
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
    total_number_flux_density: np.ndarray,
    total_mass_flux_density: np.ndarray,
    *,
    altitude_m: float,
    title_prefix: str,
    png_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)

    mesh = axes[0].pcolormesh(
        longitude_deg,
        latitude_deg,
        total_number_flux_density.T,
        shading="auto",
        cmap="coolwarm",
        norm=_symlog_norm(total_number_flux_density),
    )
    axes[0].set_title(f"Number flux at alt={altitude_m:.1f} m")
    axes[0].set_xlabel("Longitude [deg]")
    axes[0].set_ylabel("Latitude [deg]")
    fig.colorbar(mesh, ax=axes[0], label="Number flux [1/m^2/s]")

    mesh = axes[1].pcolormesh(
        longitude_deg,
        latitude_deg,
        total_mass_flux_density.T,
        shading="auto",
        cmap="coolwarm",
        norm=_symlog_norm(total_mass_flux_density),
    )
    axes[1].set_title(f"Mass flux at alt={altitude_m:.1f} m")
    axes[1].set_xlabel("Longitude [deg]")
    axes[1].set_ylabel("Latitude [deg]")
    fig.colorbar(mesh, ax=axes[1], label="Mass flux [kg/m^2/s]")

    fig.suptitle(f"{title_prefix}  shell flux map")
    fig.savefig(png_path)
    plt.close(fig)


def process_bin_file(file_path: str | Path) -> None:
    """Process one UA/MGITM `.bin` file into quicklook slice PNGs."""
    path = Path(file_path)
    output_dir = path.parent / "ua"
    prefix = output_prefix_from_input_file(path.name)

    stage_start = perf_counter()
    log.info("%s", path.name)
    log.info("Loading UA/MGITM dataset...")
    raw = read_ua_gitm_bin(path)
    smart_ds = SmartDs(raw)
    smart_ds.merge_computation_graph(build_ua_graph(raw.variables))
    output_dir.mkdir(parents=True, exist_ok=True)
    log.debug("Loading UA/MGITM dataset complete in %.2f s.", perf_counter() - stage_start)

    longitude_deg = np.asarray(smart_ds["Longitude [deg]"], dtype=float)
    latitude_deg = np.asarray(smart_ds["Latitude [deg]"], dtype=float)
    altitude_m = np.asarray(smart_ds["Altitude [m]"], dtype=float)

    lon_axis = longitude_deg[:, 0, 0]
    lat_axis = latitude_deg[0, :, 0]
    alt_axis = altitude_m[0, 0, :]
    physical_lon_mask = _physical_longitude_mask(lon_axis)
    physical_lat_mask = _physical_latitude_mask(lat_axis)
    physical_alt_mask = _physical_altitude_mask(alt_axis)
    lon_axis_physical = lon_axis[physical_lon_mask]
    lat_axis_physical = lat_axis[physical_lat_mask]
    alt_axis_physical = alt_axis[physical_alt_mask]
    i_lon = int(len(lon_axis_physical) // 2)
    i_lon_opposite = _opposite_longitude_index(lon_axis_physical, i_lon)
    i_alt = int(len(alt_axis_physical) // 2)

    time = smart_ds.raw.aux["UA_TIME"]
    shared_title = f"{path.name}  {time.isoformat(sep=' ')}"

    stage_start = perf_counter()
    log.info("Computing UA shell fluxes...")
    shell_radius_m = UA_EARTH_RBODY_M + alt_axis_physical
    shell_area_m2 = _shell_area_grid_m2(lon_axis_physical, lat_axis_physical, shell_radius_m)
    (
        shell_species_names,
        shell_number_flux,
        shell_mass_flux,
        shell_number_flux_density,
        shell_mass_flux_density,
    ) = _available_shell_flux_series(
        smart_ds,
        shell_area_m2,
        physical_lon_mask,
        physical_lat_mask,
        physical_alt_mask,
    )
    total_number_flux = np.sum(shell_number_flux, axis=0)
    total_mass_flux = np.sum(shell_mass_flux, axis=0)
    total_number_flux_density = np.sum(shell_number_flux_density, axis=0)
    total_mass_flux_density = np.sum(shell_mass_flux_density, axis=0)

    flux_npz_path = output_dir / f"{prefix}.ua.shell_flux.npz"
    np.savez_compressed(
        flux_npz_path,
        altitude_m=alt_axis_physical,
        radius_m=shell_radius_m,
        species_names=shell_species_names,
        species_number_flux_1_s=shell_number_flux,
        species_mass_flux_kg_s=shell_mass_flux,
        species_number_flux_density_1_m2_s=shell_number_flux_density,
        species_mass_flux_density_kg_m2_s=shell_mass_flux_density,
        total_number_flux_1_s=total_number_flux,
        total_mass_flux_kg_s=total_mass_flux,
        total_number_flux_density_1_m2_s=total_number_flux_density,
        total_mass_flux_density_kg_m2_s=total_mass_flux_density,
    )
    add_record("ua_shell_flux_npz %r", str(flux_npz_path.relative_to(path.parent)))

    number_flux_png = output_dir / f"{prefix}.ua.shell_number_flux.png"
    _plot_shell_flux_profile(
        alt_axis_physical,
        shell_species_names,
        shell_number_flux,
        total_number_flux,
        ylabel="Number flux [1/s]",
        title=f"{shared_title}  shell number flux",
        png_path=number_flux_png,
    )
    add_record("ua_shell_number_flux_png %r", str(number_flux_png.relative_to(path.parent)))

    mass_flux_png = output_dir / f"{prefix}.ua.shell_mass_flux.png"
    _plot_shell_flux_profile(
        alt_axis_physical,
        shell_species_names,
        shell_mass_flux,
        total_mass_flux,
        ylabel="Mass flux [kg/s]",
        title=f"{shared_title}  shell mass flux",
        png_path=mass_flux_png,
    )
    add_record("ua_shell_mass_flux_png %r", str(mass_flux_png.relative_to(path.parent)))

    shell_map_png = output_dir / f"{prefix}.ua.shell_flux_map.png"
    _plot_shell_flux_map(
        lon_axis_physical,
        lat_axis_physical,
        total_number_flux_density[:, :, i_alt],
        total_mass_flux_density[:, :, i_alt],
        altitude_m=alt_axis_physical[i_alt],
        title_prefix=shared_title,
        png_path=shell_map_png,
    )
    add_record("ua_shell_flux_map_png %r", str(shell_map_png.relative_to(path.parent)))
    log.debug("Computing UA shell fluxes complete in %.2f s.", perf_counter() - stage_start)

    stage_start = perf_counter()
    log.info("Plotting latitude-altitude UA slices...")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), constrained_layout=True)
    lat_alt_temperature = np.asarray(smart_ds["Tn [K]"], dtype=float)[physical_lon_mask, :, :][
        i_lon, physical_lat_mask, :
    ][:, physical_alt_mask]
    lat_alt_rho = _positive_for_log(
        np.asarray(smart_ds["neutral_number_density [1/m^3]"], dtype=float)[physical_lon_mask, :, :][
            i_lon, physical_lat_mask, :
        ][:, physical_alt_mask]
    )

    mesh = axes[0].pcolormesh(lat_axis_physical, alt_axis_physical, lat_alt_temperature.T, shading="auto")
    axes[0].set_title(f"Temperature at lon={lon_axis_physical[i_lon]:.1f} deg")
    axes[0].set_xlabel("Latitude [deg]")
    axes[0].set_ylabel("Altitude [m]")
    fig.colorbar(mesh, ax=axes[0], label="Temperature [K]")

    mesh = axes[1].pcolormesh(
        lat_axis_physical, alt_axis_physical, lat_alt_rho.T, shading="auto", norm=LogNorm()
    )
    axes[1].set_title(f"neutral density at lon={lon_axis_physical[i_lon]:.1f} deg")
    axes[1].set_xlabel("Latitude [deg]")
    axes[1].set_ylabel("Altitude [m]")
    fig.colorbar(mesh, ax=axes[1], label="neutral number density [1/m^3]")

    fig.suptitle(shared_title)
    out_path = output_dir / f"{prefix}.ua.lat_alt.png"
    fig.savefig(out_path)
    plt.close(fig)
    add_record("ua_lat_alt_png %r", str(out_path.relative_to(path.parent)))
    log.debug("Plotting latitude-altitude UA slices complete in %.2f s.", perf_counter() - stage_start)

    stage_start = perf_counter()
    log.info("Plotting longitude-latitude UA slices...")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), constrained_layout=True)
    lon_lat_electron_temperature = np.asarray(smart_ds["Te [K]"], dtype=float)[physical_lon_mask, :, :][
        :, physical_lat_mask, :
    ][:, :, physical_alt_mask][:, :, i_alt]
    lon_lat_sza = np.asarray(smart_ds["Solar Zenith Angle [rad]"], dtype=float)[physical_lon_mask, :, :][
        :, physical_lat_mask, :
    ][:, :, physical_alt_mask][:, :, i_alt]

    mesh = axes[0].pcolormesh(lon_axis_physical, lat_axis_physical, lon_lat_electron_temperature.T, shading="auto")
    axes[0].set_title(f"eTemperature at alt={alt_axis_physical[i_alt]:.1f} m")
    axes[0].set_xlabel("Longitude [deg]")
    axes[0].set_ylabel("Latitude [deg]")
    fig.colorbar(mesh, ax=axes[0], label="eTemperature [K]")

    mesh = axes[1].pcolormesh(lon_axis_physical, lat_axis_physical, lon_lat_sza.T, shading="auto")
    axes[1].set_title(f"Solar Zenith Angle at alt={alt_axis_physical[i_alt]:.1f} m")
    axes[1].set_xlabel("Longitude [deg]")
    axes[1].set_ylabel("Latitude [deg]")
    fig.colorbar(mesh, ax=axes[1], label="Solar Zenith Angle [rad]")

    fig.suptitle(shared_title)
    out_path = output_dir / f"{prefix}.ua.lon_lat.png"
    fig.savefig(out_path)
    plt.close(fig)
    add_record("ua_lon_lat_png %r", str(out_path.relative_to(path.parent)))
    log.debug("Plotting longitude-latitude UA slices complete in %.2f s.", perf_counter() - stage_start)

    stage_start = perf_counter()
    log.info("Plotting exaggerated-shell UA slices...")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2), constrained_layout=True)
    theta_edges = np.deg2rad(_cell_edges_1d(lat_axis_physical))
    alt_edges = _cell_edges_1d(alt_axis_physical)
    alt_span = float(alt_axis_physical[-1] - alt_axis_physical[0])
    if alt_span <= 0.0:
        raise ValueError(f"Expected positive altitude span, got {alt_span}")
    r_edges = 1.0 + UA_EXAGGERATED_SHELL_THICKNESS * (alt_edges - alt_axis_physical[0]) / alt_span
    theta_grid, r_grid = np.meshgrid(theta_edges, r_edges, indexing="ij")
    x_grid_front = r_grid * np.cos(theta_grid)
    x_grid_back = -r_grid * np.cos(theta_grid)
    y_grid = r_grid * np.sin(theta_grid)
    surface_theta = np.linspace(theta_edges[0], theta_edges[-1], 512)

    shell_temperature_front = lat_alt_temperature
    shell_temperature_back = np.asarray(smart_ds["Tn [K]"], dtype=float)[physical_lon_mask, :, :][
        i_lon_opposite, physical_lat_mask, :
    ][:, physical_alt_mask]
    shell_rho_front = lat_alt_rho
    shell_rho_back = _positive_for_log(
        np.asarray(smart_ds["neutral_number_density [1/m^3]"], dtype=float)[physical_lon_mask, :, :][
            i_lon_opposite, physical_lat_mask, :
        ][:, physical_alt_mask]
    )

    temp_min = min(float(np.nanmin(shell_temperature_front)), float(np.nanmin(shell_temperature_back)))
    temp_max = max(float(np.nanmax(shell_temperature_front)), float(np.nanmax(shell_temperature_back)))
    mesh = axes[0].pcolormesh(
        x_grid_front,
        y_grid,
        shell_temperature_front,
        shading="auto",
        vmin=temp_min,
        vmax=temp_max,
    )
    axes[0].pcolormesh(
        x_grid_back,
        y_grid,
        shell_temperature_back,
        shading="auto",
        vmin=temp_min,
        vmax=temp_max,
    )
    axes[0].plot(np.cos(surface_theta), np.sin(surface_theta), color="k", linewidth=1.0)
    axes[0].plot(-np.cos(surface_theta), np.sin(surface_theta), color="k", linewidth=1.0)
    axes[0].set_title(
        f"Temperature shell at lon={lon_axis_physical[i_lon]:.1f}/{lon_axis_physical[i_lon_opposite]:.1f} deg"
    )
    axes[0].set_xlabel("x [shell units]")
    axes[0].set_ylabel("y [shell units]")
    axes[0].set_aspect("equal")
    fig.colorbar(mesh, ax=axes[0], label="Temperature [K]")

    rho_min = min(float(np.nanmin(shell_rho_front)), float(np.nanmin(shell_rho_back)))
    rho_max = max(float(np.nanmax(shell_rho_front)), float(np.nanmax(shell_rho_back)))
    mesh = axes[1].pcolormesh(
        x_grid_front,
        y_grid,
        shell_rho_front,
        shading="auto",
        norm=LogNorm(vmin=rho_min, vmax=rho_max),
    )
    axes[1].pcolormesh(
        x_grid_back,
        y_grid,
        shell_rho_back,
        shading="auto",
        norm=LogNorm(vmin=rho_min, vmax=rho_max),
    )
    axes[1].plot(np.cos(surface_theta), np.sin(surface_theta), color="k", linewidth=1.0)
    axes[1].plot(-np.cos(surface_theta), np.sin(surface_theta), color="k", linewidth=1.0)
    axes[1].set_title(
        f"neutral density shell at lon={lon_axis_physical[i_lon]:.1f}/{lon_axis_physical[i_lon_opposite]:.1f} deg"
    )
    axes[1].set_xlabel("x [shell units]")
    axes[1].set_ylabel("y [shell units]")
    axes[1].set_aspect("equal")
    fig.colorbar(mesh, ax=axes[1], label="neutral number density [1/m^3]")

    fig.suptitle(f"{shared_title}  exaggerated shell (opposite meridians)")
    out_path = output_dir / f"{prefix}.ua.shell.png"
    fig.savefig(out_path)
    plt.close(fig)
    add_record("ua_shell_png %r", str(out_path.relative_to(path.parent)))
    log.debug("Plotting exaggerated-shell UA slices complete in %.2f s.", perf_counter() - stage_start)


__all__ = ["process_bin_file"]
