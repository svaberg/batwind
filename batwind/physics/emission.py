from __future__ import annotations

from pathlib import Path

import numpy as np
from batcamp import Octree
from batcamp import OctreeInterpolator

from batwind.smart_ds import SmartDs

DEFAULT_RESPONSE_FUNCTION_PATH = Path("/Users/dagfev/Documents/starwinds/g_lambda_T/TestResposne.dat")
# The response table stores contribution-function values in
# 10^-26 erg cm^3 s^-1 sr^-1. Convert that once to SI:
# 10^-26 * (1e-7 J / erg) * (1e-6 m^3 / cm^3) = 1e-39 W m^3 sr^-1.
RESPONSE_TABLE_SCALE_TO_SI = 1.0e-39
# The spectral-contribution `.npy` cubes store ``G_lambda(T)`` in
# ``erg cm^3 s^-1 sr^-1 A^-1``. Convert that once to SI:
# (1e-7 W s / erg) * (1e-6 m^3 / cm^3) = 1e-13 W m^3 sr^-1 A^-1.
SPECTRAL_CONTRIBUTION_SCALE_TO_SI = 1.0e-13


def load_response_table(
    response_path: Path = DEFAULT_RESPONSE_FUNCTION_PATH,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Load one band-integrated response-function table.

    Returns:
    - ``log10_temperature`` on the table grid
    - one mapping of component name to SI values in ``W m^3 sr^-1``
    """
    with response_path.open("r", encoding="utf-8") as stream:
        next(stream)
        next(stream)
        shape = tuple(int(value) for value in next(stream).split())[::-1]
        names = next(stream).split()
        data = np.loadtxt(stream)
    if tuple(names[:2]) != ("l10T", "l10ne"):
        raise ValueError(f"Unexpected response-table columns in {response_path!s}: {names[:2]!r}")
    if shape[0] != 1:
        raise ValueError(f"Expected a single-density response table in {response_path!s}, got shape {shape!r}")
    log10_temperature = np.asarray(data[:, 0].reshape(shape)[0], dtype=float)
    components = {
        name: RESPONSE_TABLE_SCALE_TO_SI * np.asarray(data[:, col_id].reshape(shape)[0], dtype=float)
        for col_id, name in enumerate(names[2:], start=2)
    }
    return log10_temperature, components


def load_spectral_contribution_table(
    spectrum_path: Path,
    *,
    density_path: Path,
    temperature_path: Path,
    wavelength_path: Path,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one precomputed ``G_lambda(T)`` table from the spectral-contribution
    `.npy` bundle.

    Returns:
    - one electron-density value in ``m^-3``
    - the temperature grid in ``K``
    - the wavelength grid in ``A``
    - the spectral contribution cube in ``W m^3 sr^-1 A^-1``
    """
    density_grid_cm3 = np.asarray(np.load(density_path), dtype=float)
    temperature_grid_k = np.asarray(np.load(temperature_path), dtype=float)
    wavelength_grid_angstrom = np.asarray(np.load(wavelength_path), dtype=float)
    spectral_contribution_table = np.asarray(np.load(spectrum_path, allow_pickle=True), dtype=float)

    expected_grid_shape = density_grid_cm3.shape
    if temperature_grid_k.shape != expected_grid_shape or wavelength_grid_angstrom.shape != expected_grid_shape:
        raise ValueError(
            "Spectral-contribution grids must share one shape, got "
            f"{density_grid_cm3.shape}, {temperature_grid_k.shape}, {wavelength_grid_angstrom.shape}"
        )
    if spectral_contribution_table.shape[:3] != expected_grid_shape:
        raise ValueError(
            "Spectral-contribution cube must match the grid shape, got "
            f"{spectral_contribution_table.shape[:3]} and {expected_grid_shape}"
        )
    if expected_grid_shape[0] != 1:
        raise ValueError(f"Expected a single-density spectral table, got shape {expected_grid_shape}")
    if spectral_contribution_table.shape[-1] != 5:
        raise ValueError(f"Expected 5 spectral components, got shape {spectral_contribution_table.shape}")

    density_plane_cm3 = density_grid_cm3[0]
    temperature_plane_k = temperature_grid_k[0]
    wavelength_plane_angstrom = wavelength_grid_angstrom[0]
    if not np.allclose(density_plane_cm3, density_plane_cm3[0, 0]):
        raise ValueError("Expected one constant density across the spectral-contribution table")
    if not np.allclose(temperature_plane_k, temperature_plane_k[:, :1]):
        raise ValueError("Expected the spectral-contribution temperature grid to vary only along the temperature axis")
    if not np.allclose(wavelength_plane_angstrom, wavelength_plane_angstrom[:1, :]):
        raise ValueError("Expected the spectral-contribution wavelength grid to vary only along the wavelength axis")

    return (
        float(density_plane_cm3[0, 0]) * 1.0e6,
        np.asarray(temperature_plane_k[:, 0], dtype=float),
        np.asarray(wavelength_plane_angstrom[0], dtype=float),
        SPECTRAL_CONTRIBUTION_SCALE_TO_SI * np.asarray(spectral_contribution_table[0], dtype=float),
    )


def band_response_values_from_components_si(
    response_components: dict[str, np.ndarray],
    component_names: tuple[str, ...],
) -> np.ndarray:
    """
    Return one band-integrated contribution function on the table grid in SI.

    The returned units are ``W m^3 sr^-1``.
    """
    response_values_si = np.zeros_like(np.asarray(next(iter(response_components.values()))), dtype=float)
    for component_name in component_names:
        try:
            response_component = response_components[component_name]
        except KeyError as exc:
            raise ValueError(f"Missing response-table component {component_name!r}") from exc
        response_values_si = response_values_si + np.asarray(response_component, dtype=float)
    return response_values_si


def band_response_values_from_spectral_contribution_si(
    temperature_grid_k: np.ndarray,
    wavelength_grid_angstrom: np.ndarray,
    spectral_contribution_values_si: np.ndarray,
    wavelength_limits_angstrom: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrate one precomputed ``G_lambda(T)`` cube over one wavelength band.

    Units:
    - input spectral contribution: ``W m^3 sr^-1 A^-1``
    - output band contribution: ``W m^3 sr^-1``
    """
    temperature_grid_k = np.asarray(temperature_grid_k, dtype=float)
    wavelength_grid_angstrom = np.asarray(wavelength_grid_angstrom, dtype=float)
    spectral_contribution_values_si = np.asarray(spectral_contribution_values_si, dtype=float)
    if spectral_contribution_values_si.shape != (temperature_grid_k.size, wavelength_grid_angstrom.size, 5):
        raise ValueError(
            "Expected one spectral-contribution cube with shape "
            f"({temperature_grid_k.size}, {wavelength_grid_angstrom.size}, 5), got {spectral_contribution_values_si.shape}"
        )

    wavelength_min_angstrom, wavelength_max_angstrom = wavelength_limits_angstrom
    wavelength_mask = (
        (wavelength_grid_angstrom >= wavelength_min_angstrom)
        & (wavelength_grid_angstrom < wavelength_max_angstrom)
    )
    if not np.any(wavelength_mask):
        raise ValueError(f"No wavelengths fall inside the band limits {wavelength_limits_angstrom}")

    total_spectral_contribution_si = np.sum(spectral_contribution_values_si[..., 1:], axis=-1)
    band_response_values_si = np.trapezoid(
        total_spectral_contribution_si[:, wavelength_mask],
        wavelength_grid_angstrom[wavelength_mask],
        axis=-1,
    )
    return np.log10(temperature_grid_k), np.asarray(band_response_values_si, dtype=float)


def interpolate_band_contribution_function_si(
    temperature_k: np.ndarray,
    response_log10_temperature: np.ndarray,
    band_response_values_si: np.ndarray,
) -> np.ndarray:
    """
    Interpolate one band-integrated contribution function onto one temperature field.

    Units:
    - input response values: ``W m^3 sr^-1``
    - output values: ``W m^3 sr^-1``
    """
    temperature_k = np.asarray(temperature_k, dtype=float)
    response_log10_temperature = np.asarray(response_log10_temperature, dtype=float)
    band_response_values_si = np.asarray(band_response_values_si, dtype=float)
    target_log10_temperature = np.log10(np.clip(temperature_k, 10 ** response_log10_temperature[0], None))
    return np.interp(
        target_log10_temperature,
        response_log10_temperature,
        band_response_values_si,
        left=band_response_values_si[0],
        right=band_response_values_si[-1],
    )


def band_emissivity_si(
    smart_ds: SmartDs,
    response_log10_temperature: np.ndarray,
    band_response_values_si: np.ndarray,
) -> np.ndarray:
    """
    Return one band emissivity field in SI units.

    Units:
    - contribution function ``G(T)``: ``W m^3 sr^-1``
    - electron density ``n_e``: ``m^-3``
    - emissivity ``epsilon = G(T) n_e^2``: ``W m^-3 sr^-1``
    """
    contribution_function_si = interpolate_band_contribution_function_si(
        np.asarray(smart_ds["te [K]"], dtype=float),
        response_log10_temperature,
        band_response_values_si,
    )
    electron_density_m3 = np.asarray(smart_ds["Ne [1/m^3]"], dtype=float)
    return contribution_function_si * electron_density_m3**2 * np.asarray(
        smart_ds["transition_region_emission_weight [none]"],
        dtype=float,
    )


def band_emissivity_from_response_table_si(
    smart_ds: SmartDs,
    component_names: tuple[str, ...],
    *,
    response_path: Path = DEFAULT_RESPONSE_FUNCTION_PATH,
) -> np.ndarray:
    """
    Return one band emissivity field from the response table in SI units.
    """
    response_log10_temperature, response_components = load_response_table(response_path)
    response_values_si = band_response_values_from_components_si(response_components, component_names)
    return band_emissivity_si(smart_ds, response_log10_temperature, response_values_si)


def band_emissivity_from_spectral_contribution_si(
    smart_ds: SmartDs,
    wavelength_limits_angstrom: tuple[float, float],
    *,
    spectrum_path: Path,
    density_path: Path,
    temperature_path: Path,
    wavelength_path: Path,
) -> np.ndarray:
    """
    Return one band emissivity field from one precomputed spectral-contribution
    `.npy` cube in SI units.
    """
    _, temperature_grid_k, wavelength_grid_angstrom, spectral_contribution_values_si = load_spectral_contribution_table(
        spectrum_path,
        density_path=density_path,
        temperature_path=temperature_path,
        wavelength_path=wavelength_path,
    )
    response_log10_temperature, response_values_si = band_response_values_from_spectral_contribution_si(
        temperature_grid_k,
        wavelength_grid_angstrom,
        spectral_contribution_values_si,
        wavelength_limits_angstrom,
    )
    return band_emissivity_si(smart_ds, response_log10_temperature, response_values_si)


def unblocked_solid_angle(radial_distance_r: np.ndarray) -> np.ndarray:
    """
    Return the unblocked solid angle outside one opaque stellar sphere.

    Units:
    - input radius: ``R_*``
    - output solid angle: ``sr``
    """
    radial_distance_r = np.asarray(radial_distance_r, dtype=float)
    return 2.0 * np.pi * (1.0 + np.sqrt(np.clip(1.0 - radial_distance_r**-2, 0.0, None)))


def point_unblocked_solid_angle_sr(smart_ds: SmartDs) -> np.ndarray:
    """
    Return the exterior unblocked solid angle in steradians at every dataset point.
    """
    return np.asarray(smart_ds["unblocked_solid_angle [sr]"], dtype=float)


def band_luminosity_si(
    smart_ds: SmartDs,
    point_emissivity_w_m3_sr: np.ndarray,
    *,
    occultation: bool = True,
    tree: Octree | None = None,
) -> float:
    """
    Return one band luminosity in SI units.

    This implements the section-3.4 quantity
    ``L = \\int_V \\omega(r) epsilon dV``.

    Units:
    - emissivity ``epsilon``: ``W m^-3 sr^-1``
    - solid-angle factor ``omega``: ``sr``
    - volume ``dV``: ``m^3``
    - luminosity ``L``: ``W``

    Implementation note:
    - the self-occultation factor ``omega(r)`` is evaluated at every dataset
      point and folded into a point-valued luminosity-density field
    - the volume integral of that weighted point field is then delegated to
      ``batcamp`` via exact whole-cell trilinear integrals
    """
    point_emissivity_w_m3_sr = np.asarray(point_emissivity_w_m3_sr, dtype=float)
    if tree is None:
        tree = Octree.from_ds(smart_ds.raw)
    body_radius_m = float(smart_ds["RBODY [m]"])
    leaf_count = int(np.asarray(tree.corners).shape[0])
    leaf_ids = np.arange(leaf_count, dtype=int)
    if occultation:
        solid_angle_sr = point_unblocked_solid_angle_sr(smart_ds)
    else:
        solid_angle_sr = np.full(point_emissivity_w_m3_sr.shape, 4.0 * np.pi, dtype=float)
    point_luminosity_density_w_m3 = point_emissivity_w_m3_sr * solid_angle_sr
    luminosity_integral_w = (
        np.asarray(OctreeInterpolator(tree, point_luminosity_density_w_m3).cell_integrals(leaf_ids), dtype=float)
        * body_radius_m**3
    )
    return float(np.sum(luminosity_integral_w))
