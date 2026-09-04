from __future__ import annotations

from pathlib import Path

import numpy as np
from batcamp import Octree
from batcamp import OctreeInterpolator
from netCDF4 import Dataset

from batwind.smart_ds import SmartDs

DEFAULT_SPECTRAL_CONTRIBUTION_PATH = Path(
    "/Users/dagfev/Documents/starwinds/spectral-contribution/outputs/wavelength-1-250-dlambda-0.05/"
    "spectral-contribution.wavelength=1-250-dlambda=0.05."
    "AbundanceName=sun_coronal_2021_chianti-min_abundance=1.0e-07.nc"
)
SPECTRAL_COMPONENT_NAMES = ("freefree", "freebound", "line", "twophoton")
# The NetCDF spectral contribution functions store ``G_lambda(T)`` in
# ``erg cm^3 s^-1 sr^-1 A^-1``. Convert that once to SI:
# (1e-7 W s / erg) * (1e-6 m^3 / cm^3) = 1e-13 W m^3 sr^-1 A^-1.
SPECTRAL_CONTRIBUTION_SCALE_TO_SI = 1.0e-13


def load_spectral_contribution_table(
    spectrum_path: Path = DEFAULT_SPECTRAL_CONTRIBUTION_PATH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one NetCDF ``G_lambda(T)`` table from spectral-contribution.

    Returns:
    - the temperature grid in ``K``
    - the wavelength grid in ``A``
    - total spectral contribution in ``W m^3 sr^-1 A^-1``
    """
    with Dataset(spectrum_path) as dataset:
        density_cm3 = np.asarray(dataset.variables["density"][:], dtype=float)
        if density_cm3.size != 1:
            raise ValueError(f"Expected a single-density spectral table, got {density_cm3.size} densities")
        temperature_grid_k = np.asarray(dataset.variables["temperature"][:], dtype=float)
        wavelength_grid_angstrom = np.asarray(dataset.variables["wavelength"][:], dtype=float)
        components = [np.asarray(dataset.variables[name][:], dtype=float) for name in SPECTRAL_COMPONENT_NAMES]

    expected_shape = (1, temperature_grid_k.size, wavelength_grid_angstrom.size)
    if any(component.shape != expected_shape for component in components):
        raise ValueError(f"Expected spectral components with shape {expected_shape}")
    total_spectral_contribution_si = SPECTRAL_CONTRIBUTION_SCALE_TO_SI * np.sum(components, axis=0)[0]
    return temperature_grid_k, wavelength_grid_angstrom, total_spectral_contribution_si


def band_response_values_from_spectral_contribution_si(
    temperature_grid_k: np.ndarray,
    wavelength_grid_angstrom: np.ndarray,
    total_spectral_contribution_si: np.ndarray,
    wavelength_limits_angstrom: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrate one total ``G_lambda(T)`` table over one wavelength band.

    Units:
    - input spectral contribution: ``W m^3 sr^-1 A^-1``
    - output band contribution: ``W m^3 sr^-1``
    """
    temperature_grid_k = np.asarray(temperature_grid_k, dtype=float)
    wavelength_grid_angstrom = np.asarray(wavelength_grid_angstrom, dtype=float)
    total_spectral_contribution_si = np.asarray(total_spectral_contribution_si, dtype=float)
    if total_spectral_contribution_si.shape != (temperature_grid_k.size, wavelength_grid_angstrom.size):
        raise ValueError(
            "Expected total spectral contribution with shape "
            f"({temperature_grid_k.size}, {wavelength_grid_angstrom.size}), got {total_spectral_contribution_si.shape}"
        )

    wavelength_min_angstrom, wavelength_max_angstrom = wavelength_limits_angstrom
    wavelength_mask = (
        (wavelength_grid_angstrom >= wavelength_min_angstrom)
        & (wavelength_grid_angstrom <= wavelength_max_angstrom)
    )
    if not np.any(wavelength_mask):
        raise ValueError(f"No wavelengths fall inside the band limits {wavelength_limits_angstrom}")

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


def band_emissivity_from_spectral_contribution_si(
    smart_ds: SmartDs,
    wavelength_limits_angstrom: tuple[float, float],
    *,
    spectrum_path: Path = DEFAULT_SPECTRAL_CONTRIBUTION_PATH,
) -> np.ndarray:
    """
    Return one band emissivity field from one NetCDF spectral-contribution table.
    """
    temperature_grid_k, wavelength_grid_angstrom, total_spectral_contribution_si = load_spectral_contribution_table(
        spectrum_path
    )
    response_log10_temperature, response_values_si = band_response_values_from_spectral_contribution_si(
        temperature_grid_k,
        wavelength_grid_angstrom,
        total_spectral_contribution_si,
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
