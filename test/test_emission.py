import numpy as np
from pathlib import Path

from batcamp import Octree
from batcamp import OctreeInterpolator
from batread import Dataset
from netCDF4 import Dataset as NetcdfDataset

from batwind.physics.emission import band_emissivity_si
from batwind.physics.emission import band_emissivity_from_spectral_contribution_si
from batwind.physics.emission import band_luminosity_si
from batwind.physics.emission import load_spectral_contribution_table
from batwind.physics.emission import unblocked_solid_angle
from batwind.recipes.batsrus import build_batsrus_graph
from batwind.smart_ds import SmartDs

ROSAT_WAVELENGTH_LIMITS_ANGSTROM = (5.0, 120.0)
SYNTHETIC_DENSITY_CM3 = 1.0e10
SYNTHETIC_TEMPERATURE_K = np.array([1.0e5, 1.0e6, 1.0e7], dtype=float)
SYNTHETIC_WAVELENGTH_ANGSTROM = np.array([1.0, 10.0, 50.0, 150.0], dtype=float)
SYNTHETIC_ROSAT_RESPONSE_SI = np.array([1.0e-30, 2.0e-30, 3.0e-30], dtype=float)


def make_one_cell_dataset(origin_r: tuple[float, float, float], width_r: float, *, variables: list[str]) -> Dataset:
    x0, y0, z0 = origin_r
    x1 = x0 + float(width_r)
    y1 = y0 + float(width_r)
    z1 = z0 + float(width_r)
    corners_xyz = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=float,
    )
    points = np.zeros((8, len(variables)), dtype=float)
    points[:, :3] = corners_xyz
    corners = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=int)
    return Dataset(points, corners, aux={}, title="one-cell", variables=variables, zone="z0")


def write_synthetic_spectral_contribution_file(tmp_path: Path) -> Path:
    spectral_contribution_si = np.zeros(
        (SYNTHETIC_TEMPERATURE_K.size, SYNTHETIC_WAVELENGTH_ANGSTROM.size),
        dtype=float,
    )
    rosat_mask = (SYNTHETIC_WAVELENGTH_ANGSTROM >= ROSAT_WAVELENGTH_LIMITS_ANGSTROM[0]) & (
        SYNTHETIC_WAVELENGTH_ANGSTROM < ROSAT_WAVELENGTH_LIMITS_ANGSTROM[1]
    )
    spectral_density_si = SYNTHETIC_ROSAT_RESPONSE_SI[:, np.newaxis] / 40.0
    spectral_contribution_si[:, rosat_mask] = spectral_density_si

    spectrum_path = tmp_path / "synthetic-spectrum.nc"
    with NetcdfDataset(spectrum_path, "w") as dataset:
        dataset.createDimension("density", 1)
        dataset.createDimension("temperature", SYNTHETIC_TEMPERATURE_K.size)
        dataset.createDimension("wavelength", SYNTHETIC_WAVELENGTH_ANGSTROM.size)
        dataset.createVariable("density", "f8", ("density",))[:] = [SYNTHETIC_DENSITY_CM3]
        dataset.createVariable("temperature", "f8", ("temperature",))[:] = SYNTHETIC_TEMPERATURE_K
        dataset.createVariable("wavelength", "f8", ("wavelength",))[:] = SYNTHETIC_WAVELENGTH_ANGSTROM
        for component_name, fraction in (("freefree", 0.25), ("freebound", 0.0), ("line", 0.75), ("twophoton", 0.0)):
            dataset.createVariable(component_name, "f8", ("density", "temperature", "wavelength"))[:] = (
                fraction * spectral_contribution_si[np.newaxis] / 1.0e-13
            )
    return spectrum_path


def test_band_emissivity_si_tracks_si_units():
    dataset = make_one_cell_dataset((1.2, 0.0, 0.0), 0.4, variables=["X [R]", "Y [R]", "Z [R]", "te [K]", "Rho [kg/m^3]"])
    dataset.points[:, 3] = 1.0e5
    dataset.points[:, 4] = 2.0 * 1.67262192595e-27
    sds = SmartDs(dataset)
    sds.merge_computation_graph(build_batsrus_graph(tuple(dataset.variables), body_radius_m=1.0))

    response_log10_temperature = np.array([4.0, 6.0], dtype=float)
    band_response_values_si = np.array([2.0, 2.0], dtype=float)
    emissivity = band_emissivity_si(sds, response_log10_temperature, band_response_values_si)

    np.testing.assert_allclose(emissivity, 8.0)


def test_band_emissivity_si_applies_transition_region_weight():
    dataset = make_one_cell_dataset((1.2, 0.0, 0.0), 0.4, variables=["X [R]", "Y [R]", "Z [R]", "te [K]", "Rho [kg/m^3]"])
    dataset.points[:, 3] = 1.0e5
    dataset.points[:, 4] = 2.0 * 1.67262192595e-27
    dataset.aux = {
        "DoExtendTransitionRegion": True,
        "TeTransitionRegionSi": 2.2e5,
        "DeltaTeModSi": 1.0e1,
    }
    sds = SmartDs(dataset)
    sds.merge_computation_graph(build_batsrus_graph(tuple(dataset.variables), body_radius_m=1.0))

    response_log10_temperature = np.array([4.0, 6.0], dtype=float)
    band_response_values_si = np.array([2.0, 2.0], dtype=float)
    emissivity = band_emissivity_si(sds, response_log10_temperature, band_response_values_si)
    expected_weight = float(sds["transition_region_emission_weight [none]"][0])

    np.testing.assert_allclose(emissivity, 8.0 * expected_weight)
    np.testing.assert_allclose(sds["transition_region_emission_weight [none]"], np.full(8, expected_weight))


def test_band_luminosity_si_matches_off_star_single_cell_formula():
    dataset = make_one_cell_dataset((1.0, 1.0, 1.0), 1.0, variables=["X [R]", "Y [R]", "Z [R]", "R [R]"])
    dataset.points[:, 3] = np.sqrt(np.sum(dataset.points[:, :3] ** 2, axis=1))
    sds = SmartDs(dataset)
    sds.merge_computation_graph(build_batsrus_graph(tuple(dataset.variables), body_radius_m=2.0))
    point_emissivity = np.full(8, 3.0, dtype=float)

    expected_volume_m3 = (1.0 * 2.0) ** 3
    tree = Octree.from_ds(dataset)

    luminosity_unocculted = band_luminosity_si(sds, point_emissivity, occultation=False)
    luminosity_occulted = band_luminosity_si(sds, point_emissivity, occultation=True)

    np.testing.assert_allclose(luminosity_unocculted, 3.0 * 4.0 * np.pi * expected_volume_m3)
    point_radius_r = np.sqrt(
        np.asarray(sds["X [R]"], dtype=float) ** 2
        + np.asarray(sds["Y [R]"], dtype=float) ** 2
        + np.asarray(sds["Z [R]"], dtype=float) ** 2
    )
    point_luminosity_density = point_emissivity * unblocked_solid_angle(point_radius_r)
    expected_occulted = float(
        np.asarray(OctreeInterpolator(tree, point_luminosity_density).cell_integrals(np.array([0], dtype=int)), dtype=float)[0]
        * float(sds["RBODY [m]"]) ** 3
    )
    np.testing.assert_allclose(luminosity_occulted, expected_occulted)


def test_load_spectral_contribution_table_reads_netcdf(tmp_path: Path):
    spectrum_path = write_synthetic_spectral_contribution_file(tmp_path)
    temperature_k, wavelength_angstrom, spectral_contribution_si = load_spectral_contribution_table(spectrum_path)

    np.testing.assert_allclose(temperature_k, SYNTHETIC_TEMPERATURE_K)
    np.testing.assert_allclose(wavelength_angstrom, SYNTHETIC_WAVELENGTH_ANGSTROM)
    assert spectral_contribution_si.shape == (SYNTHETIC_TEMPERATURE_K.size, SYNTHETIC_WAVELENGTH_ANGSTROM.size)


def test_spectral_contribution_emissivity_matches_direct_si_emissivity(tmp_path: Path):
    dataset = make_one_cell_dataset((1.2, 0.0, 0.0), 0.4, variables=["X [R]", "Y [R]", "Z [R]", "te [K]", "Rho [kg/m^3]"])
    dataset.points[:, 3] = 1.0e6
    dataset.points[:, 4] = 2.0 * 1.67262192595e-27
    sds = SmartDs(dataset)
    sds.merge_computation_graph(build_batsrus_graph(tuple(dataset.variables), body_radius_m=1.0))
    spectrum_path = write_synthetic_spectral_contribution_file(tmp_path)

    from_spectral_contribution = band_emissivity_from_spectral_contribution_si(
        sds,
        ROSAT_WAVELENGTH_LIMITS_ANGSTROM,
        spectrum_path=spectrum_path,
    )
    direct_si = band_emissivity_si(sds, np.log10(SYNTHETIC_TEMPERATURE_K), SYNTHETIC_ROSAT_RESPONSE_SI)

    np.testing.assert_allclose(from_spectral_contribution, direct_si)
