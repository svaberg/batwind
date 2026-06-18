import numpy as np
from pathlib import Path

from batcamp import Octree
from batcamp import OctreeInterpolator
from batread import Dataset

from batwind.physics.emission import band_emissivity_si
from batwind.physics.emission import band_emissivity_from_spectral_contribution_si
from batwind.physics.emission import band_emissivity_from_response_table_si
from batwind.physics.emission import band_luminosity_si
from batwind.physics.emission import load_spectral_contribution_table
from batwind.physics.emission import RESPONSE_TABLE_SCALE_TO_SI
from batwind.physics.emission import unblocked_solid_angle
from batwind.recipes.batsrus import build_batsrus_graph
from batwind.smart_ds import SmartDs

ROSAT_COMPONENTS = ("ROSAT_line", "ROSAT_cont")
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


def write_synthetic_response_table(response_path: Path) -> None:
    rosat_response_table_values = SYNTHETIC_ROSAT_RESPONSE_SI / RESPONSE_TABLE_SCALE_TO_SI
    lines = [
        "Synthetic response table\n",
        "0 0.000000 2 1 6\n",
        f"{SYNTHETIC_TEMPERATURE_K.size} 1\n",
        "l10T l10ne Hard_line Hard_cont ROSAT_line ROSAT_cont EUV_line EUV_cont\n",
    ]
    for temperature_k, rosat_response_table_value in zip(SYNTHETIC_TEMPERATURE_K, rosat_response_table_values):
        lines.append(
            f"{np.log10(temperature_k):.6e} {np.log10(SYNTHETIC_DENSITY_CM3):.6e} "
            f"0.000000e+00 0.000000e+00 "
            f"{0.25 * rosat_response_table_value:.6e} {0.75 * rosat_response_table_value:.6e} "
            f"0.000000e+00 0.000000e+00\n"
        )
    response_path.write_text("".join(lines), encoding="utf-8")


def write_synthetic_spectral_contribution_bundle(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    density_grid = np.full(
        (1, SYNTHETIC_TEMPERATURE_K.size, SYNTHETIC_WAVELENGTH_ANGSTROM.size),
        SYNTHETIC_DENSITY_CM3,
        dtype=float,
    )
    temperature_grid = np.broadcast_to(
        SYNTHETIC_TEMPERATURE_K[np.newaxis, :, np.newaxis],
        density_grid.shape,
    ).copy()
    wavelength_grid = np.broadcast_to(
        SYNTHETIC_WAVELENGTH_ANGSTROM[np.newaxis, np.newaxis, :],
        density_grid.shape,
    ).copy()
    spectral_contribution_si = np.zeros(density_grid.shape + (5,), dtype=float)
    rosat_mask = (SYNTHETIC_WAVELENGTH_ANGSTROM >= ROSAT_WAVELENGTH_LIMITS_ANGSTROM[0]) & (
        SYNTHETIC_WAVELENGTH_ANGSTROM < ROSAT_WAVELENGTH_LIMITS_ANGSTROM[1]
    )
    spectral_density_si = SYNTHETIC_ROSAT_RESPONSE_SI[:, np.newaxis] / 40.0
    spectral_contribution_si[0, :, :, 1][:, rosat_mask] = 0.25 * spectral_density_si
    spectral_contribution_si[0, :, :, 2][:, rosat_mask] = 0.75 * spectral_density_si
    spectral_contribution_raw = spectral_contribution_si / 1.0e-13

    density_path = tmp_path / "grid-density.npy"
    temperature_path = tmp_path / "grid-temperature.npy"
    wavelength_path = tmp_path / "grid-wavelength.npy"
    spectrum_path = tmp_path / "synthetic-spectrum.npy"
    np.save(density_path, density_grid)
    np.save(temperature_path, temperature_grid)
    np.save(wavelength_path, wavelength_grid)
    np.save(spectrum_path, spectral_contribution_raw)
    return spectrum_path, density_path, temperature_path, wavelength_path


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


def test_response_table_emissivities_match_direct_si_emissivities(tmp_path: Path):
    dataset = make_one_cell_dataset((1.2, 0.0, 0.0), 0.4, variables=["X [R]", "Y [R]", "Z [R]", "te [K]", "Rho [kg/m^3]"])
    dataset.points[:, 3] = 1.0e5
    dataset.points[:, 4] = 2.0 * 1.67262192595e-27
    sds = SmartDs(dataset)
    sds.merge_computation_graph(build_batsrus_graph(tuple(dataset.variables), body_radius_m=1.0))
    response_path = tmp_path / "synthetic-response.dat"
    write_synthetic_response_table(response_path)

    response_log10_temperature = np.log10(SYNTHETIC_TEMPERATURE_K)
    direct_si = band_emissivity_si(sds, response_log10_temperature, SYNTHETIC_ROSAT_RESPONSE_SI)
    from_response_table = band_emissivity_from_response_table_si(sds, ROSAT_COMPONENTS, response_path=response_path)

    np.testing.assert_allclose(from_response_table, direct_si)


def test_load_spectral_contribution_table_reads_one_density_bundle(tmp_path: Path):
    spectrum_path, density_path, temperature_path, wavelength_path = write_synthetic_spectral_contribution_bundle(tmp_path)
    density_m3, temperature_k, wavelength_angstrom, spectral_contribution_si = load_spectral_contribution_table(
        spectrum_path=spectrum_path,
        density_path=density_path,
        temperature_path=temperature_path,
        wavelength_path=wavelength_path,
    )

    np.testing.assert_allclose(density_m3, SYNTHETIC_DENSITY_CM3 * 1.0e6)
    np.testing.assert_allclose(temperature_k, SYNTHETIC_TEMPERATURE_K)
    np.testing.assert_allclose(wavelength_angstrom, SYNTHETIC_WAVELENGTH_ANGSTROM)
    assert spectral_contribution_si.shape == (SYNTHETIC_TEMPERATURE_K.size, SYNTHETIC_WAVELENGTH_ANGSTROM.size, 5)


def test_spectral_contribution_and_response_table_emissivities_are_consistent(tmp_path: Path):
    dataset = make_one_cell_dataset((1.2, 0.0, 0.0), 0.4, variables=["X [R]", "Y [R]", "Z [R]", "te [K]", "Rho [kg/m^3]"])
    dataset.points[:, 3] = 1.0e6
    dataset.points[:, 4] = 2.0 * 1.67262192595e-27
    sds = SmartDs(dataset)
    sds.merge_computation_graph(build_batsrus_graph(tuple(dataset.variables), body_radius_m=1.0))
    response_path = tmp_path / "synthetic-response.dat"
    write_synthetic_response_table(response_path)
    spectrum_path, density_path, temperature_path, wavelength_path = write_synthetic_spectral_contribution_bundle(tmp_path)

    from_response_table = band_emissivity_from_response_table_si(sds, ROSAT_COMPONENTS, response_path=response_path)
    from_spectral_contribution = band_emissivity_from_spectral_contribution_si(
        sds,
        ROSAT_WAVELENGTH_LIMITS_ANGSTROM,
        spectrum_path=spectrum_path,
        density_path=density_path,
        temperature_path=temperature_path,
        wavelength_path=wavelength_path,
    )

    np.testing.assert_allclose(from_spectral_contribution, from_response_table)
