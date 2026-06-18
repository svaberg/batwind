from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.io import FortranFile
from scipy.constants import atomic_mass
from scipy.constants import electron_mass

from batwind.data.ua_gitm import read_ua_gitm_bin
from batwind.pipelines.ua import process_bin_file
from batwind.recipes.ua import build_ua_graph
from batwind.smart_ds import SmartDs


def _write_ua_file(path: Path) -> None:
    n_lon, n_lat, n_alt = 3, 2, 4
    variables = [
        "Longitude",
        "Latitude",
        "Altitude",
        "Rho",
        "Temperature",
        "eTemperature",
        "[CO!D2!N             ]",
        "[O!U+!N              ]",
        "[e-                  ]",
        "V!Dn!N (east)",
        "V!Dn!N (up)",
        "V!Dn!N (up,CO!D2!N             )",
        "V!Di!N (up)",
        "Solar Zenith Angle",
    ]
    lon = np.deg2rad(np.array([-30.0, 0.0, 30.0], dtype=float))[:, None, None]
    lat = np.deg2rad(np.array([-45.0, 45.0], dtype=float))[None, :, None]
    alt = (1.0e3 * np.array([100.0, 150.0, 200.0, 250.0], dtype=float))[None, None, :]
    rho = np.arange(n_lon * n_lat * n_alt, dtype=float).reshape((n_lon, n_lat, n_alt), order="F") + 1.0
    temperature = 1000.0 + 10.0 * rho
    e_temperature = 1200.0 + 20.0 * rho
    co2_density = 5.0 * rho
    op_density = 3.0 * rho
    electron_density = 2.0 * rho
    neutral_east = 100.0 + rho
    neutral_up = -10.0 + 0.5 * rho
    co2_up = 50.0 + rho
    ion_up = 20.0 + 0.25 * rho
    solar_zenith_angle = 30.0 + rho
    fields = [
        np.broadcast_to(lon, (n_lon, n_lat, n_alt)),
        np.broadcast_to(lat, (n_lon, n_lat, n_alt)),
        np.broadcast_to(alt, (n_lon, n_lat, n_alt)),
        rho,
        temperature,
        e_temperature,
        co2_density,
        op_density,
        electron_density,
        neutral_east,
        neutral_up,
        co2_up,
        ion_up,
        solar_zenith_angle,
    ]

    with FortranFile(path, "w") as fh:
        fh.write_record(np.array([3.14], dtype=np.float64))
        fh.write_record(np.array([n_lon, n_lat, n_alt], dtype=np.int32))
        fh.write_record(np.array([len(variables)], dtype=np.int32))
        for name in variables:
            fh.write_record(np.frombuffer(name.encode("ascii").ljust(40, b" "), dtype=np.uint8))
        fh.write_record(np.array([2015, 3, 8, 10, 0, 0, 0], dtype=np.int32))
        for field in fields:
            fh.write_record(np.asarray(field, dtype=np.float64).reshape(-1, order="F"))


def test_read_ua_gitm_bin_returns_structured_dataset(tmp_path):
    path = tmp_path / "3DALL_t150308_100000.bin"
    _write_ua_file(path)

    dataset = read_ua_gitm_bin(path)
    smart_ds = SmartDs(dataset)
    smart_ds.merge_computation_graph(build_ua_graph(dataset.variables))

    assert dataset.points.shape == (3, 2, 4, 14)
    assert dataset.variables == [
        "Longitude",
        "Latitude",
        "Altitude",
        "Rho",
        "Temperature",
        "eTemperature",
        "[CO!D2!N             ]",
        "[O!U+!N              ]",
        "[e-                  ]",
        "V!Dn!N (east)",
        "V!Dn!N (up)",
        "V!Dn!N (up,CO!D2!N             )",
        "V!Di!N (up)",
        "Solar Zenith Angle",
    ]
    assert dataset.aux["UA_TIME"] == datetime(2015, 3, 8, 10, 0, 0)
    assert dataset.aux["UA_VERSION"] == 3.14
    assert dataset.aux["UA_NLON"] == 3
    np.testing.assert_allclose(smart_ds["Temperature"][0, 0, 0], 1010.0)
    np.testing.assert_allclose(smart_ds["Altitude"][0, 0, -1], 250000.0)
    np.testing.assert_allclose(smart_ds["Longitude [deg]"][0, 0, 0], -30.0)
    np.testing.assert_allclose(smart_ds["Latitude [deg]"][0, 0, 0], -45.0)
    np.testing.assert_allclose(smart_ds["Altitude [m]"][0, 0, -1], 250000.0)
    np.testing.assert_allclose(smart_ds["Tn [K]"][0, 0, 0], 1010.0)
    np.testing.assert_allclose(smart_ds["Te [K]"][0, 0, 0], 1220.0)
    np.testing.assert_allclose(smart_ds["neutral_number_density [1/m^3]"][0, 0, 0], 1.0)
    np.testing.assert_allclose(smart_ds["CO2 [1/m^3]"][0, 0, 0], 5.0)
    np.testing.assert_allclose(smart_ds["CO2 [kg/m^3]"][0, 0, 0], 5.0 * 44.0 * atomic_mass)
    np.testing.assert_allclose(smart_ds["O+ [1/m^3]"][0, 0, 0], 3.0)
    np.testing.assert_allclose(smart_ds["O+ [kg/m^3]"][0, 0, 0], 3.0 * 16.0 * atomic_mass)
    np.testing.assert_allclose(smart_ds["Ne [1/m^3]"][0, 0, 0], 2.0)
    np.testing.assert_allclose(smart_ds["e- [kg/m^3]"][0, 0, 0], 2.0 * electron_mass)
    np.testing.assert_allclose(
        smart_ds["neutral_mass_density [kg/m^3]"][0, 0, 0],
        5.0 * 44.0 * atomic_mass,
    )
    np.testing.assert_allclose(
        smart_ds["electron_mass_density [kg/m^3]"][0, 0, 0],
        2.0 * electron_mass,
    )
    np.testing.assert_allclose(smart_ds["Vn_east [m/s]"][0, 0, 0], 101.0)
    np.testing.assert_allclose(smart_ds["Vn_up [m/s]"][0, 0, 0], -9.5)
    np.testing.assert_allclose(smart_ds["Vn_up_CO2 [m/s]"][0, 0, 0], 51.0)
    np.testing.assert_allclose(smart_ds["Vi_up [m/s]"][0, 0, 0], 20.25)
    np.testing.assert_allclose(smart_ds["Solar Zenith Angle [rad]"][0, 0, 0], np.deg2rad(31.0))


def test_process_bin_file_writes_quicklook_pngs(tmp_path):
    path = tmp_path / "3DALL_t150308_100000.bin"
    _write_ua_file(path)

    process_bin_file(path)

    out_dir = tmp_path / "ua"
    assert (out_dir / "3dall_t150308_100000.ua.lat_alt.png").exists()
    assert (out_dir / "3dall_t150308_100000.ua.lon_lat.png").exists()
    assert (out_dir / "3dall_t150308_100000.ua.shell.png").exists()
    assert (out_dir / "3dall_t150308_100000.ua.shell_number_flux.png").exists()
    assert (out_dir / "3dall_t150308_100000.ua.shell_mass_flux.png").exists()
    assert (out_dir / "3dall_t150308_100000.ua.shell_flux_map.png").exists()
    flux_npz = out_dir / "3dall_t150308_100000.ua.shell_flux.npz"
    assert flux_npz.exists()
    with np.load(flux_npz, allow_pickle=False) as data:
        assert list(data["species_names"]) == ["CO2", "O+"]
        assert data["species_number_flux_1_s"].shape == (2, 4)
        assert data["species_mass_flux_kg_s"].shape == (2, 4)
        assert data["species_number_flux_density_1_m2_s"].shape == (2, 2, 2, 4)
        assert data["species_mass_flux_density_kg_m2_s"].shape == (2, 2, 2, 4)
        assert data["total_number_flux_1_s"].shape == (4,)
        assert data["total_mass_flux_kg_s"].shape == (4,)
        assert data["total_number_flux_density_1_m2_s"].shape == (2, 2, 4)
        assert data["total_mass_flux_density_kg_m2_s"].shape == (2, 2, 4)
