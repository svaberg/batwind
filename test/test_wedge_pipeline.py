from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import batwind.pipelines.wedge as wedge
from batwind.analysis.shells import sample_spherical_shells_fibonacci
from batwind.pyvista.convert import to_unstructured_grid
from batwind.smart_ds import SmartDs


def _make_cell_center_points():
    variables = [
        "X [R]",
        "Y [R]",
        "Z [R]",
        "Rho [g/cm^3]",
        "U_x [km/s]",
        "U_y [km/s]",
        "U_z [km/s]",
        "P [dyne/cm^2]",
    ]

    points: list[list[float]] = []

    def append_group(radius, polar, azimuth_values, *, rho_values, pressure_values, reference_vector):
        ref_x, ref_y, ref_z = reference_vector
        sin_polar = np.sin(float(polar))
        cos_polar = np.cos(float(polar))
        for azimuth, rho, pressure in zip(azimuth_values, rho_values, pressure_values, strict=True):
            cos_azimuth = np.cos(float(azimuth))
            sin_azimuth = np.sin(float(azimuth))
            x = float(radius) * sin_polar * cos_azimuth
            y = float(radius) * sin_polar * sin_azimuth
            z = float(radius) * cos_polar
            # Rotate one reference-meridian vector into the sample longitude.
            ux = cos_azimuth * ref_x - sin_azimuth * ref_y
            uy = sin_azimuth * ref_x + cos_azimuth * ref_y
            points.append([x, y, z, float(rho), ux, uy, float(ref_z), float(pressure)])

    append_group(
        2.0,
        np.pi / 3.0,
        [-0.12, -0.04, 0.04, 0.12],
        rho_values=[10.0, 12.0, 14.0, 16.0],
        pressure_values=[20.0, 22.0, 24.0, 26.0],
        reference_vector=(4.0, 5.0, 6.0),
    )
    append_group(
        3.0,
        0.4,
        [-0.08, 0.08],
        rho_values=[30.0, 34.0],
        pressure_values=[40.0, 44.0],
        reference_vector=(7.0, 8.0, 9.0),
    )
    append_group(
        2.5,
        0.75,
        [-0.06, 0.06],
        rho_values=[50.0, 54.0],
        pressure_values=[60.0, 64.0],
        reference_vector=(10.0, 11.0, 12.0),
    )
    return np.asarray(points, dtype=float), variables


def _make_geometry_points(cell_centers: np.ndarray, variables: list[str]) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(cell_centers[:, :3], dtype=float)
    offsets = np.asarray(
        [
            [-0.02, -0.01, -0.03],
            [0.02, -0.01, -0.03],
            [0.02, 0.01, -0.03],
            [-0.02, 0.01, -0.03],
            [-0.02, -0.01, 0.03],
            [0.02, -0.01, 0.03],
            [0.02, 0.01, 0.03],
            [-0.02, 0.01, 0.03],
        ],
        dtype=float,
    )

    n_cells = int(xyz.shape[0])
    points = np.zeros((n_cells * 8, len(variables)), dtype=float)
    corners = np.zeros((n_cells, 8), dtype=int)
    for cell_id, center in enumerate(xyz):
        start = 8 * cell_id
        stop = start + 8
        points[start:stop, :3] = center[None, :] + offsets
        corners[cell_id] = np.arange(start, stop, dtype=int)
    return points, corners


def _plt_tuple(points, corners, variables, *, title="demo", zone="3D   N=0000000"):
    return points, corners, {"RBODY": "1.00"}, title, list(variables), zone


def test_read_wedge_meridian_infers_pair_and_prefers_larger_point_file(tmp_path, monkeypatch):
    cell_points, variables = _make_cell_center_points()
    geometry_points, geometry_corners = _make_geometry_points(cell_points, variables)

    big_path = tmp_path / "3d__var_2_n00050000.plt"
    small_path = tmp_path / "3d__var_3_n00050000.plt"
    big_path.write_bytes(b"")
    small_path.write_bytes(b"")

    data_by_path = {
        str(big_path): _plt_tuple(geometry_points, geometry_corners, variables, title="big"),
        str(small_path): _plt_tuple(cell_points, np.zeros((1, 8), dtype=int), variables, title="small"),
    }
    monkeypatch.setattr(wedge, "read_plt", lambda filename: data_by_path[str(Path(filename))])

    result = wedge.read_wedge_meridian(small_path)

    assert result.geometry_path == big_path
    assert result.cell_center_path == small_path
    assert result.geometry.points.shape[0] == geometry_points.shape[0]
    assert result.geometry.corners.shape == geometry_corners.shape
    assert result.cell_centers.points.shape[0] == cell_points.shape[0]
    assert result.reduced_cell_centers.points.shape[0] == 3
    np.testing.assert_array_equal(result.sample_counts, np.array([4, 2, 2], dtype=int))


def test_read_wedge_meridian_rotates_vectors_before_grouped_average(tmp_path, monkeypatch):
    cell_points, variables = _make_cell_center_points()
    geometry_points, geometry_corners = _make_geometry_points(cell_points, variables)

    big_path = tmp_path / "3d__var_2_n00050000.plt"
    small_path = tmp_path / "3d__var_3_n00050000.plt"
    big_path.write_bytes(b"")
    small_path.write_bytes(b"")

    data_by_path = {
        str(big_path): _plt_tuple(geometry_points, geometry_corners, variables),
        str(small_path): _plt_tuple(cell_points, np.zeros((1, 8), dtype=int), variables),
    }
    monkeypatch.setattr(wedge, "read_plt", lambda filename: data_by_path[str(Path(filename))])

    result = wedge.read_wedge_meridian(big_path, pair_path=small_path, reference_azimuth_rad=0.0)
    reduced = result.reduced_cell_centers

    np.testing.assert_allclose(reduced["Y [R]"], [0.0, 0.0, 0.0], atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(reduced["U_x [km/s]"], [4.0, 10.0, 7.0], atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(reduced["U_y [km/s]"], [5.0, 11.0, 8.0], atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(reduced["U_z [km/s]"], [6.0, 12.0, 9.0], atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(reduced["Rho [g/cm^3]"], [13.0, 52.0, 32.0], atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(reduced["P [dyne/cm^2]"], [23.0, 62.0, 42.0], atol=1.0e-12, rtol=0.0)


def test_find_wedge_pair_paths_requires_exactly_two_same_timestep_files(tmp_path):
    lone_path = tmp_path / "3d__var_2_n00050000.plt"
    lone_path.write_bytes(b"")

    try:
        wedge.find_wedge_pair_paths(lone_path)
    except ValueError as exc:
        assert "Expected exactly two same-timestep wedge files" in str(exc)
    else:
        raise AssertionError("Expected one ValueError when only one wedge file exists")


def test_plot_wedge_meridional_field_writes_rho_quicklook(tmp_path, monkeypatch):
    cell_points, variables = _make_cell_center_points()
    geometry_points, geometry_corners = _make_geometry_points(cell_points, variables)

    big_path = tmp_path / "3d__var_2_n00050000.plt"
    small_path = tmp_path / "3d__var_3_n00050000.plt"
    png_path = tmp_path / "wedge_rho_meridional.png"
    big_path.write_bytes(b"")
    small_path.write_bytes(b"")

    data_by_path = {
        str(big_path): _plt_tuple(geometry_points, geometry_corners, variables),
        str(small_path): _plt_tuple(cell_points, np.zeros((1, 8), dtype=int), variables),
    }
    monkeypatch.setattr(wedge, "read_plt", lambda filename: data_by_path[str(Path(filename))])

    result = wedge.read_wedge_meridian(big_path, pair_path=small_path, reference_azimuth_rad=0.0)
    figure, axis = wedge.plot_wedge_meridional_field(result, output_path=png_path)
    try:
        assert png_path.exists()
        assert png_path.stat().st_size > 0
        assert axis.get_xlabel() == r"$R_{\mathrm{cyl}}$ $(R_\star)$"
        assert axis.get_ylabel() == r"$z$ $(R_\star)$"
    finally:
        plt.close(figure)


def test_axisymmetric_wedge_ds_resample_rotates_cartesian_vectors_and_sets_query_coords(tmp_path, monkeypatch):
    cell_points, variables = _make_cell_center_points()
    geometry_points, geometry_corners = _make_geometry_points(cell_points, variables)

    big_path = tmp_path / "3d__var_2_n00050000.plt"
    small_path = tmp_path / "3d__var_3_n00050000.plt"
    big_path.write_bytes(b"")
    small_path.write_bytes(b"")

    data_by_path = {
        str(big_path): _plt_tuple(geometry_points, geometry_corners, variables),
        str(small_path): _plt_tuple(cell_points, np.zeros((1, 8), dtype=int), variables),
    }
    monkeypatch.setattr(wedge, "read_plt", lambda filename: data_by_path[str(Path(filename))])

    axisym = wedge.AxisymmetricWedgeDs.from_file(
        big_path,
        pair_path=small_path,
        reference_azimuth_rad=0.0,
        body_radius_m=5.0,
    )

    radius = 3.0
    polar = 0.4
    azimuth = np.pi / 4.0
    x, y, z = wedge._rpa_to_cartesian(
        np.array([radius], dtype=float),
        np.array([polar], dtype=float),
        np.array([azimuth], dtype=float),
    )
    sampled = axisym.resample(
        np.column_stack((x, y, z)),
        fields=["Rho [kg/m^3]", "U_x [m/s]", "U_y [m/s]", "U_z [m/s]", "U_r [m/s]", "Lon [deg]", "Lat [deg]", "R [m]"],
        method="octree",
    )

    assert isinstance(sampled, SmartDs)
    np.testing.assert_allclose(sampled["X [R]"], x, atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(sampled["Y [R]"], y, atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(sampled["Z [R]"], z, atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(sampled["Rho [kg/m^3]"], [32000.0], atol=1.0e-10, rtol=0.0)

    cos_azimuth = np.cos(azimuth)
    sin_azimuth = np.sin(azimuth)
    expected_ux = 1000.0 * (cos_azimuth * 7.0 - sin_azimuth * 8.0)
    expected_uy = 1000.0 * (sin_azimuth * 7.0 + cos_azimuth * 8.0)
    expected_uz = 9000.0
    expected_ur = (
        expected_ux * x[0]
        + expected_uy * y[0]
        + expected_uz * z[0]
    ) / radius

    np.testing.assert_allclose(sampled["U_x [m/s]"], [expected_ux], atol=1.0e-10, rtol=0.0)
    np.testing.assert_allclose(sampled["U_y [m/s]"], [expected_uy], atol=1.0e-10, rtol=0.0)
    np.testing.assert_allclose(sampled["U_z [m/s]"], [expected_uz], atol=1.0e-10, rtol=0.0)
    np.testing.assert_allclose(sampled["U_r [m/s]"], [expected_ur], atol=1.0e-10, rtol=0.0)
    np.testing.assert_allclose(sampled["Lon [deg]"], [45.0], atol=1.0e-10, rtol=0.0)
    np.testing.assert_allclose(sampled["Lat [deg]"], [np.rad2deg((0.5 * np.pi) - polar)], atol=1.0e-10, rtol=0.0)
    np.testing.assert_allclose(sampled["R [m]"], [15.0], atol=1.0e-10, rtol=0.0)


def test_axisymmetric_wedge_ds_drives_shell_sampling_helper(tmp_path, monkeypatch):
    cell_points, variables = _make_cell_center_points()
    geometry_points, geometry_corners = _make_geometry_points(cell_points, variables)

    big_path = tmp_path / "3d__var_2_n00050000.plt"
    small_path = tmp_path / "3d__var_3_n00050000.plt"
    big_path.write_bytes(b"")
    small_path.write_bytes(b"")

    data_by_path = {
        str(big_path): _plt_tuple(geometry_points, geometry_corners, variables),
        str(small_path): _plt_tuple(cell_points, np.zeros((1, 8), dtype=int), variables),
    }
    monkeypatch.setattr(wedge, "read_plt", lambda filename: data_by_path[str(Path(filename))])

    axisym = wedge.AxisymmetricWedgeDs.from_file(
        small_path,
        reference_azimuth_rad=0.0,
        body_radius_m=2.0,
    )
    shells = sample_spherical_shells_fibonacci(
        axisym,
        [2.4],
        fields=["Rho [kg/m^3]", "U_r [m/s]"],
        n_points=8,
        method="nearest",
        length_unit_to_m=2.0,
    )

    assert isinstance(shells, SmartDs)
    assert shells["Rho [kg/m^3]"].shape == (1, 8, 1)
    assert shells["U_r [m/s]"].shape == (1, 8, 1)
    assert shells["dA [m^2]"].shape == (1, 8, 1)
    assert np.all(np.isfinite(shells["Rho [kg/m^3]"]))
    assert np.all(np.isfinite(shells["U_r [m/s]"]))


def test_axisymmetric_wedge_revolves_to_structured_volume(tmp_path, monkeypatch):
    cell_points, variables = _make_cell_center_points()
    geometry_points, geometry_corners = _make_geometry_points(cell_points, variables)

    big_path = tmp_path / "3d__var_2_n00050000.plt"
    small_path = tmp_path / "3d__var_3_n00050000.plt"
    big_path.write_bytes(b"")
    small_path.write_bytes(b"")

    data_by_path = {
        str(big_path): _plt_tuple(geometry_points, geometry_corners, variables),
        str(small_path): _plt_tuple(cell_points, np.zeros((1, 8), dtype=int), variables),
    }
    monkeypatch.setattr(wedge, "read_plt", lambda filename: data_by_path[str(Path(filename))])

    axisym = wedge.AxisymmetricWedgeDs.from_file(
        big_path,
        pair_path=small_path,
        reference_azimuth_rad=0.0,
        body_radius_m=2.0,
    )
    volume_ds = wedge.revolve_axisymmetric_wedge_to_volume(
        axisym,
        n_radius=4,
        n_polar=4,
        n_azimuth=6,
    )

    assert isinstance(volume_ds, SmartDs)
    assert volume_ds.raw.points.shape == (96, len(axisym.raw.variables))
    assert volume_ds.raw.corners.shape == (54, 8)
    grid = to_unstructured_grid(
        volume_ds,
        point_data={"Rho [kg/m^3]": volume_ds["Rho [kg/m^3]"]},
    )
    assert grid.n_points == 96
    assert grid.n_cells == 54
    radius_range, polar_range = wedge.infer_axisymmetric_spherical_ranges(axisym)
    radius_nodes = np.linspace(radius_range[0], radius_range[1], 4)
    polar_nodes = np.linspace(polar_range[0], polar_range[1], 4)
    azimuth_nodes = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    interior_xyz = np.column_stack(
        wedge._rpa_to_cartesian(
            np.asarray([0.5 * (radius_nodes[0] + radius_nodes[1])], dtype=float),
            np.asarray([0.5 * (polar_nodes[0] + polar_nodes[1])], dtype=float),
            np.asarray([0.5 * (azimuth_nodes[0] + azimuth_nodes[1])], dtype=float),
        )
    )
    sampled = volume_ds.resample(
        interior_xyz,
        fields=["Rho [kg/m^3]"],
        method="octree",
    )
    assert np.isfinite(sampled["Rho [kg/m^3]"]).all()


def test_axisymmetric_wedge_ds_tolerates_missing_param_blocks(tmp_path, monkeypatch):
    cell_points, variables = _make_cell_center_points()
    geometry_points, geometry_corners = _make_geometry_points(cell_points, variables)

    big_path = tmp_path / "3d__var_2_n00050000.plt"
    small_path = tmp_path / "3d__var_3_n00050000.plt"
    param_path = tmp_path / "PARAM.in"
    big_path.write_bytes(b"")
    small_path.write_bytes(b"")
    param_path.write_text("")

    data_by_path = {
        str(big_path): _plt_tuple(geometry_points, geometry_corners, variables),
        str(small_path): _plt_tuple(cell_points, np.zeros((1, 8), dtype=int), variables),
    }

    class _DummyParamIn:
        sessions = [{"root": {}}]

    monkeypatch.setattr(wedge, "read_plt", lambda filename: data_by_path[str(Path(filename))])
    monkeypatch.setattr(wedge, "find_param_in", lambda _path: param_path)
    monkeypatch.setattr(wedge.ParamIn, "from_file", classmethod(lambda cls, _path: _DummyParamIn()))
    monkeypatch.setattr(wedge.StarParams, "from_param_in", classmethod(lambda cls, *_args, **_kwargs: None))
    monkeypatch.setattr(wedge.TransitionRegionParams, "from_param_in", classmethod(lambda cls, *_args, **_kwargs: None))

    axisym = wedge.AxisymmetricWedgeDs.from_file(
        big_path,
        pair_path=small_path,
        reference_azimuth_rad=0.0,
        body_radius_m=1.0,
    )

    assert isinstance(axisym, wedge.AxisymmetricWedgeDs)
    assert "Star_radius_m" not in axisym.raw.aux
