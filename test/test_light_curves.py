import numpy as np

from batread import Dataset

from batwind.physics.light_curves import band_intensity_image_si
from batwind.physics.light_curves import band_light_curve_si
from batwind.physics.light_curves import view_direction_from_inclination_phase
from batwind.recipes.batsrus import build_batsrus_graph
from batwind.smart_ds import SmartDs


def make_equatorial_octree_dataset() -> Dataset:
    variables = ["X [R]", "Y [R]", "Z [R]"]
    x_edges = np.array([-0.2, 0.2], dtype=float)
    y_edges = np.linspace(-1.4, 1.4, 9)
    z_edges = np.array([-0.2, 0.2], dtype=float)
    grid_x, grid_y, grid_z = np.meshgrid(x_edges, y_edges, z_edges, indexing="ij")

    def idx(i: int, j: int, k: int) -> int:
        return (i * y_edges.size + j) * z_edges.size + k

    corners = []
    for j in range(y_edges.size - 1):
        corners.append(
            [
                idx(0, j, 0),
                idx(1, j, 0),
                idx(1, j + 1, 0),
                idx(0, j + 1, 0),
                idx(0, j, 1),
                idx(1, j, 1),
                idx(1, j + 1, 1),
                idx(0, j + 1, 1),
            ]
        )
    return Dataset(
        np.column_stack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel())),
        np.asarray(corners, dtype=int),
        aux={},
        title="equatorial-octree",
        variables=variables,
        zone="z8",
    )


def test_view_direction_from_inclination_phase_matches_library_convention():
    np.testing.assert_allclose(view_direction_from_inclination_phase(0.0, 123.0), [0.0, 0.0, 1.0])
    np.testing.assert_allclose(view_direction_from_inclination_phase(90.0, 0.0), [0.0, 1.0, 0.0], atol=1.0e-12)
    np.testing.assert_allclose(view_direction_from_inclination_phase(90.0, 90.0), [1.0, 0.0, 0.0], atol=1.0e-12)


def test_band_light_curve_si_is_periodic_and_phase_variable_for_asymmetric_emission():
    dataset = make_equatorial_octree_dataset()
    sds = SmartDs(dataset)
    sds.merge_computation_graph(build_batsrus_graph(tuple(dataset.variables), body_radius_m=1.0))
    y_r = np.asarray(sds["Y [R]"], dtype=float)
    point_emissivity = np.zeros(y_r.shape, dtype=float)
    point_emissivity[y_r <= -1.05] = 4.0
    point_emissivity[y_r >= 1.05] = 1.0

    out = band_light_curve_si(
        sds,
        point_emissivity,
        np.array([0.0, 180.0, 360.0]),
        inclination_deg=90.0,
        image_n=64,
        side_length_r=4.0,
    )

    np.testing.assert_allclose(out["radiant_intensity_w_sr"][0], out["radiant_intensity_w_sr"][2], rtol=1.0e-12)
    assert out["radiant_intensity_w_sr"][0] > out["radiant_intensity_w_sr"][1]


def test_band_intensity_image_si_is_positive_for_visible_emission():
    dataset = make_equatorial_octree_dataset()
    sds = SmartDs(dataset)
    sds.merge_computation_graph(build_batsrus_graph(tuple(dataset.variables), body_radius_m=1.0))
    y_r = np.asarray(sds["Y [R]"], dtype=float)
    point_emissivity = np.zeros(y_r.shape, dtype=float)
    point_emissivity[np.abs(y_r) >= 1.05] = 2.0

    out = band_intensity_image_si(
        sds,
        point_emissivity,
        inclination_deg=90.0,
        phase_deg=0.0,
        image_n=64,
        side_length_r=4.0,
    )

    assert out["image"].shape == (64, 64)
    assert np.sum(out["image"]) > 0.0
