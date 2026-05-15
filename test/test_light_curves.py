import numpy as np

from batread import Dataset

from batwind.physics.light_curves import band_intensity_image_si
from batwind.physics.light_curves import band_light_curve_si
from batwind.physics.light_curves import view_direction_from_inclination_phase
from batwind.recipes.batsrus import build_batsrus_graph
from batwind.smart_ds import SmartDs


def make_two_cell_equatorial_dataset() -> Dataset:
    variables = ["X [R]", "Y [R]", "Z [R]"]
    front = np.array(
        [
            [-0.2, -1.4, -0.2],
            [0.2, -1.4, -0.2],
            [0.2, -1.0, -0.2],
            [-0.2, -1.0, -0.2],
            [-0.2, -1.4, 0.2],
            [0.2, -1.4, 0.2],
            [0.2, -1.0, 0.2],
            [-0.2, -1.0, 0.2],
        ],
        dtype=float,
    )
    back = np.array(
        [
            [-0.2, 1.0, -0.2],
            [0.2, 1.0, -0.2],
            [0.2, 1.4, -0.2],
            [-0.2, 1.4, -0.2],
            [-0.2, 1.0, 0.2],
            [0.2, 1.0, 0.2],
            [0.2, 1.4, 0.2],
            [-0.2, 1.4, 0.2],
        ],
        dtype=float,
    )
    points = np.vstack([front, back])
    corners = np.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 15],
        ],
        dtype=int,
    )
    return Dataset(points, corners, aux={}, title="equatorial-two-cell", variables=variables, zone="z2")


def test_view_direction_from_inclination_phase_matches_library_convention():
    np.testing.assert_allclose(view_direction_from_inclination_phase(0.0, 123.0), [0.0, 0.0, 1.0])
    np.testing.assert_allclose(view_direction_from_inclination_phase(90.0, 0.0), [0.0, 1.0, 0.0], atol=1.0e-12)
    np.testing.assert_allclose(view_direction_from_inclination_phase(90.0, 90.0), [1.0, 0.0, 0.0], atol=1.0e-12)


def test_band_light_curve_si_is_periodic_and_phase_variable_for_asymmetric_emission():
    dataset = make_two_cell_equatorial_dataset()
    sds = SmartDs(dataset)
    sds.merge_computation_graph(build_batsrus_graph(tuple(dataset.variables), body_radius_m=1.0))
    point_emissivity = np.concatenate([np.full(8, 4.0, dtype=float), np.full(8, 1.0, dtype=float)])

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
    dataset = make_two_cell_equatorial_dataset()
    sds = SmartDs(dataset)
    sds.merge_computation_graph(build_batsrus_graph(tuple(dataset.variables), body_radius_m=1.0))
    point_emissivity = np.concatenate([np.full(8, 2.0, dtype=float), np.full(8, 2.0, dtype=float)])

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
