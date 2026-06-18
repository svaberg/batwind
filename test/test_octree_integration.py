from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from batcamp import Octree
from batcamp import OctreeInterpolator
from batread import Dataset

from batwind.algorithms.octree_integration import compute_octree_leaf_geometry
from batwind.algorithms.octree_integration import compute_octree_leaf_centers_and_volumes
from batwind.algorithms.octree_integration import cumulative_radius
from batwind.algorithms.octree_integration import cumulative_radius_exact_rpa
from batwind.algorithms.octree_integration import integrate_leaf_scalar
from batwind.algorithms.octree_integration import integrate_radial_shells_exact_rpa
from batwind.algorithms.octree_integration import leaf_point_mean
from batwind.algorithms.octree_integration import radial_emission_profile
from batwind.algorithms.octree_integration import radial_emission_profile_exact_rpa
from batwind.data.samples import data_file


def make_dataset_3d_one_cell():
    variables = [
        "X [R]",
        "Y [R]",
        "Z [R]",
    ]
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    corners = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=int)
    return Dataset(points, corners, aux={}, title="cube", variables=variables, zone="zcube")


def test_compute_octree_leaf_geometry_on_one_cell_cube():
    tree = Octree.from_ds(make_dataset_3d_one_cell())

    radius_r, volume_cm3 = compute_octree_leaf_geometry(tree, body_radius_cm=100.0)

    np.testing.assert_allclose(radius_r, [np.sqrt(3.0) * 0.5])
    np.testing.assert_allclose(volume_cm3, [100.0**3])


def test_leaf_point_mean_on_one_cell_cube():
    tree = Octree.from_ds(make_dataset_3d_one_cell())
    point_values = np.arange(8.0)

    out = leaf_point_mean(tree, point_values)

    np.testing.assert_allclose(out, [3.5])


def test_compute_octree_leaf_centers_and_volumes_matches_analytic_total_volume():
    tree = Octree.from_ds(make_dataset_3d_one_cell())

    leaf_centers, leaf_volumes = compute_octree_leaf_centers_and_volumes(tree, length_scale=1.0)

    np.testing.assert_allclose(leaf_centers, [[0.5, 0.5, 0.5]])
    np.testing.assert_allclose(np.sum(leaf_volumes), 1.0)


def test_integrate_leaf_scalar_of_one_returns_total_volume():
    tree = Octree.from_ds(make_dataset_3d_one_cell())
    point_values = np.ones(8, dtype=float)

    out = integrate_leaf_scalar(tree, point_values, length_scale=1.0)

    np.testing.assert_allclose(out, 1.0)


def test_sample_data_octree_volume_matches_integral_of_one():
    ds = Dataset.from_file(str(data_file("3d__var_2_n00060005.plt")))
    tree = Octree.from_ds(ds)

    leaf_count = int(np.asarray(tree.corners).shape[0])
    bounds = np.asarray(tree.cell_bounds, dtype=float)[:leaf_count]
    direct_total_volume = np.sum(np.prod(bounds[:, :, 1], axis=1))
    ones_total_volume = integrate_leaf_scalar(tree, np.ones(ds.points.shape[0], dtype=float), length_scale=1.0)

    _, helper_leaf_volumes = compute_octree_leaf_centers_and_volumes(tree, length_scale=1.0)

    np.testing.assert_allclose(np.sum(helper_leaf_volumes), direct_total_volume)
    np.testing.assert_allclose(ones_total_volume, direct_total_volume)


def test_sample_data_octree_volume_histogram_plot(tmp_path: Path):
    ds = Dataset.from_file(str(data_file("3d__var_2_n00060005.plt")))
    tree = Octree.from_ds(ds)

    _, leaf_volumes = compute_octree_leaf_centers_and_volumes(tree, length_scale=1.0)
    volume_edges = np.geomspace(np.min(leaf_volumes), np.max(leaf_volumes), 32)
    shell_volume = np.histogram(leaf_volumes, bins=volume_edges, weights=leaf_volumes)[0]
    cumulative_fraction = np.cumsum(shell_volume) / np.sum(shell_volume)

    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    ax.stairs(shell_volume, volume_edges, linewidth=2.0, label="Volume per leaf-volume bin")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Leaf volume [$R_*^3$]")
    ax.set_ylabel(r"Integrated volume in bin [$R_*^3$]")
    ax.set_title("Sample-data octree leaf-volume histogram")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(volume_edges[1:], cumulative_fraction, color="black", linewidth=1.5, label="Cumulative fraction")
    ax2.set_ylabel("Cumulative volume fraction")
    ax2.set_ylim(0.0, 1.02)

    png_path = tmp_path / "sample_data_octree_volume_histogram.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    assert png_path.exists()


def test_cumulative_radius_returns_requested_fraction():
    radial_distance_r = np.array([1.0, 2.0, 4.0], dtype=float)
    cell_emission = np.array([2.0, 3.0, 5.0], dtype=float)

    r50 = cumulative_radius(radial_distance_r, cell_emission, 0.5)
    r90 = cumulative_radius(radial_distance_r, cell_emission, 0.9)

    np.testing.assert_allclose(r50, 2.0)
    np.testing.assert_allclose(r90, 3.6)


def test_radial_emission_profile_bins_and_accumulates():
    radial_distance_r = np.array([1.0, 1.5, 3.0, 6.0], dtype=float)
    cell_emission = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)

    radius_r, shell_emission, cumulative_fraction = radial_emission_profile(
        radial_distance_r,
        cell_emission,
        n_bins=3,
    )

    assert radius_r.shape == (3,)
    np.testing.assert_allclose(shell_emission.sum(), 10.0)
    np.testing.assert_allclose(cumulative_fraction[-1], 1.0)


def test_exact_rpa_radial_shells_sum_to_full_spherical_volume_integral():
    ds = Dataset.from_file(str(data_file("3d__var_2_n00060005.plt")))
    tree = Octree.from_ds(ds)
    assert tree.tree_coord == "rpa"

    point_values = np.ones(ds.points.shape[0], dtype=float)
    radial_edges_r = np.asarray(tree.radial_edges, dtype=float)
    radial_edges_r = radial_edges_r[np.isfinite(radial_edges_r)]
    shell_integrals = integrate_radial_shells_exact_rpa(tree, point_values, radial_edges_r, length_scale=1.0)
    full_integral = float(
        OctreeInterpolator(tree, point_values).integrate_box(
            np.array([radial_edges_r[0], 0.0, 0.0], dtype=float),
            np.array([radial_edges_r[-1], np.pi, 2.0 * np.pi], dtype=float),
        )
    )

    np.testing.assert_allclose(np.sum(shell_integrals), full_integral)


def test_exact_rpa_cumulative_radius_matches_analytic_full_sphere_volume_for_constant_field():
    ds = Dataset.from_file(str(data_file("3d__var_2_n00060005.plt")))
    tree = Octree.from_ds(ds)
    assert tree.tree_coord == "rpa"

    point_values = np.ones(ds.points.shape[0], dtype=float)
    radial_edges_r = np.asarray(tree.radial_edges, dtype=float)
    radial_edges_r = radial_edges_r[np.isfinite(radial_edges_r)]
    r_min = float(radial_edges_r[0])
    r_max = float(radial_edges_r[-1])
    expected_r90 = (r_min**3 + 0.90 * (r_max**3 - r_min**3)) ** (1.0 / 3.0)

    r90 = cumulative_radius_exact_rpa(tree, point_values, 0.90, length_scale=1.0)
    radius_r, shell_emission, cumulative_fraction = radial_emission_profile_exact_rpa(
        tree,
        point_values,
        length_scale=1.0,
    )

    np.testing.assert_allclose(r90, expected_r90, rtol=0.0, atol=1.0e-6)
    np.testing.assert_allclose(np.sum(shell_emission), 4.0 * np.pi * (r_max**3 - r_min**3) / 3.0)
    np.testing.assert_allclose(cumulative_fraction[-1], 1.0)
    assert radius_r.shape == shell_emission.shape
