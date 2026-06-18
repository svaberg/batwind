from pathlib import Path
import tempfile

import colorcet as cc
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
import numpy as np
import pytest
from matplotlib.cm import ScalarMappable
from colorspacious import cspace_convert

from batcamp import camera_rays
from batcamp import Octree
from batread import Dataset

from batwind.algorithms.line_profiles import histogram_leaf_los_velocity
from batwind.algorithms.line_profiles import render_los_velocity_histogram_cube
from batwind.algorithms.line_profiles import summarize_spectral_cube
from batwind.data.samples import data_file
from batwind.smart_ds import SmartDs

TOY_LINE_REST_WAVELENGTH_ANGSTROM = 171.073
TOY_LINE_LOG10_T_PEAK = 5.90
TOY_LINE_LOG10_T_WIDTH = 0.08
TOY_LINE_PEAK_CONTRIBUTION_W_M3_SR = 1.0
TOY_LINE_VELOCITY_LIMIT_KMS = 5.0
SC_SAMPLE = "3d__var_4_n00044000.plt"


@pytest.fixture
def sc_sample_path() -> Path:
    return data_file(SC_SAMPLE)


def toy_line_contribution_function_w_m3_sr(temperature_k: np.ndarray) -> np.ndarray:
    """
    Return one toy narrow-line contribution function in SI units.

    The Gaussian factor is dimensionless. The prefactor sets the units to
    ``W m^3 sr^-1``, so multiplying by ``n_e^2`` yields a local line emissivity
    in ``W m^-3 sr^-1``.
    """
    contribution = np.zeros_like(temperature_k, dtype=float)
    valid_temperature = temperature_k > 0.0
    log10_temperature = np.log10(temperature_k[valid_temperature])
    contribution[valid_temperature] = TOY_LINE_PEAK_CONTRIBUTION_W_M3_SR * np.exp(
        -0.5 * ((log10_temperature - TOY_LINE_LOG10_T_PEAK) / TOY_LINE_LOG10_T_WIDTH) ** 2
    )
    return contribution


def toy_line_emissivity_w_m3_sr(sds: SmartDs) -> np.ndarray:
    """
    Return one toy line emissivity field in SI units.

    This test line is not physically calibrated, but it carries the same
    dimensions as the paper's line model:
    - contribution function ``G(T)``: ``W m^3 sr^-1``
    - electron density ``n_e``: ``m^-3``
    - emissivity ``epsilon = G(T) n_e^2``: ``W m^-3 sr^-1``
    """
    point_temperature_k = np.asarray(sds["te [K]"], dtype=float)
    point_electron_density_m3 = np.asarray(sds["Ne [1/m^3]"], dtype=float)
    point_line_contribution_w_m3_sr = toy_line_contribution_function_w_m3_sr(point_temperature_k)
    return point_line_contribution_w_m3_sr * point_electron_density_m3**2


def make_dataset_two_cells_x():
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
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 0.0, 1.0],
            [2.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    corners = np.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 8, 9, 2, 5, 10, 11, 6],
        ],
        dtype=int,
    )
    return Dataset(points, corners, aux={}, title="two-cells-x", variables=variables, zone="z2x")


def test_histogram_leaf_los_velocity_with_pointwise_x_velocity():
    dataset = make_dataset_two_cells_x()
    tree = Octree.from_ds(dataset)
    point_coords = np.asarray(dataset.points, dtype=float)
    point_velocity_vectors = np.column_stack(
        (
            point_coords[:, 0],
            np.zeros(point_coords.shape[0], dtype=float),
            np.zeros(point_coords.shape[0], dtype=float),
        )
    )
    velocity_edges = np.array([0.0, 1.0, 2.0], dtype=float)

    out = histogram_leaf_los_velocity(
        tree,
        point_velocity_vectors,
        velocity_edges,
        view_direction=np.array([1.0, 0.0, 0.0]),
        length_scale=1.0,
    )

    np.testing.assert_allclose(out["leaf_los_velocity"], [0.5, 1.5])
    np.testing.assert_allclose(out["leaf_volumes"], [1.0, 1.0])
    np.testing.assert_allclose(out["histogram"], [1.0, 1.0])


def test_render_los_velocity_histogram_cube_on_two_cells():
    dataset = make_dataset_two_cells_x()
    tree = Octree.from_ds(dataset)
    point_coords = np.asarray(dataset.points, dtype=float)
    point_velocity_vectors = np.column_stack(
        (
            point_coords[:, 0],
            np.zeros(point_coords.shape[0], dtype=float),
            np.zeros(point_coords.shape[0], dtype=float),
        )
    )
    velocity_edges = np.array([0.0, 1.0, 2.0], dtype=float)
    origins = np.array([[[-1.0, 0.5, 0.5]]], dtype=float)
    directions = np.array([[[1.0, 0.0, 0.0]]], dtype=float)

    out = render_los_velocity_histogram_cube(
        tree,
        point_velocity_vectors,
        velocity_edges,
        origins,
        directions,
        length_scale=1.0,
    )

    np.testing.assert_allclose(out["spectral_cube"].shape, (1, 1, 2))
    np.testing.assert_allclose(out["spectral_cube"][0, 0], [1.0, 1.0])
    np.testing.assert_allclose(out["ray_segment_counts"], [[2]])


def test_histogram_leaf_los_velocity_plot(tmp_path: Path):
    dataset = make_dataset_two_cells_x()
    tree = Octree.from_ds(dataset)
    point_coords = np.asarray(dataset.points, dtype=float)
    point_velocity_vectors = np.column_stack(
        (
            point_coords[:, 0],
            np.zeros(point_coords.shape[0], dtype=float),
            np.zeros(point_coords.shape[0], dtype=float),
        )
    )
    velocity_edges = np.linspace(0.0, 2.0, 9)
    out = histogram_leaf_los_velocity(
        tree,
        point_velocity_vectors,
        velocity_edges,
        view_direction=np.array([1.0, 0.0, 0.0]),
        length_scale=1.0,
    )

    fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    ax.stairs(out["histogram"], out["velocity_edges"], linewidth=2.0)
    ax.set_xlabel("LOS velocity [arb.]")
    ax.set_ylabel(r"Volume per bin [arb.$^3$]")
    ax.set_title("Two-cell LOS velocity histogram")
    ax.grid(True, alpha=0.3)
    png_path = tmp_path / "two_cell_velocity_histogram.png"
    fig.savefig(png_path)
    plt.close(fig)

    assert png_path.exists()


@pytest.mark.pooch
def test_histogram_leaf_los_velocity_plot_on_sc_sample(tmp_path: Path, sc_sample_path: Path):
    dataset = Dataset.from_file(str(sc_sample_path))
    tree = Octree.from_ds(dataset)
    variables = {name: i for i, name in enumerate(dataset.variables)}
    point_velocity_vectors = np.column_stack(
        (
            np.asarray(dataset.points[:, variables["U_x [km/s]"]], dtype=float),
            np.asarray(dataset.points[:, variables["U_y [km/s]"]], dtype=float),
            np.asarray(dataset.points[:, variables["U_z [km/s]"]], dtype=float),
        )
    )
    projected_velocity = point_velocity_vectors[:, 1]
    velocity_limit = float(np.max(np.abs(projected_velocity)))
    velocity_edges = np.linspace(-velocity_limit, velocity_limit, 97)

    out = histogram_leaf_los_velocity(
        tree,
        point_velocity_vectors,
        velocity_edges,
        view_direction=np.array([0.0, 1.0, 0.0]),
        length_scale=1.0,
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    ax.stairs(out["histogram"], out["velocity_edges"], linewidth=2.0)
    ax.set_xlabel("LOS velocity along +Y [km/s]")
    ax.set_ylabel(r"Volume per bin [$R_*^3$]")
    ax.set_title("SC sample LOS velocity histogram")
    ax.grid(True, alpha=0.3)
    png_path = tmp_path / "sc_sample_los_velocity_histogram.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    assert png_path.exists()

    histogram_fraction = out["histogram"] / np.sum(out["leaf_weights"])
    histogram_density = histogram_fraction / np.diff(out["velocity_edges"])
    cumulative_fraction = np.cumsum(histogram_fraction)
    differential_density = np.gradient(cumulative_fraction, out["velocity_centers"])

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    ax.plot(out["velocity_centers"], cumulative_fraction, linewidth=2.0)
    ax.set_xlabel("LOS velocity along +Y [km/s]")
    ax.set_ylabel("Cumulative volume fraction")
    ax.set_title("SC sample cumulative LOS velocity distribution")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.stairs(histogram_density, out["velocity_edges"], color="C1", linewidth=1.2)
    ax2.plot(out["velocity_centers"], differential_density, color="C2", linewidth=1.8)
    ax2.set_ylabel(r"Density [1 / (km/s)]", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")

    cumulative_png_path = tmp_path / "sc_sample_los_velocity_cumulative.png"
    fig.savefig(cumulative_png_path, dpi=200)
    plt.close(fig)

    assert cumulative_png_path.exists()


@pytest.mark.pooch
def test_weighted_line_profile_plot_on_sc_sample(tmp_path: Path, sc_sample_path: Path):
    sds = SmartDs.from_file(sc_sample_path)
    tree = Octree.from_ds(sds.raw)
    variables = {name: i for i, name in enumerate(sds.raw.variables)}
    point_velocity_vectors = np.column_stack(
        (
            np.asarray(sds.raw.points[:, variables["U_x [km/s]"]], dtype=float),
            np.asarray(sds.raw.points[:, variables["U_y [km/s]"]], dtype=float),
            np.asarray(sds.raw.points[:, variables["U_z [km/s]"]], dtype=float),
        )
    )
    body_radius_m = float(sds["RBODY [m]"])
    point_line_emissivity_w_m3_sr = toy_line_emissivity_w_m3_sr(sds)

    projected_velocity = point_velocity_vectors[:, 1]
    velocity_limit = float(np.max(np.abs(projected_velocity)))
    velocity_edges = np.linspace(-velocity_limit, velocity_limit, 97)
    out = histogram_leaf_los_velocity(
        tree,
        point_velocity_vectors,
        velocity_edges,
        view_direction=np.array([0.0, 1.0, 0.0]),
        length_scale=body_radius_m,
        point_weights=point_line_emissivity_w_m3_sr,
    )

    histogram_fraction = out["histogram"] / np.sum(out["leaf_weights"])
    histogram_density = histogram_fraction / np.diff(out["velocity_edges"])
    cumulative_fraction = np.cumsum(histogram_fraction)
    differential_density = np.gradient(cumulative_fraction, out["velocity_centers"])

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    ax.plot(out["velocity_centers"], cumulative_fraction, linewidth=2.0)
    ax.set_xlabel("LOS velocity along +Y [km/s]")
    ax.set_ylabel("Cumulative line fraction")
    ax.set_title(f"Weighted LOS line profile ({TOY_LINE_REST_WAVELENGTH_ANGSTROM:.3f} A)")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.stairs(histogram_density, out["velocity_edges"], color="C1", linewidth=1.2)
    ax2.plot(out["velocity_centers"], differential_density, color="C2", linewidth=1.8)
    ax2.set_ylabel(r"Line fraction density [1 / (km/s)]", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")

    png_path = tmp_path / "sc_sample_weighted_line_profile.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    assert np.sum(out["histogram"]) > 0.0
    assert png_path.exists()


@pytest.mark.pooch
def test_weighted_line_spectral_image_plot_on_sc_sample(tmp_path: Path, sc_sample_path: Path):
    sds = SmartDs.from_file(sc_sample_path)
    tree = Octree.from_ds(sds.raw)
    variables = {name: i for i, name in enumerate(sds.raw.variables)}
    point_velocity_vectors = np.column_stack(
        (
            np.asarray(sds.raw.points[:, variables["U_x [km/s]"]], dtype=float),
            np.asarray(sds.raw.points[:, variables["U_y [km/s]"]], dtype=float),
            np.asarray(sds.raw.points[:, variables["U_z [km/s]"]], dtype=float),
        )
    )
    body_radius_m = float(sds["RBODY [m]"])
    point_line_emissivity_w_m3_sr = toy_line_emissivity_w_m3_sr(sds)

    x = np.asarray(sds["X [R]"], dtype=float)
    y = np.asarray(sds["Y [R]"], dtype=float)
    z = np.asarray(sds["Z [R]"], dtype=float)
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    z_min, z_max = float(np.min(z)), float(np.max(z))
    x_center = 0.5 * (x_min + x_max)
    z_center = 0.5 * (z_min + z_max)
    width = 4.0
    height = 4.0
    y_pad = max(1.0e-6 * (y_max - y_min), 1.0e-6)
    image_size = 128
    origins, directions = camera_rays(
        origin=(x_center, y_min - y_pad, z_center),
        target=(x_center, y_max + y_pad, z_center),
        up=(0.0, 0.0, 1.0),
        nx=image_size,
        ny=image_size,
        width=width,
        height=height,
        projection="parallel",
    )
    velocity_edges = np.linspace(-TOY_LINE_VELOCITY_LIMIT_KMS, TOY_LINE_VELOCITY_LIMIT_KMS, 18)
    cache_dir = Path(tempfile.gettempdir()) / "batwind-line-profile-cache"
    cache_dir.mkdir(exist_ok=True)
    cube_cache_path = cache_dir / f"sc_weighted_line_spectral_cube_{image_size}x{image_size}_17bins_pm5kms_si_line.npz"
    if cube_cache_path.exists():
        cached = np.load(cube_cache_path)
        cube = np.asarray(cached["spectral_cube"], dtype=float)
        velocity_centers = np.asarray(cached["velocity_centers"], dtype=float)
    else:
        out = render_los_velocity_histogram_cube(
            tree,
            point_velocity_vectors,
            velocity_edges,
            origins,
            directions,
            point_weights=point_line_emissivity_w_m3_sr,
            length_scale=body_radius_m,
        )
        cube = out["spectral_cube"]
        velocity_centers = out["velocity_centers"]
        np.savez(
            cube_cache_path,
            spectral_cube=cube,
            velocity_centers=velocity_centers,
        )

    cube_artifact_path = tmp_path / "sc_weighted_line_spectral_cube.npz"
    np.savez(
        cube_artifact_path,
        spectral_cube=cube,
        velocity_centers=velocity_centers,
        velocity_edges=velocity_edges,
    )

    total_image = np.sum(cube, axis=-1)
    one_bin_image = total_image
    positive_one_bin = one_bin_image[one_bin_image > 0.0]
    one_bin_vmin = float(np.percentile(positive_one_bin, 1.0))
    one_bin_vmax = float(np.max(positive_one_bin))

    fig, ax = plt.subplots(figsize=(6.8, 6.0), constrained_layout=True)
    im = ax.imshow(
        one_bin_image,
        origin="lower",
        extent=(x_center - 0.5 * width, x_center + 0.5 * width, z_center - 0.5 * height, z_center + 0.5 * height),
        cmap="gray",
        norm=LogNorm(vmin=one_bin_vmin, vmax=one_bin_vmax),
    )
    ax.set_title("Weighted line traced image (one velocity bin)")
    ax.set_xlabel("X [R]")
    ax.set_ylabel("Z [R]")
    one_bin_cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    one_bin_cbar.set_label(r"Weighted line intensity [W m$^{-2}$ sr$^{-1}$]")
    one_bin_png_path = tmp_path / "sc_weighted_line_one_bin.png"
    fig.savefig(one_bin_png_path, dpi=200)
    plt.close(fig)

    assert one_bin_png_path.exists()

    blue_image = np.sum(cube[..., :4], axis=-1)
    core_image = np.sum(cube[..., 6:10], axis=-1)
    red_image = np.sum(cube[..., -4:], axis=-1)

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 8.0), constrained_layout=True)
    panels = [
        (total_image, "Total line"),
        (blue_image, "Blue wing"),
        (core_image, "Line core"),
        (red_image, "Red wing"),
    ]
    extent = (x_center - 0.5 * width, x_center + 0.5 * width, z_center - 0.5 * height, z_center + 0.5 * height)
    for ax, (image, title) in zip(axes.ravel(), panels, strict=True):
        im = ax.imshow(image, origin="lower", extent=extent, cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("X [R]")
        ax.set_ylabel("Z [R]")
        fig.colorbar(im, ax=ax, shrink=0.8)

    png_path = tmp_path / "sc_weighted_line_spectral_image.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    assert np.sum(cube) > 0.0
    assert png_path.exists()

    summary = summarize_spectral_cube(cube, velocity_centers)
    mean_velocity = summary["mean_velocity"]
    # The current summary image is built from histogram moments after binning in
    # LOS velocity. If the visual banding becomes too strong, first consider
    # increasing the number of velocity bins, or alternatively accumulating the
    # display channels directly during ray integration instead of summarizing the
    # finished histogram afterward.
    brightness_reference = one_bin_image
    positive_intensity = brightness_reference[brightness_reference > 0.0]
    value = np.zeros_like(brightness_reference, dtype=float)
    if positive_intensity.size:
        v_lo = float(np.percentile(positive_intensity, 1.0))
        v_hi = float(np.max(positive_intensity))
        if v_hi > v_lo:
            value = np.clip(
                (np.log10(np.maximum(brightness_reference, v_lo)) - np.log10(v_lo))
                / (np.log10(v_hi) - np.log10(v_lo)),
                0.0,
                1.0,
            )
        else:
            value = np.where(brightness_reference > 0.0, 1.0, 0.0)
    velocity_scale = np.zeros_like(mean_velocity, dtype=float)
    if positive_intensity.size:
        velocity_scale = np.clip(mean_velocity / TOY_LINE_VELOCITY_LIMIT_KMS, -1.0, 1.0)
    velocity_display_kms = TOY_LINE_VELOCITY_LIMIT_KMS * velocity_scale
    gray = value[..., None]
    summary_gray = np.clip(gray**0.8, 0.0, 1.0)
    tint_cmap = cc.cm["CET_I1"]
    tint = tint_cmap(0.5 * (velocity_scale + 1.0))[..., :3]
    tint_lab = cspace_convert(tint, "sRGB1", "CIELab")
    summary_lab = tint_lab.copy()
    summary_lab[..., 0] = 100.0 * summary_gray[..., 0]
    rgb_image = np.clip(cspace_convert(summary_lab, "CIELab", "sRGB1"), 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(6.8, 6.0), constrained_layout=True)
    im = ax.imshow(value, origin="lower", extent=extent, cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title("Pure intensity (value)")
    ax.set_xlabel("X [R]")
    ax.set_ylabel("Z [R]")
    pure_intensity_cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    pure_intensity_cbar.set_label(r"Display brightness [dimensionless]")
    intensity_png_path = tmp_path / "sc_weighted_line_pure_intensity.png"
    fig.savefig(intensity_png_path, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 6.0), constrained_layout=True)
    im = ax.imshow(
        velocity_display_kms,
        origin="lower",
        extent=extent,
        vmin=-TOY_LINE_VELOCITY_LIMIT_KMS,
        vmax=TOY_LINE_VELOCITY_LIMIT_KMS,
        cmap=tint_cmap,
    )
    ax.set_title("Pure tint")
    ax.set_xlabel("X [R]")
    ax.set_ylabel("Z [R]")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Mean LOS velocity [km/s]")
    cbar.ax.yaxis.set_ticks(velocity_edges[1:-1], minor=True)
    cbar.ax.tick_params(which="minor", length=2.5)
    tint_png_path = tmp_path / "sc_weighted_line_pure_tint.png"
    fig.savefig(tint_png_path, dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(7.8, 6.0), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[0.065, 1.0, 0.065], wspace=0.12)
    cax_left = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1])
    cax_right = fig.add_subplot(gs[0, 2])
    ax.imshow(rgb_image, origin="lower", extent=extent)
    ax.set_title("Weighted line summary image")
    ax.set_xlabel("X [R]")
    ax.set_ylabel("Z [R]")
    brightness_cbar = fig.colorbar(
        ScalarMappable(norm=LogNorm(vmin=one_bin_vmin, vmax=one_bin_vmax), cmap="gray"),
        cax=cax_left,
    )
    brightness_cbar.set_label(r"Weighted line intensity [W m$^{-2}$ sr$^{-1}$]")
    cax_left.yaxis.set_ticks_position("left")
    cax_left.yaxis.set_label_position("left")
    tint_cbar = fig.colorbar(
        ScalarMappable(
            norm=Normalize(vmin=-TOY_LINE_VELOCITY_LIMIT_KMS, vmax=TOY_LINE_VELOCITY_LIMIT_KMS),
            cmap=tint_cmap,
        ),
        cax=cax_right,
    )
    tint_cbar.set_label("Mean LOS velocity [km/s]")
    tint_cbar.ax.yaxis.set_ticks(velocity_edges[1:-1], minor=True)
    tint_cbar.ax.tick_params(which="minor", length=2.5)
    summary_png_path = tmp_path / "sc_weighted_line_summary.png"
    fig.savefig(summary_png_path, dpi=200)
    plt.close(fig)

    assert intensity_png_path.exists()
    assert tint_png_path.exists()
    assert summary_png_path.exists()
    assert cube_artifact_path.exists()
