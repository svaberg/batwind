from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import pyvista as pv
from batwind.smart_ds import SmartDs

from batwind.pyvista import (
    alfven_surface_averages,
    build_alfven_surface,
    build_current_sheet_surface,
    current_sheet_orientation,
    plot_alfven_surface,
    plot_current_sheet_surface,
    plot_pyvista_viewport,
)
from batwind.pyvista.isosurfaces import alfven_surface_radius_map
from test.pyvista_test_support import make_structured_smart_ds, scalar_bar_actor, scalar_mesh_actor


EXAMPLE_PLT = Path("examples/3d__var_1_n00000000.plt")
_MU0 = 4.0e-7 * np.pi


def _make_mhd_smart_ds(field_fn, *, n: int = 20, extent: float = 1.5):
    xs = np.linspace(-extent, extent, n)
    ys = np.linspace(-extent, extent, n)
    zs = np.linspace(-extent, extent, n)
    x, y, z = np.meshgrid(xs, ys, zs, indexing="ij")
    rho_cgs, ux_kms, uy_kms, uz_kms, bx_gauss, by_gauss, bz_gauss = field_fn(x, y, z)
    return make_structured_smart_ds(
        [
            x.ravel(),
            y.ravel(),
            z.ravel(),
            rho_cgs.ravel(),
            ux_kms.ravel(),
            uy_kms.ravel(),
            uz_kms.ravel(),
            bx_gauss.ravel(),
            by_gauss.ravel(),
            bz_gauss.ravel(),
        ],
        n=n,
        aux={"RBODY": 1.0},
        title="synthetic-mhd",
        variables=[
            "X [R]",
            "Y [R]",
            "Z [R]",
            "Rho [g/cm^3]",
            "U_x [km/s]",
            "U_y [km/s]",
            "U_z [km/s]",
            "B_x [Gauss]",
            "B_y [Gauss]",
            "B_z [Gauss]",
        ],
        zone="synthetic",
    )


def test_build_alfven_surface_on_synthetic_unit_sphere():
    density_si = 1.0e-9
    magnetic_scale_si = 1.0e-4
    wind_speed_si = magnetic_scale_si / np.sqrt(_MU0 * density_si)
    smart_ds = _make_mhd_smart_ds(
        lambda x, y, z: (
            np.full_like(x, density_si / 1e3),
            np.full_like(x, wind_speed_si / 1e3),
            np.zeros_like(x),
            np.zeros_like(x),
            x,
            y,
            z,
        )
    )

    _grid, surface = build_alfven_surface(smart_ds)
    radii = np.linalg.norm(np.asarray(surface.points, dtype=float), axis=1)
    mach = np.asarray(surface.point_data["M_A [none]"], dtype=float)

    assert surface.n_points > 0
    assert surface.n_cells > 0
    assert surface.active_scalars_name == "U [m/s]"
    assert np.nanmax(np.abs(radii - 1.0)) < 0.01
    assert np.nanmax(np.abs(mach - 1.0)) < 1e-6


def test_alfven_surface_averages_match_synthetic_projected_surface():
    density_si = 1.0e-9
    magnetic_scale_si = 1.0e-4
    speed_scale_si = magnetic_scale_si / np.sqrt(_MU0 * density_si)
    amplitude = 0.25

    smart_ds = _make_mhd_smart_ds(
        lambda x, y, z: _synthetic_alfven_average_field(
            x,
            y,
            z,
            density_si=density_si,
            speed_scale_si=speed_scale_si,
            amplitude=amplitude,
        ),
        n=31,
        extent=2.0,
    )

    average_radius, average_cyl_radius = alfven_surface_averages(smart_ds)

    assert average_radius == pytest.approx(1.0 + amplitude * np.pi / 4.0, abs=0.03)
    assert average_cyl_radius == pytest.approx(np.pi / 4.0 + amplitude * (2.0 / 3.0), abs=0.03)


def test_alfven_surface_radius_map_matches_synthetic_surface():
    density_si = 1.0e-9
    magnetic_scale_si = 1.0e-4
    speed_scale_si = magnetic_scale_si / np.sqrt(_MU0 * density_si)
    amplitude = 0.25

    smart_ds = _make_mhd_smart_ds(
        lambda x, y, z: _synthetic_alfven_average_field(
            x,
            y,
            z,
            density_si=density_si,
            speed_scale_si=speed_scale_si,
            amplitude=amplitude,
        ),
        n=31,
        extent=2.0,
    )

    out = alfven_surface_radius_map(smart_ds, n_polar=18, n_azimuth=36)

    polar = np.asarray(out["polar [rad]"], dtype=float)
    azimuth = np.asarray(out["azimuth [rad]"], dtype=float)
    solid_angle = np.asarray(out["cell_solid_angle [sr]"], dtype=float)
    radius_map = np.asarray(out["alfven_radius [R]"], dtype=float)
    expected = 1.0 + amplitude * np.sin(polar)

    assert polar.shape == (18, 36)
    assert azimuth.shape == polar.shape
    assert solid_angle.shape == polar.shape
    assert radius_map.shape == polar.shape
    assert np.isclose(np.sum(solid_angle), 4.0 * np.pi)
    assert np.all(np.isfinite(radius_map))
    assert np.nanmax(np.abs(radius_map - expected)) < 0.03


def test_build_current_sheet_surface_on_synthetic_midplane():
    smart_ds = _make_mhd_smart_ds(
        lambda x, y, z: (
            np.full_like(x, 1.0e-12),
            np.full_like(x, 300.0),
            np.zeros_like(x),
            np.zeros_like(x),
            np.zeros_like(x),
            np.ones_like(x),
            np.zeros_like(x),
        )
    )

    grid, surface = build_current_sheet_surface(smart_ds)
    radial = np.asarray(grid.point_data["B_r [T]"]).ravel()
    surface_br = np.asarray(surface.point_data["B_r [T]"]).ravel()

    assert surface.n_points > 0
    assert surface.n_cells > 0
    assert surface.active_scalars_name == "U [m/s]"
    assert radial.min() < 0.0 < radial.max()
    np.testing.assert_allclose(np.asarray(surface.points[:, 1], dtype=float), 0.0, atol=1e-12)
    assert np.nanmax(np.abs(surface_br)) < 1e-10


def test_current_sheet_orientation_matches_tilted_synthetic_plane():
    angle_deg = 35.0
    angle = np.deg2rad(angle_deg)
    normal = np.array([np.sin(angle), 0.0, np.cos(angle)], dtype=float)
    smart_ds = _make_mhd_smart_ds(
        lambda x, y, z: (
            np.full_like(x, 1.0e-12),
            np.full_like(x, 300.0),
            np.zeros_like(x),
            np.zeros_like(x),
            np.full_like(x, normal[0]),
            np.full_like(x, normal[1]),
            np.full_like(x, normal[2]),
        ),
        n=21,
    )

    inclination = current_sheet_orientation(smart_ds, rmin=0.2, rmax=1.4, max_points=10_000)

    assert inclination == pytest.approx(angle_deg, abs=0.05)


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_alfven_surface_radius_map_from_example_file_is_bounded():
    smart_ds = SmartDs.from_file(str(EXAMPLE_PLT), batsrus=True, spherical=True)
    average_radius, _average_cyl_radius = alfven_surface_averages(smart_ds)
    out = alfven_surface_radius_map(smart_ds, n_polar=12, n_azimuth=24)
    radius_map = np.asarray(out["alfven_radius [R]"], dtype=float)
    finite = radius_map[np.isfinite(radius_map)]

    assert radius_map.shape == (12, 24)
    assert finite.size > 0
    assert np.all(finite > 0.0)
    assert np.nanmin(finite) <= average_radius <= np.nanmax(finite)


def _synthetic_alfven_average_field(x, y, z, *, density_si: float, speed_scale_si: float, amplitude: float):
    r = np.sqrt(x * x + y * y + z * z)
    cyl = np.sqrt(x * x + y * y)
    sin_theta = np.divide(cyl, r, out=np.zeros_like(r), where=r > 0.0)
    target_radius = 1.0 + amplitude * sin_theta
    speed_si = speed_scale_si * target_radius
    return (
        np.full_like(x, density_si / 1e3),
        speed_si / 1e3,
        np.zeros_like(x),
        np.zeros_like(x),
        x,
        y,
        z,
    )


def _project_points(plotter, points: np.ndarray) -> np.ndarray:
    _forward, right, up = _camera_basis(plotter)
    points = np.asarray(points, dtype=float)
    return np.column_stack((points @ right, points @ up))


def _camera_depths(plotter, points: np.ndarray) -> np.ndarray:
    position = np.asarray(plotter.camera.position, dtype=float)
    forward = _camera_forward(plotter)
    points = np.asarray(points, dtype=float)
    return (points - position) @ forward


def _camera_forward(plotter) -> np.ndarray:
    position = np.asarray(plotter.camera.position, dtype=float)
    focal_point = np.asarray(plotter.camera.focal_point, dtype=float)
    forward = focal_point - position
    return forward / np.linalg.norm(forward)


def _camera_basis(plotter) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = _camera_forward(plotter)
    view_up = np.asarray(plotter.camera.up, dtype=float)
    right = np.cross(forward, view_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return forward, right, up


def _set_oblique_z_up_camera(plotter, radius: float) -> None:
    direction = np.array([2.4, -1.7, 1.1], dtype=float)
    direction /= np.linalg.norm(direction)
    plotter.camera.position = tuple(radius * direction)
    plotter.camera.focal_point = (0.0, 0.0, 0.0)
    plotter.camera.up = (0.0, 0.0, 1.0)
    plotter.reset_camera_clipping_range()


def _push_slice_behind_foreground(plotter, background_slice, foreground_points: np.ndarray, *, margin: float):
    foreground_max_depth = float(np.max(_camera_depths(plotter, foreground_points)))
    pushed_slice = _view_aligned_midplane_slice(
        plotter,
        background_slice,
        depth=foreground_max_depth + float(margin),
    )
    return pushed_slice, foreground_max_depth


def _view_aligned_midplane_slice(plotter, midplane_slice, *, depth: float):
    points = np.asarray(midplane_slice.points, dtype=float)
    x = points[:, 0]
    z = points[:, 2]
    origin_depth = float(_camera_depths(plotter, np.array([[0.0, 0.0, 0.0]], dtype=float))[0])
    forward, right, up = _camera_basis(plotter)
    depth_offset = float(depth) - origin_depth

    billboard = midplane_slice.copy(deep=True)
    billboard.points = (
        x[:, None] * right[None, :]
        + z[:, None] * up[None, :]
        + depth_offset * forward[None, :]
    )
    return billboard


def _render_slice_image(
    slice_mesh,
    *,
    alfven_radius: float,
    u_clim: tuple[float, float],
    screenshot,
    camera_mode: str,
) -> np.ndarray:
    plotter = pv.Plotter(off_screen=True, window_size=(1200, 1200))
    try:
        plotter.add_mesh(
            slice_mesh,
            scalars="U [m/s]",
            cmap="viridis",
            clim=u_clim,
            lighting=False,
            opacity=0.95,
            show_scalar_bar=False,
        )
        if camera_mode == "xz":
            plotter.view_xz()
            plotter.camera.up = (0.0, 0.0, 1.0)
            plotter.camera.focal_point = (0.0, 0.0, 0.0)
            plotter.reset_camera_clipping_range()
        elif camera_mode == "oblique":
            _set_oblique_z_up_camera(plotter, radius=3.2 * alfven_radius)
        else:
            raise ValueError(f"Unsupported camera_mode '{camera_mode}'")
        plotter.enable_parallel_projection()
        plotter.camera.parallel_scale = 1.05 * alfven_radius
        plotter.reset_camera_clipping_range()
        plotter.render()
        image = plotter.screenshot(return_img=True)
        plt.imsave(screenshot, image)
    finally:
        plotter.close()
    return image


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_alfven_surface_averages_run_on_example_file():
    smart_ds = SmartDs.from_file(str(EXAMPLE_PLT), batsrus=True, spherical=True)
    average_radius, average_cyl_radius = alfven_surface_averages(smart_ds)
    radius_map = np.asarray(alfven_surface_radius_map(smart_ds, n_polar=12, n_azimuth=24)["alfven_radius [R]"], dtype=float)
    finite = radius_map[np.isfinite(radius_map)]

    assert np.isfinite(average_radius)
    assert np.isfinite(average_cyl_radius)
    assert finite.size > 0
    assert np.nanmin(finite) > 0.0
    assert average_radius >= np.nanmin(finite)
    assert average_radius <= np.nanmax(finite)
    assert average_cyl_radius >= 0.0
    assert average_cyl_radius <= average_radius


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_build_alfven_surface_from_example_file():
    smart_ds = SmartDs.from_file(str(EXAMPLE_PLT), batsrus=True, spherical=True)
    grid, surface = build_alfven_surface(smart_ds)

    assert grid.n_points > 0
    assert surface.n_points > 0
    assert surface.n_cells > 0

    assert "M_A [none]" in grid.point_data
    assert "U [m/s]" in grid.point_data
    assert "U [m/s]" in surface.point_data
    assert "Normals" in surface.point_data
    assert surface.active_scalars_name == "U [m/s]"

    mach = np.asarray(grid.point_data["M_A [none]"]).ravel()
    finite = mach[np.isfinite(mach)]
    assert finite.size > 0
    assert finite.min() < 1.0 < finite.max()
    surface_mach = np.asarray(surface.point_data["M_A [none]"]).ravel()
    assert np.nanmax(np.abs(surface_mach - 1.0)) < 1e-6


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_build_current_sheet_surface_from_example_file():
    smart_ds = SmartDs.from_file(str(EXAMPLE_PLT), batsrus=True, spherical=True)
    grid, surface = build_current_sheet_surface(smart_ds)

    assert grid.n_points > 0
    assert surface.n_points > 0
    assert surface.n_cells > 0

    assert "B_r [T]" in grid.point_data
    assert "B_r [T]" in surface.point_data
    assert "U [m/s]" in surface.point_data
    assert "Normals" in surface.point_data
    assert surface.active_scalars_name == "U [m/s]"

    radial = np.asarray(grid.point_data["B_r [T]"]).ravel()
    finite = radial[np.isfinite(radial)]
    assert finite.size > 0
    assert finite.min() < 0.0 < finite.max()

    surface_br = np.asarray(surface.point_data["B_r [T]"]).ravel()
    assert np.nanmax(np.abs(surface_br)) < 1e-8


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_current_sheet_orientation_runs_on_example_file():
    smart_ds = SmartDs.from_file(str(EXAMPLE_PLT), batsrus=True, spherical=True)
    inclination = current_sheet_orientation(smart_ds, rmax=30.0)

    assert inclination >= 0.0
    assert inclination <= 90.0


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_plot_alfven_surface_off_screen_writes_screenshot(tmp_path):
    screenshot = tmp_path / "alfven-surface.png"
    viewport_screenshot = tmp_path / "alfven-surface-viewport.png"
    u_vmin = 0.0
    u_vmax = 5.0e5

    fig = None
    smart_ds = SmartDs.from_file(str(EXAMPLE_PLT), batsrus=True, spherical=True)
    plotter, surface = plot_alfven_surface(
        smart_ds,
        vmin=u_vmin,
        vmax=u_vmax,
        off_screen=True,
        screenshot=screenshot,
    )
    try:
        view_radius = 1.05 * float(np.linalg.norm(np.asarray(surface.points, dtype=float), axis=1).max())
        scalar_bar = scalar_bar_actor(plotter)
        colorbar_actor = scalar_mesh_actor(plotter)
        fig, ax = plt.subplots(figsize=(7.0, 7.0), dpi=180, constrained_layout=True)
        fig, ax, colorbar, image = plot_pyvista_viewport(
            plotter,
            fig=fig,
            ax=ax,
            colorbar_actor=colorbar_actor,
            colorbar_label="U [m/s]",
            view="isometric",
            view_center=(0.0, 0.0, 0.0),
            parallel_scale=view_radius,
        )
        ax.grid(True, color="0.88", linewidth=0.6)
        ax.axhline(0.0, color="0.82", linewidth=0.8)
        ax.axvline(0.0, color="0.82", linewidth=0.8)
        fig.savefig(viewport_screenshot, dpi=180)

        assert surface.n_points > 0
        assert screenshot.exists()
        assert screenshot.stat().st_size > 0
        assert viewport_screenshot.exists()
        assert viewport_screenshot.stat().st_size > 0
        assert scalar_bar.GetTitleTextProperty().GetFontSize() >= 36
        assert scalar_bar.GetLabelTextProperty().GetFontSize() >= 30
        assert tuple(colorbar_actor.mapper.scalar_range) == pytest.approx((u_vmin, u_vmax))
        assert colorbar is not None
        assert colorbar.ax.yaxis.label.get_text() == "U [m/s]"
        assert colorbar.mappable.norm.vmin == pytest.approx(u_vmin)
        assert colorbar.mappable.norm.vmax == pytest.approx(u_vmax)
        assert ax.xaxis.get_minorticklocs().size > 0
        assert ax.yaxis.get_minorticklocs().size > 0
        assert colorbar.ax.yaxis.get_minorticklocs().size > 0
        assert np.any(np.any(image[:, :, :3] != image[0, 0, :3], axis=2))
    finally:
        if fig is not None:
            plt.close(fig)
        plotter.close()


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_plot_current_sheet_surface_off_screen_writes_screenshot(tmp_path):
    screenshot = tmp_path / "current-sheet.png"
    viewport_screenshot = tmp_path / "current-sheet-viewport.png"

    fig = None
    smart_ds = SmartDs.from_file(str(EXAMPLE_PLT), batsrus=True, spherical=True)
    plotter, surface = plot_current_sheet_surface(
        smart_ds,
        off_screen=True,
        screenshot=screenshot,
    )
    try:
        view_radius = 1.05 * float(np.linalg.norm(np.asarray(surface.points, dtype=float), axis=1).max())
        scalar_bar = scalar_bar_actor(plotter)
        colorbar_actor = scalar_mesh_actor(plotter)
        fig, ax = plt.subplots(figsize=(7.0, 7.0), dpi=180, constrained_layout=True)
        fig, ax, colorbar, image = plot_pyvista_viewport(
            plotter,
            fig=fig,
            ax=ax,
            colorbar_actor=colorbar_actor,
            colorbar_label="U [m/s]",
            view="isometric",
            view_center=(0.0, 0.0, 0.0),
            parallel_scale=view_radius,
        )
        ax.grid(True, color="0.88", linewidth=0.6)
        ax.axhline(0.0, color="0.82", linewidth=0.8)
        ax.axvline(0.0, color="0.82", linewidth=0.8)
        fig.savefig(viewport_screenshot, dpi=180)

        assert surface.n_points > 0
        assert screenshot.exists()
        assert screenshot.stat().st_size > 0
        assert viewport_screenshot.exists()
        assert viewport_screenshot.stat().st_size > 0
        assert scalar_bar.GetTitleTextProperty().GetFontSize() >= 36
        assert scalar_bar.GetLabelTextProperty().GetFontSize() >= 30
        assert colorbar is not None
        assert colorbar.ax.yaxis.label.get_text() == "U [m/s]"
        assert ax.xaxis.get_minorticklocs().size > 0
        assert ax.yaxis.get_minorticklocs().size > 0
        assert colorbar.ax.yaxis.get_minorticklocs().size > 0
        assert np.any(np.any(image[:, :, :3] != image[0, 0, :3], axis=2))
    finally:
        if fig is not None:
            plt.close(fig)
        plotter.close()


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_pushed_back_midplane_slice_matches_midplane_slice_render(tmp_path):
    midplane_screenshot = tmp_path / "midplane-slice.png"
    pushed_screenshot = tmp_path / "midplane-slice-pushed-back.png"
    u_clim = (0.0, 5.0e5)

    smart_ds = SmartDs.from_file(str(EXAMPLE_PLT), batsrus=True, spherical=True)
    grid, alfven_surface = build_alfven_surface(smart_ds)
    _grid, current_sheet = build_current_sheet_surface(smart_ds)

    alfven_radius = float(np.linalg.norm(np.asarray(alfven_surface.points, dtype=float), axis=1).max())
    current_sheet_limit = 1.02 * alfven_radius
    clipped_current_sheet = current_sheet.clip_surface(pv.Sphere(radius=current_sheet_limit), invert=True)
    background_slice = grid.slice(normal=(0.0, 1.0, 0.0), origin=(0.0, 0.0, 0.0))

    foreground_points = np.vstack(
        [
            np.asarray(alfven_surface.points, dtype=float),
            np.asarray(clipped_current_sheet.points, dtype=float),
        ]
    )
    probe_plotter = pv.Plotter(off_screen=True)
    try:
        _set_oblique_z_up_camera(probe_plotter, radius=3.2 * alfven_radius)
        pushed_slice, foreground_max_depth = _push_slice_behind_foreground(
            probe_plotter,
            background_slice,
            foreground_points,
            margin=0.08 * alfven_radius,
        )
        pushed_min_depth = float(np.min(_camera_depths(probe_plotter, np.asarray(pushed_slice.points, dtype=float))))
        pushed_depth_spread = float(np.ptp(_camera_depths(probe_plotter, np.asarray(pushed_slice.points, dtype=float))))
    finally:
        probe_plotter.close()

    midplane_image = _render_slice_image(
        background_slice,
        alfven_radius=alfven_radius,
        u_clim=u_clim,
        screenshot=midplane_screenshot,
        camera_mode="xz",
    )
    pushed_image = _render_slice_image(
        pushed_slice,
        alfven_radius=alfven_radius,
        u_clim=u_clim,
        screenshot=pushed_screenshot,
        camera_mode="oblique",
    )

    assert background_slice.n_points > 0
    assert pushed_slice.n_points == background_slice.n_points
    assert pushed_min_depth > foreground_max_depth + 0.05 * alfven_radius
    assert pushed_depth_spread < 1e-6
    assert midplane_screenshot.exists()
    assert midplane_screenshot.stat().st_size > 0
    assert pushed_screenshot.exists()
    assert pushed_screenshot.stat().st_size > 0
    image_diff = np.abs(midplane_image.astype(int) - pushed_image.astype(int))
    assert image_diff.max() <= 1
    assert np.count_nonzero(np.any(image_diff != 0, axis=2)) <= 2


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_plot_alfven_surface_with_current_sheet_off_screen_writes_screenshot(tmp_path):
    screenshot = tmp_path / "alfven-current-sheet.png"
    viewport_screenshot = tmp_path / "alfven-current-sheet-viewport.png"
    u_vmin = 0.0
    u_vmax = 5.0e5

    smart_ds = SmartDs.from_file(str(EXAMPLE_PLT), batsrus=True, spherical=True)
    grid, alfven_surface = build_alfven_surface(smart_ds)
    _grid, current_sheet = build_current_sheet_surface(smart_ds)

    alfven_radius = float(np.linalg.norm(np.asarray(alfven_surface.points, dtype=float), axis=1).max())
    current_sheet_limit = 1.02 * alfven_radius
    clipped_current_sheet = current_sheet.clip_surface(pv.Sphere(radius=current_sheet_limit), invert=True)
    background_slice = grid.slice(normal=(0.0, 1.0, 0.0), origin=(0.0, 0.0, 0.0))
    u_clim = (u_vmin, u_vmax)

    fig = None
    plotter = pv.Plotter(off_screen=True)
    try:
        _set_oblique_z_up_camera(plotter, radius=3.2 * alfven_radius)
        foreground_points = np.vstack(
            [
                np.asarray(alfven_surface.points, dtype=float),
                np.asarray(clipped_current_sheet.points, dtype=float),
            ]
        )
        background_slice, foreground_max_depth = _push_slice_behind_foreground(
            plotter,
            background_slice,
            foreground_points,
            margin=0.08 * alfven_radius,
        )
        slice_actor = plotter.add_mesh(
            background_slice,
            scalars="U [m/s]",
            cmap="viridis",
            clim=u_clim,
            lighting=False,
            opacity=0.95,
            show_scalar_bar=False,
        )
        current_sheet_actor = plotter.add_mesh(
            clipped_current_sheet,
            color=(0.65, 0.65, 0.65),
            opacity=0.35,
            smooth_shading=True,
            show_scalar_bar=False,
        )
        alfven_actor = plotter.add_mesh(
            alfven_surface,
            scalars="U [m/s]",
            cmap="viridis",
            clim=u_clim,
            smooth_shading=True,
            show_scalar_bar=False,
        )
        plotter.add_title("Alfven surface + current sheet")
        _set_oblique_z_up_camera(plotter, radius=3.2 * alfven_radius)
        plotter.show(screenshot=str(screenshot), auto_close=False)

        fig, ax = plt.subplots(figsize=(7.2, 7.0), dpi=180, constrained_layout=True)
        fig, ax, colorbar, image = plot_pyvista_viewport(
            plotter,
            fig=fig,
            ax=ax,
            colorbar_actor=alfven_actor,
            colorbar_label="U [m/s]",
            view=None,
            axis_labels=("view x [R]", "Z [R]"),
            view_center=(0.0, 0.0, 0.0),
            parallel_scale=1.05 * alfven_radius,
            render_size=(1200, 1200),
        )
        ax.grid(True, color="0.88", linewidth=0.6)
        ax.axhline(0.0, color="0.82", linewidth=0.8)
        ax.axvline(0.0, color="0.82", linewidth=0.8)
        fig.savefig(viewport_screenshot, dpi=180)

        projected_z = _project_points(plotter, np.array([[0.0, 0.0, 1.0]], dtype=float))[0]
        projected_y = _project_points(plotter, np.array([[0.0, 1.0, 0.0]], dtype=float))[0]
        current_sheet_radii = np.linalg.norm(np.asarray(clipped_current_sheet.points, dtype=float), axis=1)
        slice_min_depth = float(np.min(_camera_depths(plotter, np.asarray(background_slice.points, dtype=float))))

        assert background_slice.n_points > 0
        assert tuple(slice_actor.mapper.scalar_range) == pytest.approx(u_clim)
        assert tuple(alfven_actor.mapper.scalar_range) == pytest.approx(u_clim)
        assert slice_min_depth > foreground_max_depth + 0.05 * alfven_radius
        assert clipped_current_sheet.n_points > 0
        assert clipped_current_sheet.n_cells > 0
        assert np.max(current_sheet_radii) <= current_sheet_limit + 1e-6
        assert current_sheet_actor.prop.opacity == pytest.approx(0.35)
        np.testing.assert_allclose(current_sheet_actor.prop.color.float_rgb, (0.65, 0.65, 0.65), atol=2e-3)
        assert np.linalg.norm(np.asarray(plotter.camera.position, dtype=float)) > 0.0
        assert abs(projected_z[0]) < 1e-6
        assert projected_z[1] > 0.0
        assert abs(projected_y[0]) > 0.05
        assert not plotter.scalar_bars
        assert colorbar is not None
        assert colorbar.ax.yaxis.label.get_text() == "U [m/s]"
        assert colorbar.mappable.norm.vmin == pytest.approx(u_vmin)
        assert colorbar.mappable.norm.vmax == pytest.approx(u_vmax)
        assert ax.get_ylabel() == "Z [R]"
        assert ax.xaxis.get_minorticklocs().size > 0
        assert ax.yaxis.get_minorticklocs().size > 0
        assert colorbar.ax.yaxis.get_minorticklocs().size > 0
        assert screenshot.exists()
        assert screenshot.stat().st_size > 0
        assert viewport_screenshot.exists()
        assert viewport_screenshot.stat().st_size > 0
        assert np.any(np.any(image[:, :, :3] != image[0, 0, :3], axis=2))
    finally:
        if fig is not None:
            plt.close(fig)
        plotter.close()
