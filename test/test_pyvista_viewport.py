import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.colors import ListedColormap, Normalize

from batwind.pyvista import plot_pyvista_viewport
from batwind.pyvista.viewport import apply_matplotlib_color_source


pv.OFF_SCREEN = True
_VIEW_HALF_SIZE = 2.25
_TEST_RENDER_SIZE = (768, 768)


def _new_plotter():
    plotter = pv.Plotter(off_screen=True, window_size=(256, 256))
    plotter.set_background("white")
    return plotter


def _test_l_shape():
    parts = [
        pv.Cube(center=(0.5, 0.0, 0.0), x_length=1.0, y_length=0.18, z_length=0.18),
        pv.Cube(center=(0.0, 0.0, 1.0), x_length=0.18, y_length=0.18, z_length=2.0),
    ]
    mesh = parts[0]
    for part in parts[1:]:
        mesh = mesh.merge(part)
    mesh = mesh.clean()
    mesh["Z [R]"] = np.asarray(mesh.points[:, 2], dtype=float)
    return mesh


def _orange_pixel_count(image: np.ndarray) -> int:
    rgb = image[:, :, :3]
    mask = (rgb[:, :, 0] > 200) & (rgb[:, :, 1] > 80) & (rgb[:, :, 1] < 220) & (rgb[:, :, 2] < 120)
    return int(mask.sum())


def _dominant_color_pixel_count(image: np.ndarray, color: str) -> int:
    rgb = image[:, :, :3]
    if color == "red":
        mask = (rgb[:, :, 0] > 180) & (rgb[:, :, 1] < 90) & (rgb[:, :, 2] < 90)
    elif color == "blue":
        mask = (rgb[:, :, 2] > 180) & (rgb[:, :, 0] < 90) & (rgb[:, :, 1] < 140)
    else:
        raise ValueError(f"Unsupported color '{color}'")
    return int(mask.sum())


def _nonwhite_fraction_in_box(image: np.ndarray, *, xlim, ylim, x0: float, x1: float, y0: float, y1: float) -> float:
    mask = np.any(np.flipud(image)[:, :, :3] != 255, axis=2)
    ny, nx = mask.shape
    x_edges = np.linspace(xlim[0], xlim[1], nx + 1)
    y_edges = np.linspace(ylim[0], ylim[1], ny + 1)
    x_mask = (x_edges[:-1] < x1) & (x_edges[1:] > x0)
    y_mask = (y_edges[:-1] < y1) & (y_edges[1:] > y0)
    box = mask[np.ix_(y_mask, x_mask)]
    return float(box.mean())


def _actor_scalar_mappable(actor) -> ScalarMappable:
    lut = actor.mapper.lookup_table
    colors = np.asarray(lut.values, dtype=float)[:, :4] / 255.0
    return ScalarMappable(norm=Normalize(*lut.scalar_range), cmap=ListedColormap(colors))


def _project_points(plotter: pv.Plotter, points: np.ndarray, *, view_center=(0.0, 0.0, 0.0)) -> np.ndarray:
    position, focal_point, view_up = plotter.camera_position
    position = np.asarray(position, dtype=float)
    focal_point = np.asarray(focal_point, dtype=float)
    view_up = np.asarray(view_up, dtype=float)
    view_center = np.asarray(view_center, dtype=float)

    forward = focal_point - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, view_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    offsets = np.asarray(points, dtype=float) - view_center
    return np.column_stack((offsets @ right, offsets @ up))


def _add_projected_world_axes(
    ax,
    plotter: pv.Plotter,
    *,
    axis_length: float = 1.0,
    anchor_uv: tuple[float, float] = (-1.55, -1.55),
):
    origin = np.zeros((1, 3), dtype=float)
    endpoints = np.array(
        [
            [axis_length, 0.0, 0.0],
            [0.0, axis_length, 0.0],
            [0.0, 0.0, axis_length],
        ],
        dtype=float,
    )
    origin_uv = _project_points(plotter, origin)[0]
    endpoints_uv = _project_points(plotter, endpoints)
    directions_uv = endpoints_uv - origin_uv
    anchor_uv = np.asarray(anchor_uv, dtype=float)
    colors = ("C3", "C2", "C0")
    labels = ("X", "Y", "Z")

    for direction, color, label in zip(directions_uv, colors, labels, strict=True):
        endpoint = anchor_uv + direction
        ax.plot([anchor_uv[0], endpoint[0]], [anchor_uv[1], endpoint[1]], color=color, lw=2.5)
        ax.text(endpoint[0], endpoint[1], label, color=color, ha="center", va="center")


def _assert_axis_alignment(plotter: pv.Plotter, horizontal_axis: np.ndarray, vertical_axis: np.ndarray) -> None:
    projected = _project_points(
        plotter,
        np.vstack([horizontal_axis, vertical_axis]),
        view_center=(0.0, 0.0, 0.0),
    )
    horizontal, vertical = projected
    assert horizontal[0] > 0.0
    assert abs(horizontal[1]) < 1e-6
    assert vertical[1] > 0.0
    assert abs(vertical[0]) < 1e-6


def _format_verification_axes(ax) -> None:
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xticks(_ticks_in_range(*xlim, step=1.0))
    ax.set_yticks(_ticks_in_range(*ylim, step=1.0))
    ax.set_xticks(_ticks_in_range(*xlim, step=0.5), minor=True)
    ax.set_yticks(_ticks_in_range(*ylim, step=0.5), minor=True)
    ax.grid(True, which="major", color="0.82", linewidth=0.8)
    ax.grid(True, which="minor", color="0.9", linewidth=0.5)


def _ticks_in_range(lower: float, upper: float, *, step: float) -> np.ndarray:
    start = np.ceil(lower / step) * step
    stop = np.floor(upper / step) * step
    if start > stop:
        return np.array([], dtype=float)
    count = int(round((stop - start) / step)) + 1
    return start + step * np.arange(count, dtype=float)


def test_plot_pyvista_viewport_is_isometric_and_uses_physical_axes(tmp_path):
    output = tmp_path / "viewport-isometric-l.png"
    plotter = _new_plotter()
    plotter.add_mesh(_test_l_shape(), color="black", lighting=False)

    fig, ax, colorbar, image = plot_pyvista_viewport(
        plotter,
        view_center=(0.0, 0.0, 0.0),
        parallel_scale=_VIEW_HALF_SIZE,
        render_size=_TEST_RENDER_SIZE,
        overlay_text="isometric test",
    )
    try:
        _add_projected_world_axes(ax, plotter, axis_length=0.9)
        fig.savefig(output, dpi=120)

        position, focal_point, _view_up = plotter.camera_position
        view = np.asarray(position, dtype=float) - np.asarray(focal_point, dtype=float)
        direction = np.abs(view / np.linalg.norm(view))
        x_half = plotter.camera.parallel_scale

        assert plotter.camera.parallel_projection
        np.testing.assert_allclose(direction[0], direction[1], atol=1e-6)
        np.testing.assert_allclose(direction[1], direction[2], atol=1e-6)
        np.testing.assert_allclose(ax.get_xlim(), (-x_half, x_half), atol=1e-6)
        np.testing.assert_allclose(ax.get_ylim(), (-x_half, x_half), atol=1e-6)
        assert colorbar is None
        assert image.shape == (_TEST_RENDER_SIZE[1], _TEST_RENDER_SIZE[0], 3)
        assert {text.get_text() for text in ax.texts} >= {"X", "Y", "Z", "isometric test"}
        assert output.exists()
        assert output.stat().st_size > 0
    finally:
        plt.close(fig)
        plotter.close()


def test_plot_pyvista_viewport_renders_plain_l_geometry(tmp_path):
    output = tmp_path / "viewport-plain-l.png"
    plotter = _new_plotter()
    plotter.add_mesh(_test_l_shape(), color=(255, 127, 14), lighting=False)

    fig, ax, colorbar, image = plot_pyvista_viewport(
        plotter,
        view_center=(0.0, 0.0, 0.0),
        parallel_scale=_VIEW_HALF_SIZE,
        render_size=_TEST_RENDER_SIZE,
        overlay_text="plain L",
    )
    try:
        _add_projected_world_axes(ax, plotter, axis_length=0.9)
        fig.savefig(output, dpi=120)
        nonwhite = np.any(image[:, :, :3] != 255, axis=2)

        assert colorbar is None
        assert nonwhite.sum() > 2000
        assert _orange_pixel_count(image) > 1500
        assert ax.texts[0].get_text() == "plain L"
        assert output.exists()
        assert output.stat().st_size > 0
    finally:
        plt.close(fig)
        plotter.close()


def test_plot_pyvista_viewport_uses_pyvista_colormap_on_z(tmp_path):
    output_a = tmp_path / "viewport-z-viridis.png"
    output_b = tmp_path / "viewport-z-plasma.png"
    mesh = _test_l_shape()

    plotter_a = _new_plotter()
    actor_a = plotter_a.add_mesh(
        mesh,
        scalars="Z [R]",
        cmap="viridis",
        clim=(0.0, 2.0),
        lighting=False,
        show_scalar_bar=False,
    )
    plotter_b = _new_plotter()
    actor_b = plotter_b.add_mesh(
        mesh,
        scalars="Z [R]",
        cmap="plasma",
        clim=(0.0, 2.0),
        lighting=False,
        show_scalar_bar=False,
    )

    fig_a, ax_a, colorbar_a, image_a = plot_pyvista_viewport(
        plotter_a,
        colorbar_actor=actor_a,
        colorbar_label="Z [R]",
        view_center=(0.0, 0.0, 0.0),
        parallel_scale=_VIEW_HALF_SIZE,
        render_size=_TEST_RENDER_SIZE,
        overlay_text="viridis",
    )
    fig_b, ax_b, colorbar_b, image_b = plot_pyvista_viewport(
        plotter_b,
        colorbar_actor=actor_b,
        colorbar_label="Z [R]",
        view_center=(0.0, 0.0, 0.0),
        parallel_scale=_VIEW_HALF_SIZE,
        render_size=_TEST_RENDER_SIZE,
        overlay_text="plasma",
    )
    try:
        _add_projected_world_axes(ax_a, plotter_a, axis_length=0.9)
        _add_projected_world_axes(ax_b, plotter_b, axis_length=0.9)
        fig_a.savefig(output_a, dpi=120)
        fig_b.savefig(output_b, dpi=120)

        diff = np.abs(image_a[:, :, :3].astype(int) - image_b[:, :, :3].astype(int)).sum()

        assert colorbar_a is not None
        assert colorbar_b is not None
        assert colorbar_a.ax.yaxis.label.get_text() == "Z [R]"
        assert colorbar_b.ax.yaxis.label.get_text() == "Z [R]"
        assert diff > 250_000
        assert output_a.exists()
        assert output_b.exists()
        assert output_a.stat().st_size > 0
        assert output_b.stat().st_size > 0
    finally:
        plt.close(fig_a)
        plt.close(fig_b)
        plotter_a.close()
        plotter_b.close()


def test_plot_pyvista_viewport_four_views_share_colorbar(tmp_path):
    output = tmp_path / "viewport-four-views.png"
    mesh = _test_l_shape()
    views = [
        ("xy", "XY", np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
        ("xz", "XZ", np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
        ("yz", "YZ", np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
        ("isometric", "Isometric", None, None),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.0), constrained_layout=True)
    plotters = []
    first_actor = None
    try:
        xz_image = None
        xz_xlim = None
        xz_ylim = None
        for ax, (view, title, horizontal_axis, vertical_axis) in zip(axes.ravel(), views, strict=True):
            plotter = _new_plotter()
            actor = plotter.add_mesh(
                mesh,
                scalars="Z [R]",
                cmap="viridis",
                clim=(0.0, 2.0),
                lighting=False,
                show_scalar_bar=False,
            )
            if first_actor is None:
                first_actor = actor

            _fig, _ax, colorbar, image = plot_pyvista_viewport(
                plotter,
                view=view,
                view_center=(0.0, 0.0, 0.0),
                parallel_scale=_VIEW_HALF_SIZE,
                render_size=_TEST_RENDER_SIZE,
                fig=fig,
                ax=ax,
            )
            assert colorbar is None
            assert image.shape == (_TEST_RENDER_SIZE[1], _TEST_RENDER_SIZE[0], 3)
            ax.set_title(title)
            _format_verification_axes(ax)
            ax.axhline(0.0, color="0.75", lw=0.8)
            ax.axvline(0.0, color="0.75", lw=0.8)
            if view == "isometric":
                _add_projected_world_axes(ax, plotter, axis_length=0.9)
            else:
                _assert_axis_alignment(plotter, horizontal_axis, vertical_axis)
            if view == "xz":
                xz_image = image
                xz_xlim = ax.get_xlim()
                xz_ylim = ax.get_ylim()

            plotters.append(plotter)

        colorbar = fig.colorbar(
            _actor_scalar_mappable(first_actor),
            ax=axes.ravel().tolist(),
            pad=0.03,
            shrink=0.92,
        )
        colorbar.set_label("Z [R]")
        fig.savefig(output, dpi=140)

        assert axes[0, 0].get_xlabel() == "X [R]"
        assert axes[0, 0].get_ylabel() == "Y [R]"
        assert axes[0, 1].get_xlabel() == "X [R]"
        assert axes[0, 1].get_ylabel() == "Z [R]"
        assert axes[1, 0].get_xlabel() == "Y [R]"
        assert axes[1, 0].get_ylabel() == "Z [R]"
        assert colorbar.ax.yaxis.label.get_text() == "Z [R]"
        assert _nonwhite_fraction_in_box(xz_image, xlim=xz_xlim, ylim=xz_ylim, x0=0.35, x1=0.95, y0=-0.08, y1=0.12) > 0.35
        assert _nonwhite_fraction_in_box(xz_image, xlim=xz_xlim, ylim=xz_ylim, x0=-0.08, x1=0.12, y0=0.65, y1=1.95) > 0.35
        assert _nonwhite_fraction_in_box(xz_image, xlim=xz_xlim, ylim=xz_ylim, x0=0.35, x1=0.95, y0=0.65, y1=1.95) < 0.08
        assert _nonwhite_fraction_in_box(xz_image, xlim=xz_xlim, ylim=xz_ylim, x0=-0.08, x1=0.12, y0=2.05, y1=2.25) < 0.02
        assert output.exists()
        assert output.stat().st_size > 0
    finally:
        plt.close(fig)
        for plotter in plotters:
            plotter.close()


def test_apply_matplotlib_colorbar_to_pyvista_actor(tmp_path):
    output = tmp_path / "viewport-mpl-colorbar.png"
    mesh = _test_l_shape()

    source_fig, source_ax = plt.subplots(figsize=(1.8, 4.0), constrained_layout=True)
    source_mappable = ScalarMappable(
        norm=PowerNorm(gamma=0.55, vmin=0.0, vmax=2.0),
        cmap=LinearSegmentedColormap.from_list(
            "mpl_demo",
            ["#001219", "#0a9396", "#ee9b00", "#bb3e03", "#9b2226"],
        ),
    )
    source_colorbar = source_fig.colorbar(source_mappable, cax=source_ax)
    source_colorbar.set_label("Z [R]")

    plotter = _new_plotter()
    actor = plotter.add_mesh(mesh, scalars="Z [R]", lighting=False, show_scalar_bar=False)
    apply_matplotlib_color_source(actor, source_colorbar)

    fig, ax, colorbar, image = plot_pyvista_viewport(
        plotter,
        colorbar_actor=actor,
        colorbar_label="Z [R]",
        view="isometric",
        view_center=(0.0, 0.0, 0.0),
        parallel_scale=_VIEW_HALF_SIZE,
        render_size=_TEST_RENDER_SIZE,
        overlay_text="mpl -> pyvista",
    )
    try:
        _format_verification_axes(ax)
        _add_projected_world_axes(ax, plotter, axis_length=0.9)
        fig.savefig(output, dpi=120)

        expected = np.asarray(source_mappable.to_rgba(np.linspace(0.0, 2.0, 256), bytes=True), dtype=np.uint8)
        actual = np.asarray(actor.mapper.lookup_table.values, dtype=np.uint8)

        assert colorbar is not None
        assert colorbar.ax.yaxis.label.get_text() == "Z [R]"
        assert np.array_equal(actual, expected)
        assert tuple(actor.mapper.lookup_table.scalar_range) == (0.0, 2.0)
        assert np.any(image[:, :, :3] != 255)
        assert output.exists()
        assert output.stat().st_size > 0
    finally:
        plt.close(fig)
        plt.close(source_fig)
        plotter.close()


def test_plot_pyvista_viewport_infers_render_size_from_matplotlib_axes(tmp_path):
    output = tmp_path / "viewport-inferred-resolution.png"
    plotter = _new_plotter()
    plotter.add_mesh(_test_l_shape(), color="black", lighting=False)

    fig, ax = plt.subplots(figsize=(6.8, 5.2), dpi=180, constrained_layout=True)
    fig.canvas.draw()
    bbox = ax.get_window_extent()
    expected_width = int(round(bbox.width))
    expected_height = int(round(bbox.height))

    fig, ax, colorbar, image = plot_pyvista_viewport(
        plotter,
        fig=fig,
        ax=ax,
        view="xz",
        view_center=(0.0, 0.0, 0.0),
        parallel_scale=_VIEW_HALF_SIZE,
        overlay_text="dpi inferred",
    )
    try:
        _format_verification_axes(ax)
        fig.savefig(output, dpi=180)

        assert colorbar is None
        assert image.shape == (expected_height, expected_width, 3)
        assert tuple(plotter.window_size) == (expected_width, expected_height)
        assert expected_width > 700
        assert expected_height > 700
        assert output.exists()
        assert output.stat().st_size > 0
    finally:
        plt.close(fig)
        plotter.close()


def test_plot_pyvista_viewport_can_target_nonzero_world_center(tmp_path):
    output = tmp_path / "viewport-offset-center.png"
    target_center = np.array([10.0, 0.0, 5.0], dtype=float)
    plotter = _new_plotter()
    plotter.add_mesh(pv.Cube(center=(0.0, 0.0, 0.0), x_length=0.8, y_length=0.8, z_length=0.8), color="red", lighting=False)
    plotter.add_mesh(
        pv.Cube(center=tuple(target_center), x_length=0.8, y_length=0.8, z_length=0.8),
        color="blue",
        lighting=False,
    )

    fig, ax = plt.subplots(figsize=(6.5, 6.0), constrained_layout=True)
    ax.set_xlim(-100.0, 100.0)
    ax.set_ylim(-50.0, 50.0)

    fig, ax, colorbar, image = plot_pyvista_viewport(
        plotter,
        fig=fig,
        ax=ax,
        view="xz",
        view_center=tuple(target_center),
        parallel_scale=1.25,
        render_size=_TEST_RENDER_SIZE,
        overlay_text="offset target",
    )
    try:
        fig.savefig(output, dpi=120)

        np.testing.assert_allclose(plotter.camera.focal_point, target_center, atol=1e-12)
        np.testing.assert_allclose(ax.get_xlim(), (8.75, 11.25), atol=1e-12)
        np.testing.assert_allclose(ax.get_ylim(), (3.75, 6.25), atol=1e-12)
        assert colorbar is None
        assert _dominant_color_pixel_count(image, "blue") > 15_000
        assert _dominant_color_pixel_count(image, "red") == 0
        assert output.exists()
        assert output.stat().st_size > 0
    finally:
        plt.close(fig)
        plotter.close()
