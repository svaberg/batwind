from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.colors import ListedColormap, Normalize


def plot_pyvista_viewport(
    plotter: pv.Plotter,
    *,
    colorbar_actor=None,
    colorbar_label: str | None = None,
    overlay_text: str | None = None,
    view: str | None = "isometric",
    axis_labels: tuple[str, str] | None = None,
    view_center=(0.0, 0.0, 0.0),
    parallel_scale: float | None = None,
    render_size: tuple[int, int] | None = None,
    fig=None,
    ax=None,
):
    if ax is None and fig is None:
        fig, ax = plt.subplots(figsize=(6.5, 6.0), constrained_layout=True)
    elif ax is None:
        ax = fig.add_subplot(111)
    elif fig is None:
        fig = ax.figure

    if view is None:
        x_label, y_label = axis_labels or ("view x [R]", "view y [R]")
    else:
        x_label, y_label = _apply_view(plotter, view)
        if axis_labels is not None:
            x_label, y_label = axis_labels
    _set_view_center(plotter, view_center)
    plotter.enable_parallel_projection()
    if parallel_scale is not None:
        plotter.camera.parallel_scale = float(parallel_scale)
        plotter.reset_camera_clipping_range()
    plotter.window_size = _resolve_render_size(fig, ax, render_size)
    plotter.render()
    image = plotter.screenshot(return_img=True)
    extent = _parallel_projection_extent(
        plotter,
        center=_view_plane_center(view, view_center),
    )

    ax.imshow(np.flipud(image), origin="lower", extent=extent)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if overlay_text is not None:
        ax.text(0.02, 0.98, overlay_text, transform=ax.transAxes, ha="left", va="top")

    colorbar = None
    if colorbar_actor is not None:
        colorbar = fig.colorbar(_actor_scalar_mappable(colorbar_actor), ax=ax, pad=0.02)
        if colorbar_label is not None:
            colorbar.set_label(colorbar_label)

    return fig, ax, colorbar, image


def apply_matplotlib_color_source(
    actor,
    color_source: ScalarMappable | Colorbar,
    *,
    n_colors: int = 256,
):
    mappable = color_source.mappable if isinstance(color_source, Colorbar) else color_source
    scalar_range = _mappable_scalar_range(mappable)
    samples = np.linspace(scalar_range[0], scalar_range[1], int(n_colors))
    colors = np.asarray(mappable.to_rgba(samples, bytes=True), dtype=np.uint8)

    lut = actor.mapper.lookup_table
    lut.values = colors
    lut.scalar_range = scalar_range
    actor.mapper.scalar_range = scalar_range
    return actor


def _actor_scalar_mappable(actor) -> ScalarMappable:
    lut = actor.mapper.lookup_table
    colors = np.asarray(lut.values, dtype=float)[:, :4] / 255.0
    cmap = ListedColormap(colors)
    norm = Normalize(*lut.scalar_range)
    return ScalarMappable(norm=norm, cmap=cmap)


def _mappable_scalar_range(mappable: ScalarMappable) -> tuple[float, float]:
    vmin = getattr(mappable.norm, "vmin", None)
    vmax = getattr(mappable.norm, "vmax", None)
    if vmin is None or vmax is None:
        clim = mappable.get_clim()
        vmin, vmax = clim
    if vmin is None or vmax is None:
        raise ValueError("Matplotlib color source must define a finite scalar range")
    return float(vmin), float(vmax)


def _resolve_render_size(fig, ax, render_size) -> tuple[int, int]:
    if render_size is not None:
        width, height = render_size
        return max(1, int(width)), max(1, int(height))
    return _axes_pixel_size(fig, ax)


def _axes_pixel_size(fig, ax) -> tuple[int, int]:
    # Match the PyVista off-screen image to the Matplotlib axes footprint.
    fig.canvas.draw()
    bbox = ax.get_window_extent()
    return max(1, int(round(bbox.width))), max(1, int(round(bbox.height)))


def _apply_view(plotter: pv.Plotter, view: str) -> tuple[str, str]:
    if view == "xy":
        plotter.view_xy()
        return "X [R]", "Y [R]"
    if view == "xz":
        plotter.view_xz()
        return "X [R]", "Z [R]"
    if view == "yz":
        plotter.view_yz()
        return "Y [R]", "Z [R]"
    if view == "isometric":
        plotter.view_isometric()
        return "view x [R]", "view y [R]"
    raise ValueError(f"Unsupported viewport view '{view}'")


def _view_plane_center(view: str | None, view_center) -> tuple[float, float]:
    view_center = np.asarray(view_center, dtype=float)
    if view is None:
        return 0.0, 0.0
    if view == "xy":
        return float(view_center[0]), float(view_center[1])
    if view == "xz":
        return float(view_center[0]), float(view_center[2])
    if view == "yz":
        return float(view_center[1]), float(view_center[2])
    if view == "isometric":
        return 0.0, 0.0
    raise ValueError(f"Unsupported viewport view '{view}'")


def _set_view_center(plotter: pv.Plotter, view_center) -> None:
    view_center = np.asarray(view_center, dtype=float)
    focal_point = np.asarray(plotter.camera.focal_point, dtype=float)
    position = np.asarray(plotter.camera.position, dtype=float)
    offset = position - focal_point
    plotter.camera.focal_point = tuple(view_center)
    plotter.camera.position = tuple(view_center + offset)
    plotter.reset_camera_clipping_range()


def _parallel_projection_extent(
    plotter: pv.Plotter,
    *,
    center=(0.0, 0.0),
) -> tuple[float, float, float, float]:
    width, height = plotter.window_size
    y_half = float(plotter.camera.parallel_scale)
    x_half = y_half * float(width) / float(height)
    center = np.asarray(center, dtype=float)
    return (
        float(center[0] - x_half),
        float(center[0] + x_half),
        float(center[1] - y_half),
        float(center[1] + y_half),
    )

__all__ = ["apply_matplotlib_color_source", "plot_pyvista_viewport"]
