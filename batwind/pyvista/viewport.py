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
    draft: bool = False,
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

    if render_size is not None:
        width, height = render_size
        plotter.window_size = max(1, int(width)), max(1, int(height))
    else:
        fig.canvas.draw()
        bbox = ax.get_window_extent()
        plotter.window_size = max(1, int(round(bbox.width))), max(1, int(round(bbox.height)))

    if view is None:
        x_label, y_label = axis_labels or ("view x [R]", "view y [R]")
    else:
        if view == "xy":
            plotter.view_xy()
            x_label, y_label = "X [R]", "Y [R]"
        elif view == "xz":
            plotter.view_xz()
            x_label, y_label = "X [R]", "Z [R]"
        elif view == "yz":
            plotter.view_yz()
            x_label, y_label = "Y [R]", "Z [R]"
        elif view == "isometric":
            plotter.view_isometric()
            x_label, y_label = "view x [R]", "view y [R]"
        else:
            raise ValueError(f"Unsupported viewport view '{view}'")
        if axis_labels is not None:
            x_label, y_label = axis_labels
    view_center = np.asarray(view_center, dtype=float)
    focal_point = np.asarray(plotter.camera.focal_point, dtype=float)
    position = np.asarray(plotter.camera.position, dtype=float)
    offset = position - focal_point
    plotter.camera.focal_point = tuple(view_center)
    plotter.camera.position = tuple(view_center + offset)
    plotter.reset_camera_clipping_range()
    plotter.enable_parallel_projection()
    if parallel_scale is not None:
        plotter.camera.parallel_scale = float(parallel_scale)
        plotter.reset_camera_clipping_range()
    scalar_bar_visibility = [int(actor.GetVisibility()) for actor in plotter.scalar_bars.values()]
    axes_enabled = bool(plotter.renderer.axes_enabled)
    for actor in plotter.scalar_bars.values():
        actor.SetVisibility(bool(draft))
    if plotter.renderer.axes_widget is not None:
        if draft:
            plotter.show_axes()
        else:
            plotter.hide_axes()
    try:
        plotter.render()
        image = plotter.screenshot(return_img=True)
    finally:
        for actor, visible in zip(plotter.scalar_bars.values(), scalar_bar_visibility, strict=True):
            actor.SetVisibility(bool(visible))
        if plotter.renderer.axes_widget is not None:
            if axes_enabled:
                plotter.show_axes()
            else:
                plotter.hide_axes()
    if view == "xy":
        center = (float(view_center[0]), float(view_center[1]))
    elif view == "xz":
        center = (float(view_center[0]), float(view_center[2]))
    elif view == "yz":
        center = (float(view_center[1]), float(view_center[2]))
    else:
        center = (0.0, 0.0)
    width, height = plotter.window_size
    y_half = float(plotter.camera.parallel_scale)
    x_half = y_half * float(width) / float(height)
    extent = (
        float(center[0] - x_half),
        float(center[0] + x_half),
        float(center[1] - y_half),
        float(center[1] + y_half),
    )

    ax.imshow(np.flipud(image), origin="lower", extent=extent)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.minorticks_on()
    ax.tick_params(which="major", length=5.0)
    ax.tick_params(which="minor", length=3.0)

    colorbar = None
    if colorbar_actor is not None:
        lut = colorbar_actor.mapper.lookup_table
        colors = np.asarray(lut.values, dtype=float)[:, :4] / 255.0
        colorbar = fig.colorbar(
            ScalarMappable(
                norm=Normalize(*lut.scalar_range),
                cmap=ListedColormap(colors),
            ),
            ax=ax,
            pad=0.02,
            fraction=0.055,
            shrink=0.9,
            aspect=28,
        )
        if colorbar_label is not None:
            colorbar.set_label(colorbar_label)
        colorbar.minorticks_on()
        colorbar.ax.tick_params(which="major", length=5.0)
        colorbar.ax.tick_params(which="minor", length=3.0)

    return fig, ax, colorbar, image


def apply_matplotlib_color_source(
    actor,
    color_source: ScalarMappable | Colorbar,
):
    mappable = color_source.mappable if isinstance(color_source, Colorbar) else color_source
    vmin = getattr(mappable.norm, "vmin", None)
    vmax = getattr(mappable.norm, "vmax", None)
    if vmin is None or vmax is None:
        vmin, vmax = mappable.get_clim()
    if vmin is None or vmax is None:
        raise ValueError("Matplotlib color source must define a finite scalar range")
    scalar_range = float(vmin), float(vmax)
    samples = np.linspace(scalar_range[0], scalar_range[1], 256)
    colors = np.asarray(mappable.to_rgba(samples, bytes=True), dtype=np.uint8)

    lut = actor.mapper.lookup_table
    lut.values = colors
    lut.scalar_range = scalar_range
    actor.mapper.scalar_range = scalar_range
    return actor


__all__ = ["apply_matplotlib_color_source", "plot_pyvista_viewport"]
