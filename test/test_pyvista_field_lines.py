from pathlib import Path
import inspect

import matplotlib.pyplot as plt
import numpy as np
import pytest

pytest.importorskip("pyvista")

from batwind.pyvista import (
    build_magnetic_field_lines,
    open_flux_and_area_fractions,
    plot_magnetic_field_lines,
    plot_pyvista_viewport,
)
from batwind.pyvista.fields import radial_component
from test.pyvista_test_support import make_structured_dataset, scalar_bar_actor, scalar_mesh_actor


EXAMPLE_PLT = Path("examples/3d__var_1_n00000000.plt")

def _make_magnetic_dataset(vector_field, *, n: int = 9, extent: float = 2.0):
    xs = np.linspace(-extent, extent, n)
    ys = np.linspace(-extent, extent, n)
    zs = np.linspace(-extent, extent, n)
    x, y, z = np.meshgrid(xs, ys, zs, indexing="ij")
    bx, by, bz = vector_field(x, y, z)
    return make_structured_dataset(
        [
            x.ravel(),
            y.ravel(),
            z.ravel(),
            bx.ravel(),
            by.ravel(),
            bz.ravel(),
        ],
        n=n,
        aux={"RBODY": 1.0},
        title="synthetic-magnetic",
        variables=[
            "X [R]",
            "Y [R]",
            "Z [R]",
            "B_x [Gauss]",
            "B_y [Gauss]",
            "B_z [Gauss]",
        ],
        zone="synthetic",
    )


def test_field_line_defaults_use_hundreds_of_seeds():
    assert inspect.signature(build_magnetic_field_lines).parameters["n_seeds"].default == 256
    assert inspect.signature(plot_magnetic_field_lines).parameters["n_seeds"].default == 256


def test_build_magnetic_field_lines_on_uniform_field_marks_all_lines_open():
    dataset = _make_magnetic_dataset(
        lambda x, y, z: (np.ones_like(x), np.zeros_like(y), np.zeros_like(z)),
    )
    _grid, source, lines = build_magnetic_field_lines(dataset, n_seeds=24)

    is_open = np.asarray(lines.cell_data["field_line_is_open"], dtype=bool)
    end_radius = np.asarray(lines.cell_data["field_line_end_radius [R]"], dtype=float)

    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(source.points, dtype=float), axis=1),
        1.02,
        atol=1e-12,
    )
    assert lines.n_cells == 48
    assert np.all(is_open)
    assert np.all(end_radius > 1.2)


def test_build_magnetic_field_lines_on_toroidal_field_marks_all_lines_closed():
    dataset = _make_magnetic_dataset(
        lambda x, y, z: (-y, x, np.zeros_like(z)),
    )
    _grid, _source, lines = build_magnetic_field_lines(dataset, n_seeds=24)

    is_open = np.asarray(lines.cell_data["field_line_is_open"], dtype=bool)
    end_radius = np.asarray(lines.cell_data["field_line_end_radius [R]"], dtype=float)
    max_radius = np.asarray(lines.cell_data["field_line_max_radius [R]"], dtype=float)

    assert lines.n_cells == 48
    assert not np.any(is_open)
    np.testing.assert_allclose(end_radius, 1.02, atol=2e-3)
    np.testing.assert_allclose(max_radius, 1.02, atol=2e-3)


def test_open_flux_and_area_fractions_on_uniform_field_are_all_open():
    dataset = _make_magnetic_dataset(
        lambda x, y, z: (np.ones_like(x), np.zeros_like(y), np.zeros_like(z)),
    )
    out = open_flux_and_area_fractions(dataset, n_seeds=24, open_radius=1.2)

    assert out["open_count"] == 24
    assert out["closed_count"] == 0
    assert out["undetermined_count"] == 0
    assert out["open_area_fraction [none]"] == pytest.approx(1.0)
    assert out["open_flux_fraction [none]"] == pytest.approx(1.0)
    assert out["undetermined_area_fraction [none]"] == pytest.approx(0.0)
    assert out["undetermined_flux_fraction [none]"] == pytest.approx(0.0)


def test_open_flux_and_area_fractions_on_toroidal_field_are_closed_with_zero_radial_flux():
    dataset = _make_magnetic_dataset(
        lambda x, y, z: (-y, x, np.zeros_like(z)),
    )
    out = open_flux_and_area_fractions(dataset, n_seeds=24, open_radius=1.2)

    assert out["open_count"] == 0
    assert out["closed_count"] == 24
    assert out["undetermined_count"] == 0
    assert out["open_area_fraction [none]"] == pytest.approx(0.0)
    assert out["undetermined_area_fraction [none]"] == pytest.approx(0.0)
    assert out["open_flux_fraction [none]"] == pytest.approx(0.0, abs=1e-12)
    assert out["undetermined_flux_fraction [none]"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_build_magnetic_field_lines_from_example_file():
    grid, source, lines = build_magnetic_field_lines(str(EXAMPLE_PLT), n_seeds=24)
    body_radius = float(np.asarray(grid.field_data["RBODY [R]"]).ravel()[0])

    assert grid.n_points > 0
    assert source.n_points == 24
    assert lines.n_points > 0
    assert lines.n_cells > 0

    assert "B_vec [T]" in grid.point_data
    assert "B_vec [T]" in lines.point_data
    assert "B_r [T]" in lines.point_data
    assert "field_line_is_open" in lines.cell_data
    assert "field_line_end_radius [R]" in lines.cell_data
    assert "field_line_max_radius [R]" in lines.cell_data

    seed_radii = np.linalg.norm(np.asarray(source.points, dtype=float), axis=1)
    np.testing.assert_allclose(seed_radii, 1.02 * body_radius, atol=1e-12)

    computed_br = radial_component(
        np.asarray(lines.point_data["B_vec [T]"], dtype=float),
        np.asarray(lines.points, dtype=float),
    )
    np.testing.assert_allclose(
        np.asarray(lines.point_data["B_r [T]"], dtype=float),
        computed_br,
        atol=0.0,
        rtol=0.0,
    )

    is_open = np.asarray(lines.cell_data["field_line_is_open"], dtype=bool)
    assert is_open.any()
    assert (~is_open).any()

    end_radius = np.asarray(lines.cell_data["field_line_end_radius [R]"], dtype=float)
    max_radius = np.asarray(lines.cell_data["field_line_max_radius [R]"], dtype=float)
    assert np.all(end_radius[~is_open] <= 1.2 + 1e-6)
    assert np.all(end_radius[is_open] > 1.2 - 1e-6)
    assert np.all(max_radius + 1e-12 >= end_radius)

    assert np.any(np.asarray(lines.point_data["B_r [T]"]) > 0.0)
    assert np.any(np.asarray(lines.point_data["B_r [T]"]) < 0.0)


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_open_flux_and_area_fractions_on_example_file_are_bounded():
    out = open_flux_and_area_fractions(str(EXAMPLE_PLT), n_seeds=48, open_radius=30.0)

    assert out["n_seeds"] == 48
    assert out["open_count"] + out["closed_count"] + out["undetermined_count"] == 48
    assert 0.0 <= out["open_area_fraction [none]"] <= 1.0
    assert 0.0 <= out["undetermined_area_fraction [none]"] <= 1.0
    assert out["open_area_fraction [none]"] + out["undetermined_area_fraction [none]"] <= 1.0
    assert np.isfinite(out["open_flux_fraction [none]"])
    assert np.isfinite(out["undetermined_flux_fraction [none]"])
    assert 0.0 <= out["open_flux_fraction [none]"] <= 1.0
    assert 0.0 <= out["undetermined_flux_fraction [none]"] <= 1.0
    assert out["open_flux_fraction [none]"] + out["undetermined_flux_fraction [none]"] <= 1.0


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_plot_magnetic_field_lines_off_screen_writes_screenshot(tmp_path):
    screenshot = tmp_path / "magnetic-field-lines.png"
    viewport_screenshot = tmp_path / "magnetic-field-lines-viewport.png"
    plot_radius = 3.0
    open_line_plot_radius = 1.5
    default_n_seeds = inspect.signature(plot_magnetic_field_lines).parameters["n_seeds"].default

    fig = None
    plotter, lines, source, _grid = plot_magnetic_field_lines(
        str(EXAMPLE_PLT),
        plot_radius=plot_radius,
        open_line_plot_radius=open_line_plot_radius,
        off_screen=True,
        screenshot=screenshot,
    )
    try:
        scalar_bar = scalar_bar_actor(plotter)
        colorbar_actor = scalar_mesh_actor(plotter)
        fig, ax = plt.subplots(figsize=(7.0, 7.0), dpi=180, constrained_layout=True)
        fig, ax, colorbar, image = plot_pyvista_viewport(
            plotter,
            fig=fig,
            ax=ax,
            colorbar_actor=colorbar_actor,
            colorbar_label="symlog B_r [arb]",
            view="isometric",
            view_center=(0.0, 0.0, 0.0),
            parallel_scale=1.05 * plot_radius,
        )
        ax.grid(True, color="0.88", linewidth=0.6)
        ax.axhline(0.0, color="0.82", linewidth=0.8)
        ax.axvline(0.0, color="0.82", linewidth=0.8)
        fig.savefig(viewport_screenshot, dpi=180)

        assert lines.n_points > 0
        assert source.n_points == default_n_seeds
        assert "symlog B_r [arb]" in lines.point_data
        assert np.linalg.norm(lines.points, axis=1).max() <= 3.0 + 1e-6

        is_open = np.asarray(lines.cell_data["field_line_is_open"], dtype=bool)
        open_lines = lines.extract_cells(np.flatnonzero(is_open))
        assert open_lines.n_cells > 0
        assert np.linalg.norm(open_lines.points, axis=1).max() <= 1.5 + 1e-6

        assert screenshot.exists()
        assert screenshot.stat().st_size > 0
        assert viewport_screenshot.exists()
        assert viewport_screenshot.stat().st_size > 0
        assert scalar_bar.GetTitleTextProperty().GetFontSize() >= 36
        assert scalar_bar.GetLabelTextProperty().GetFontSize() >= 30
        assert colorbar is not None
        assert colorbar.ax.yaxis.label.get_text() == "symlog B_r [arb]"
        assert np.any(np.any(image[:, :, :3] != image[0, 0, :3], axis=2))
    finally:
        if fig is not None:
            plt.close(fig)
        plotter.close()
