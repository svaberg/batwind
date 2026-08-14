from __future__ import annotations

import logging
from pathlib import Path
import shutil

import numpy as np
from batread import Dataset

import batwind.pipelines.volume as volume
from batwind.pipelines.recorder import BatwindRecordHandler
from batwind.smart_ds import SmartDs


EXAMPLE_PLT = Path("examples/3d__var_1_n00000000.plt")


def test_process_plt_file_records_3d_quantities_and_writes_surface_plots(tmp_path, monkeypatch):
    monkeypatch.setattr(volume, "FIELDLINE_FRACTION_N_SEEDS", 48)
    monkeypatch.setattr(volume, "FIELD_LINE_OVERLAY_N_SEEDS", 48)
    monkeypatch.setattr(volume, "ANGULAR_MAP_N_POLAR", 8)
    monkeypatch.setattr(volume, "ANGULAR_MAP_N_AZIMUTH", 16)
    monkeypatch.setattr(volume, "SURFACE_RENDER_N_POLAR", 12)
    monkeypatch.setattr(volume, "SURFACE_RENDER_N_AZIMUTH", 24)
    monkeypatch.setattr(volume, "SURFACE_VIEWPORT_FIGSIZE", (5.0, 5.0))
    monkeypatch.setattr(volume, "SURFACE_VIEWPORT_DPI", 120)
    monkeypatch.setattr(volume, "SURFACE_VIEWPORT_RENDER_SIZE", (900, 900))
    monkeypatch.setattr(volume, "LOS_GRID_N", 64)
    monkeypatch.setattr(volume, "LOS_EXAMPLE_GRID_N", 48)
    monkeypatch.setattr(volume, "CORONAL_EMISSION_TOTALS_IMAGE_N", 64)
    original_from_file = SmartDs.from_file
    monkeypatch.setattr(
        volume.SmartDs,
        "from_file",
        classmethod(lambda cls, file, **kwargs: original_from_file(str(file), body_radius_m=1.0, **kwargs)),
    )

    copied_plt = tmp_path / EXAMPLE_PLT.name
    shutil.copyfile(EXAMPLE_PLT, copied_plt)
    target: dict[str, object] = {}
    logger = logging.getLogger(f"recorder.{volume.__name__}")
    handler = BatwindRecordHandler(
        target,
        file_key=str(copied_plt),
        artifacts_root=tmp_path / "batwind-pipe.artifacts",
    )
    old_handlers = list(logger.handlers)
    old_level = logger.level
    old_propagate = logger.propagate
    logger.handlers = [handler]
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    try:
        volume.process_plt_file(copied_plt)
    finally:
        logger.handlers = old_handlers
        logger.setLevel(old_level)
        logger.propagate = old_propagate

    out = {key: value["value"] for key, value in target.items()}
    full_png = tmp_path / out["volume_field_line_max_radius_surface_png"]
    closed_png = tmp_path / out["volume_closed_field_line_envelope_png"]
    hard_overlay_png = tmp_path / out["volume_hard_los_example_field_lines_png"]
    assert full_png.exists()
    assert full_png.stat().st_size > 0
    assert closed_png.exists()
    assert closed_png.stat().st_size > 0
    assert hard_overlay_png.exists()
    assert hard_overlay_png.stat().st_size > 0
    assert str(full_png.relative_to(tmp_path)).startswith("volume/")
    assert str(closed_png.relative_to(tmp_path)).startswith("volume/")
    assert str(hard_overlay_png.relative_to(tmp_path)).startswith("volume/")
    assert out["polar_map_rad"]
    assert out["azimuth_map_rad"]
    assert out["angular_cell_solid_angle_sr"]
    assert out["alfven_radius_map_R"]
    assert out["field_line_max_radius_map_R"]
    assert 0.0 <= out["open_flux_fraction"] <= 1.0
    assert 0.0 <= out["open_area_fraction"] <= 1.0
    assert 0.0 <= out["current_sheet_inclination_deg"] <= 90.0


def test_process_smart_ds_accepts_preloaded_dataset(tmp_path, monkeypatch):
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
    smart_ds = SmartDs(
        Dataset(
            points,
            corners,
            aux={"RBODY": "1.00"},
            title="preloaded",
            variables=["X [R]", "Y [R]", "Z [R]"],
            zone="preloaded",
        )
    )
    call_order: list[str] = []

    monkeypatch.setattr(
        volume,
        "sample_shell_grid",
        lambda _smart_ds, _radii: ({}, np.array([2.0], dtype=float), False),
    )
    monkeypatch.setattr(volume, "record_wind_mass_loss", lambda *args, **kwargs: call_order.append("mass"))
    monkeypatch.setattr(volume, "record_wind_torque", lambda *args, **kwargs: call_order.append("torque"))
    monkeypatch.setattr(volume, "record_open_magnetic_flux", lambda *args, **kwargs: call_order.append("flux"))
    monkeypatch.setattr(volume, "record_energy_flux", lambda *args, **kwargs: call_order.append("energy"))
    monkeypatch.setattr(volume, "record_3d_quantities", lambda _smart_ds: call_order.append("3d") or 5.0)
    monkeypatch.setattr(volume, "save_field_line_surface_plots", lambda *args, **kwargs: call_order.append("surface"))
    monkeypatch.setattr(volume, "save_los_images", lambda *args, **kwargs: call_order.append("los"))

    path = tmp_path / "3d__var_2_n00050000.plt"
    path.write_text("")
    volume.process_smart_ds(smart_ds, path=path)

    assert call_order == ["mass", "torque", "flux", "energy", "3d", "surface", "los"]
    shell_png = tmp_path / "volume" / "3d_var_2_n00050000.shells.png"
    assert shell_png.exists()
    assert shell_png.stat().st_size > 0


def test_save_los_images_skips_coronal_emission_when_te_is_missing(tmp_path, monkeypatch):
    points = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    corners = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=int)
    smart_ds = SmartDs(
        Dataset(
            points,
            corners,
            aux={"RBODY [m]": 1.0},
            title="rho-only",
            variables=["X [R]", "Y [R]", "Z [R]", "Rho [kg/m^3]"],
            zone="rho-only",
        )
    )

    monkeypatch.setattr(volume, "build_los_geometry", lambda _smart_ds: (None, None, (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)))
    monkeypatch.setattr(volume, "build_los_interpolator", lambda _tree, _values: object())
    monkeypatch.setattr(
        volume,
        "render_rho2_los_image",
        lambda *_args, **_kwargs: (np.ones((2, 2), dtype=float), (-1.0, 1.0, -1.0, 1.0), np.ones((2, 2), dtype=int)),
    )
    monkeypatch.setattr(volume, "build_magnetic_field_lines", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("Coronal emission branch should be skipped when te [K] is missing")
    ))

    def _write_npz(npz_path, *_args, **_kwargs):
        np.savez_compressed(npz_path, demo=np.array([1.0], dtype=float))

    def _write_png(_npz_path, png_path):
        png_path.write_bytes(b"png")

    monkeypatch.setattr(volume, "save_los_colormesh_npz", _write_npz)
    monkeypatch.setattr(volume, "plot_los_colormesh_npz", _write_png)
    monkeypatch.setattr(volume, "save_example_los_colormesh_npz", _write_npz)
    monkeypatch.setattr(volume, "plot_example_los_colormesh_npz", _write_png)

    volume.save_los_images(smart_ds, tmp_path, "demo", tmp_path, 5.0)

    assert (tmp_path / "demo.rho2_los.png").exists()
    assert (tmp_path / "demo.rho2_los_side.png").exists()
    assert (tmp_path / "demo.rho2_los_example.png").exists()
    assert not (tmp_path / "demo.hard_los_example.png").exists()
