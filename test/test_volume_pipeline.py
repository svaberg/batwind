from __future__ import annotations

import logging
from pathlib import Path
import shutil

import batwind.pipelines.volume as volume
from batwind.pipelines.recorder import BatwindRecordHandler
from batwind.smart_ds import SmartDs


EXAMPLE_PLT = Path("examples/3d__var_1_n00000000.plt")


def test_process_plt_file_records_3d_quantities_and_writes_surface_plots(tmp_path, monkeypatch):
    monkeypatch.setattr(volume, "FIELDLINE_FRACTION_N_SEEDS", 48)
    monkeypatch.setattr(volume, "ANGULAR_MAP_N_POLAR", 8)
    monkeypatch.setattr(volume, "ANGULAR_MAP_N_AZIMUTH", 16)
    monkeypatch.setattr(volume, "SURFACE_RENDER_N_POLAR", 12)
    monkeypatch.setattr(volume, "SURFACE_RENDER_N_AZIMUTH", 24)
    monkeypatch.setattr(volume, "SURFACE_VIEWPORT_FIGSIZE", (5.0, 5.0))
    monkeypatch.setattr(volume, "SURFACE_VIEWPORT_DPI", 120)
    monkeypatch.setattr(volume, "SURFACE_VIEWPORT_RENDER_SIZE", (900, 900))
    monkeypatch.setattr(volume, "LOS_GRID_N", 64)
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
    assert full_png.exists()
    assert full_png.stat().st_size > 0
    assert closed_png.exists()
    assert closed_png.stat().st_size > 0
    assert str(full_png.relative_to(tmp_path)).startswith("volume/")
    assert str(closed_png.relative_to(tmp_path)).startswith("volume/")
    assert out["polar_map_rad"]
    assert out["azimuth_map_rad"]
    assert out["angular_cell_solid_angle_sr"]
    assert out["alfven_radius_map_R"]
    assert out["field_line_max_radius_map_R"]
    assert 0.0 <= out["open_flux_fraction"] <= 1.0
    assert 0.0 <= out["open_area_fraction"] <= 1.0
    assert 0.0 <= out["current_sheet_inclination_deg"] <= 90.0
