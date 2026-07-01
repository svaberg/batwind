from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

import batwind.pipelines.shell as shell_pipeline
from batwind.pipelines.recorder import BatwindRecordHandler


class FakeMagneticShellDs:
    def __init__(self) -> None:
        self._fields = {
            "Lon [deg]": np.array([0.0, 1.0, 0.0, 1.0], dtype=float),
            "Lat [deg]": np.array([0.0, 0.0, 1.0, 1.0], dtype=float),
            "B_r [T]": np.array([1.0, 2.0, 3.0, 4.0], dtype=float),
            "bphi [T]": np.array([-1.0, -2.0, -3.0, -4.0], dtype=float),
            "btheta [T]": np.array([0.5, 1.0, 1.5, 2.0], dtype=float),
        }

    def __contains__(self, name: str) -> bool:
        return name in self._fields

    def __getitem__(self, name: str):
        return self._fields[name]


def test_process_plt_file_plots_magnetic_shell_component_maps(monkeypatch, tmp_path):
    file_path = tmp_path / "shl_demo.plt"
    file_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        shell_pipeline.SmartDs,
        "from_file",
        staticmethod(lambda *args, **kwargs: FakeMagneticShellDs()),
    )

    target: dict[str, object] = {}
    logger = logging.getLogger(f"recorder.{shell_pipeline.__name__}")
    handler = BatwindRecordHandler(
        target,
        file_key=str(file_path),
        artifacts_root=tmp_path / "batwind-pipe.artifacts",
    )
    old_handlers = list(logger.handlers)
    old_level = logger.level
    old_propagate = logger.propagate
    logger.handlers = [handler]
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    try:
        shell_pipeline.process_plt_file(file_path)
    finally:
        logger.handlers = old_handlers
        logger.setLevel(old_level)
        logger.propagate = old_propagate

    out = {key: value["value"] for key, value in target.items()}
    assert sorted(key for key in out if key.endswith("_png")) == ["shell_magnetic_components_png"]
    assert "shell_mass_flux_map_png" not in out

    output_path = tmp_path / out["shell_magnetic_components_png"]
    assert output_path.exists()
    assert output_path.stat().st_size > 0
