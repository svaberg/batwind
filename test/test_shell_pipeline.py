from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import batwind.pipelines.shell as shell_pipeline
from batwind.pipelines.recorder import BatwindRecordHandler


class FakeMagneticShellDs:
    def __init__(self) -> None:
        self._fields = {
            "I": np.array([1.0, 2.0, 1.0, 2.0], dtype=float),
            "J": np.array([1.0, 1.0, 2.0, 2.0], dtype=float),
            "Lon [deg]": np.array([0.0, 1.0, 0.0, 1.0], dtype=float),
            "Lat [deg]": np.array([0.0, 0.0, 1.0, 1.0], dtype=float),
            "B_r [T]": np.array([1.0, 2.0, 3.0, 4.0], dtype=float),
            "bphi [T]": np.array([-1.0, -2.0, -3.0, -4.0], dtype=float),
            "btheta [T]": np.array([0.5, 1.0, 1.5, 2.0], dtype=float),
        }
        self.raw = SimpleNamespace(variables=list(self._fields))

    def __getitem__(self, name: str):
        return self._fields[name]


class FakeStructuredShellDs:
    def __init__(self) -> None:
        self._fields = {
            "I": np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2], dtype=float),
            "J": np.array([1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3], dtype=float),
            "K": np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2], dtype=float),
            "R [R]": np.array([2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0], dtype=float),
            "Lon [deg]": np.array([0.0, 0.0, 10.0, 10.0, 20.0, 20.0, 0.0, 0.0, 10.0, 10.0, 20.0, 20.0], dtype=float),
            "Lat [deg]": np.array([-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0], dtype=float),
            "RBODY [m]": np.array([2.0], dtype=float),
            "mass_flux [kg/m^2/s]": np.array([1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0, 6.0, 60.0], dtype=float),
        }
        self.raw = SimpleNamespace(variables=list(self._fields))

    def __getitem__(self, name: str):
        return self._fields[name]


def test_process_plt_file_plots_magnetic_shell_component_maps(monkeypatch, tmp_path):
    file_path = tmp_path / "shl_demo_n00000042.plt"
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


def test_process_plt_file_passes_iteration_label_to_magnetic_plot(monkeypatch, tmp_path):
    file_path = tmp_path / "shl_demo_n00000042.plt"
    file_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        shell_pipeline.SmartDs,
        "from_file",
        staticmethod(lambda *args, **kwargs: FakeMagneticShellDs()),
    )

    captured: dict[str, object] = {}

    def fake_plot(component_maps, *, iteration_label, lon_nodes, lat_nodes, output_path):
        captured["iteration_label"] = iteration_label
        captured["output_path"] = output_path
        output_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(shell_pipeline, "plot_shell_component_stack_png", fake_plot)

    shell_pipeline.process_plt_file(file_path)

    assert captured["iteration_label"] == "Iteration n00000042"
    assert Path(captured["output_path"]).name == "shl_demo_n00000042.magnetic_components.png"


def test_shell_map_and_profile_uses_native_ijk_layout():
    smart_ds = FakeStructuredShellDs()

    grid_shape, _, _, _, shell_areas_m2 = shell_pipeline.load_shell_grid(smart_ds)
    mass_flux_map, mass_loss_kg_s = shell_pipeline.shell_map_and_profile(
        smart_ds["mass_flux [kg/m^2/s]"],
        grid_shape=grid_shape,
        shell_areas_m2=shell_areas_m2,
    )

    np.testing.assert_allclose(mass_flux_map, [[30.0, 40.0]])

    solid_angle = (
        np.sin(np.deg2rad(np.array([10.0])[:, None])) - np.sin(np.deg2rad(np.array([-10.0])[:, None]))
    ) * np.deg2rad(np.array([10.0, 10.0]))[None, :]
    np.testing.assert_allclose(mass_loss_kg_s, [7.0 * 16.0 * solid_angle[0, 0], 70.0 * 36.0 * solid_angle[0, 0]])
