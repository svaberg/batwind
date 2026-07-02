from __future__ import annotations

import logging

import batwind.pipelines.log as log_pipeline
from batwind.pipelines.recorder import BatwindRecordHandler


def test_zdi_ramp_ranges_offset_session_local_iterations_to_absolute_iterations():
    info = log_pipeline.SessionInfo(
        label="s3",
        start=92000,
        stop=95000,
        local_bools={"UseZdiBoundary": True, "UseZdiMagnetogram": True},
        local_scalars={"ZdiRampIterStart": "1000", "ZdiRampIterStop": "2000", "TypeZdiRamp": "cosine"},
        active_bools={"UseZdiBoundary": True, "UseZdiMagnetogram": True},
        active_scalars={"ZdiRampIterStart": "1000", "ZdiRampIterStop": "2000", "TypeZdiRamp": "cosine"},
        summary="ZDI=T/T; Ramp=1000-2000",
    )

    assert log_pipeline.zdi_ramp_ranges([info]) == [(93000, 94000, "cosine")]


def test_flux_panel_title_uses_human_readable_labels():
    assert log_pipeline.flux_panel_title("rho", "total") == "Mass Flux"
    assert log_pipeline.flux_panel_title("jz", "total") == "Angular Momentum Flux"
    assert log_pipeline.flux_panel_title("rho", "in") == "Mass Flux In"


def test_nonzero_log_magnitudes_uses_absolute_values_and_drops_zeros():
    values = log_pipeline.np.array([-3.0, 0.0, 2.0, -1.0])
    assert log_pipeline.np.array_equal(log_pipeline.nonzero_log_magnitudes(values), log_pipeline.np.array([3.0, 2.0, 1.0]))


def test_log_axis_limits_fall_back_for_all_zero_values():
    assert log_pipeline.log_axis_limits(log_pipeline.np.array([0.0, 0.0])) == (1.0e-12, 1.0)


def test_process_log_file_writes_plots_and_session_report(tmp_path):
    (tmp_path / "PARAM.in").write_text(
        "\n".join(
            [
                "Begin session: 1",
                "T DoAmr",
                "T UseZdiBoundary",
                "0 ZdiRampIterStart",
                "20 ZdiRampIterStop",
                "40 MaxIteration",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    file_path = tmp_path / "log_n000010.log"
    file_path.write_text(
        "\n".join(
            [
                "demo log",
                "it rhoflx_R=1.0 jzflx_R=1.0",
                "10 1.0 2.0",
                "20 2.0 3.0",
                "30 3.0 4.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    target: dict[str, object] = {}
    logger = logging.getLogger(f"recorder.{log_pipeline.__name__}")
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
        log_pipeline.process_log_file(file_path)
    finally:
        logger.handlers = old_handlers
        logger.setLevel(old_level)
        logger.propagate = old_propagate

    out = {key: value["value"] for key, value in target.items()}
    assert sorted(out) == ["log_all_columns_png", "log_rho_jz_png", "log_sessions_txt"]

    all_columns_path = tmp_path / out["log_all_columns_png"]
    rho_jz_path = tmp_path / out["log_rho_jz_png"]
    report_path = tmp_path / out["log_sessions_txt"]
    assert all_columns_path.exists()
    assert rho_jz_path.exists()
    assert report_path.exists()
    assert all_columns_path.stat().st_size > 0
    assert rho_jz_path.stat().st_size > 0
    report_text = report_path.read_text(encoding="utf-8")
    assert "log_n000010.log" in report_text
    assert "AMR=T" in report_text
    assert "Ramp=0-20" in report_text
