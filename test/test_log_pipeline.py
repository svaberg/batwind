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


def test_zdi_ramp_ranges_do_not_invent_second_ramp_from_inherited_values():
    sessions = [
        log_pipeline.SessionInfo(
            label="s2",
            start=2000,
            stop=5000,
            local_bools={"UseZdiBoundary": True, "UseZdiMagnetogram": True},
            local_scalars={"ZdiRampIterStart": "2000", "ZdiRampIterStop": "3000", "TypeZdiRamp": "cosine"},
            active_bools={"UseZdiBoundary": True, "UseZdiMagnetogram": True},
            active_scalars={"ZdiRampIterStart": "2000", "ZdiRampIterStop": "3000", "TypeZdiRamp": "cosine"},
            summary="ZDI=T/T; Ramp=2000-3000",
        ),
        log_pipeline.SessionInfo(
            label="s3",
            start=5000,
            stop=6000,
            local_bools={"UseHeatFluxRegion": True},
            local_scalars={},
            active_bools={"UseZdiBoundary": True, "UseZdiMagnetogram": True, "UseHeatFluxRegion": True},
            active_scalars={"ZdiRampIterStart": "2000", "ZdiRampIterStop": "3000", "TypeZdiRamp": "cosine"},
            summary="ZDI=T/T; Ramp=2000-3000",
        ),
    ]

    assert log_pipeline.zdi_ramp_ranges(sessions) == [(2000, 3000, "cosine")]


def test_flux_panel_title_uses_human_readable_labels():
    assert log_pipeline.flux_panel_title("rho", "total") == "Mass Flux"
    assert log_pipeline.flux_panel_title("jz", "total") == "Angular Momentum Flux"
    assert log_pipeline.flux_panel_title("rho", "in") == "Mass Flux In"


def test_corrected_panel_series_returns_raw_jz_values():
    data = log_pipeline.np.array(
        [
            [10.0, 1.0, 2.0, 3.0, 7.0, 8.0, 9.0],
            [20.0, 4.0, 5.0, 6.0, 10.0, 11.0, 12.0],
        ]
    )
    member_lookup = {
        ("rho", "total"): [("20", 1), ("30", 2), ("40", 3)],
        ("jz", "total"): [("20", 4), ("30", 5), ("40", 6)],
    }

    plotted, title = log_pipeline.corrected_panel_series(
        base_name="jz",
        variant_name="total",
        members=member_lookup[("jz", "total")],
        member_lookup=member_lookup,
        data=data,
    )

    assert title == "Angular Momentum Flux"
    assert [radius for radius, _ in plotted] == ["20", "30", "40"]
    assert log_pipeline.np.array_equal(plotted[0][1], data[:, 4])
    assert log_pipeline.np.array_equal(plotted[1][1], data[:, 5])
    assert log_pipeline.np.array_equal(plotted[2][1], data[:, 6])


def test_nonzero_log_magnitudes_uses_absolute_values_and_drops_zeros():
    values = log_pipeline.np.array([-3.0, 0.0, 2.0, -1.0])
    assert log_pipeline.np.array_equal(log_pipeline.nonzero_log_magnitudes(values), log_pipeline.np.array([3.0, 2.0, 1.0]))


def test_log_axis_limits_fall_back_for_all_zero_values():
    assert log_pipeline.log_axis_limits(log_pipeline.np.array([0.0, 0.0])) == (1.0e-12, 1.0)


def test_merge_log_block_data_prefers_later_overlap_and_stays_monotonic():
    block = [
        (
            log_pipeline.Path("log_n000000.log"),
            "demo",
            ["it", "rhoflx_R=1.0"],
            log_pipeline.np.array(
                [
                    [0.0, 1.0],
                    [10.0, 2.0],
                    [20.0, 3.0],
                ]
            ),
        ),
        (
            log_pipeline.Path("log_n000020.log"),
            "demo",
            ["it", "rhoflx_R=1.0"],
            log_pipeline.np.array(
                [
                    [20.0, 30.0],
                    [30.0, 40.0],
                ]
            ),
        ),
    ]

    merged = log_pipeline.merge_log_block_data(block)

    assert log_pipeline.np.array_equal(merged[:, 0], log_pipeline.np.array([0.0, 10.0, 20.0, 30.0]))
    assert log_pipeline.np.array_equal(merged[:, 1], log_pipeline.np.array([1.0, 2.0, 30.0, 40.0]))
    assert log_pipeline.iteration_is_strictly_monotonic(merged)


def test_process_log_file_writes_merged_plots_for_contiguous_neighbors(tmp_path):
    (tmp_path / "PARAM.in").write_text(
        "\n".join(
            [
                "Begin session: 1",
                "T DoAmr",
                "T UseZdiBoundary",
                "0 ZdiRampIterStart",
                "40 ZdiRampIterStop",
                "80 MaxIteration",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "log_n000000.log").write_text(
        "\n".join(
            [
                "demo log",
                "it rhoflx_R=1.0 jzflx_R=1.0",
                "0 1.0 2.0",
                "10 2.0 3.0",
                "20 3.0 4.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    file_path = tmp_path / "log_n000020.log"
    file_path.write_text(
        "\n".join(
            [
                "demo log",
                "it rhoflx_R=1.0 jzflx_R=1.0",
                "20 10.0 20.0",
                "30 11.0 21.0",
                "40 12.0 22.0",
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
    assert sorted(out) == [
        "log_all_columns_merged_png",
        "log_all_columns_png",
        "log_rho_jz_merged_png",
        "log_rho_jz_png",
        "log_sessions_txt",
    ]
    assert (tmp_path / out["log_all_columns_merged_png"]).exists()
    assert (tmp_path / out["log_rho_jz_merged_png"]).exists()
    assert "log_n000000__to__log_n000020.merged.all_columns.png" in out["log_all_columns_merged_png"]
    assert "log_n000000__to__log_n000020.merged.rho_jz.png" in out["log_rho_jz_merged_png"]


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
    assert "Ramp=10-20" in report_text
