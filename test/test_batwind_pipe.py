from __future__ import annotations

from datetime import datetime
import json
import logging
import os
from pathlib import Path

import numpy as np
import pytest

from batwind.pipelines.dummy_pipeline import name_codepoints_payload
from batwind.pipelines.dummy_pipeline import name_letter_counts
from batwind.pipelines.dummy_pipeline import name_profile_payload
from batwind.pipelines.dummy_pipeline import name_waveform_payload
from batwind.pipelines.dummy_pipeline import process_plt_file
from batwind.pipelines.batwind_pipe import discover_input_files
from batwind.pipelines.batwind_pipe import configure_logger
from batwind.pipelines.batwind_pipe import main
from batwind.pipelines.batwind_pipe import run_batwind_pipe
import batwind.pipelines.batwind_pipe as batwind_pipe_module


def test_discover_input_files_finds_supported_extensions_in_current_directory(tmp_path):
    (tmp_path / "a.plt").write_text("")
    (tmp_path / "b.PLT").write_text("")
    (tmp_path / "c.dat").write_text("")
    (tmp_path / "d.DAT").write_text("")
    (tmp_path / "log_n000000.log").write_text("")
    (tmp_path / "3DALL_t000000_000000.bin").write_text("")
    (tmp_path / "ignore.txt").write_text("")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "c.plt").write_text("")

    files = discover_input_files(tmp_path)
    assert [path.name for path in files] == ["3DALL_t000000_000000.bin", "a.plt", "b.PLT", "c.dat", "d.DAT", "log_n000000.log"]


def test_discover_input_files_includes_known_component_io2_directories_from_run_root(tmp_path, caplog):
    (tmp_path / "SC" / "IO2").mkdir(parents=True)
    (tmp_path / "IH" / "IO2").mkdir(parents=True)
    (tmp_path / "GM" / "IO2").mkdir(parents=True)
    (tmp_path / "SC" / "IO2" / "x=0_sc.plt").write_text("")
    (tmp_path / "IH" / "IO2" / "3d__ih.plt").write_text("")
    (tmp_path / "GM" / "IO2" / "log_n000000.log").write_text("")
    (tmp_path / "SC" / "other").mkdir(parents=True)
    (tmp_path / "SC" / "other" / "ignore.plt").write_text("")

    with caplog.at_level(logging.INFO, logger="batwind.pipelines.batwind_pipe"):
        files = discover_input_files(tmp_path)

    assert [path.relative_to(tmp_path).as_posix() for path in files] == [
        "GM/IO2/log_n000000.log",
        "IH/IO2/3d__ih.plt",
        "SC/IO2/x=0_sc.plt",
    ]
    assert f"Scanning {tmp_path}..." in [record.getMessage() for record in caplog.records]
    assert f"Entering {tmp_path / 'SC' / 'IO2'}..." in [record.getMessage() for record in caplog.records]
    assert f"Entering {tmp_path / 'IH' / 'IO2'}..." in [record.getMessage() for record in caplog.records]
    assert f"Entering {tmp_path / 'GM' / 'IO2'}..." in [record.getMessage() for record in caplog.records]


def test_name_letter_counts_counts_alpha_only():
    assert name_letter_counts("a1-b2") == (1, 1)


def test_name_profile_payload_outputs_float_string_array():
    value = name_profile_payload("alpha")
    assert value == (0.4, "consonant-rich", [5, 2, 3])


def test_name_codepoints_payload_outputs_numpy_array():
    value = name_codepoints_payload("ab")
    assert np.array_equal(value, np.array([97, 98], dtype=np.int32))


def test_name_waveform_payload_outputs_large_numpy_array():
    value = name_waveform_payload("ab")
    assert isinstance(value, np.ndarray)
    assert value.dtype == np.float64
    assert value.shape == (131072,)


def test_dummy_pipeline_process_without_sink_does_not_fail(tmp_path, caplog):
    file_path = tmp_path / "gamma.plt"
    file_path.write_text("")
    with caplog.at_level(logging.INFO, logger="batwind.pipelines.dummy_pipeline"):
        process_plt_file(file_path)
    assert [record.getMessage() for record in caplog.records] == [
        "gamma.plt",
    ]


def test_run_batwind_pipe_default_noclobber_skips_already_processed_files(tmp_path):
    (tmp_path / "alpha.plt").write_text("")
    (tmp_path / "beta.plt").write_text("")

    first = run_batwind_pipe(tmp_path, pipeline="dummy")
    second = run_batwind_pipe(tmp_path, pipeline="dummy")

    assert [path.name for path in first.processed_files] == ["alpha.plt", "beta.plt"]
    assert second.processed_files == []
    assert [path.name for path in second.skipped_files] == ["alpha.plt", "beta.plt"]


def test_run_batwind_pipe_clobber_reprocesses_files(tmp_path):
    (tmp_path / "alpha.plt").write_text("")
    (tmp_path / "beta.plt").write_text("")

    first = run_batwind_pipe(tmp_path, pipeline="dummy")
    second = run_batwind_pipe(tmp_path, pipeline="dummy", noclobber=False)

    assert [path.name for path in first.processed_files] == ["alpha.plt", "beta.plt"]
    assert [path.name for path in second.processed_files] == ["alpha.plt", "beta.plt"]
    assert second.skipped_files == []


def test_run_batwind_pipe_logs_each_processed_file_at_info(tmp_path, caplog):
    file_path = tmp_path / "alpha.plt"
    file_path.write_text("")

    def process_file(_path):
        return None

    with caplog.at_level(logging.INFO, logger="batwind.pipelines.batwind_pipe"):
        run_batwind_pipe(tmp_path, process_file=process_file)

    assert "Processing alpha.plt..." in [record.getMessage() for record in caplog.records]


def test_run_batwind_pipe_reprocesses_when_source_file_is_newer(tmp_path):
    file_path = tmp_path / "alpha.plt"
    file_path.write_text("")
    calls: list[str] = []

    def process_file(path):
        calls.append(Path(path).name)

    run_batwind_pipe(tmp_path, process_file=process_file)

    state_path = tmp_path / "batwind-pipe.custom.processed.json"
    payload = json.loads(state_path.read_text())
    end_time = datetime.fromisoformat(
        payload["computed_results"]["alpha.plt"]["meta"]["end_time_utc"].replace("Z", "+00:00")
    )
    newer_ns = int(end_time.timestamp() * 1_000_000_000) + 2_000_000_000
    os.utime(file_path, ns=(newer_ns, newer_ns))

    second = run_batwind_pipe(tmp_path, process_file=process_file)

    assert calls == ["alpha.plt", "alpha.plt"]
    assert [path.name for path in second.processed_files] == ["alpha.plt"]
    assert second.skipped_files == []


def test_run_batwind_pipe_reprocesses_when_recorded_output_is_missing(tmp_path):
    file_path = tmp_path / "alpha.plt"
    file_path.write_text("")
    output_path = tmp_path / "outputs" / "alpha.txt"
    calls: list[str] = []

    def process_file(path):
        calls.append(Path(path).name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("done")
        recorder_log = logging.getLogger("recorder.test_pipeline")
        recorder_log.setLevel(logging.DEBUG)
        recorder_log.debug("output_txt %r", str(output_path.relative_to(tmp_path)))

    run_batwind_pipe(tmp_path, process_file=process_file)
    output_path.unlink()

    second = run_batwind_pipe(tmp_path, process_file=process_file)

    assert calls == ["alpha.plt", "alpha.plt"]
    assert [path.name for path in second.processed_files] == ["alpha.plt"]
    assert second.skipped_files == []


def test_run_batwind_pipe_discovers_component_io2_directories_from_run_root(tmp_path):
    sc_file = tmp_path / "SC" / "IO2" / "alpha.plt"
    ih_file = tmp_path / "IH" / "IO2" / "beta.plt"
    gm_file = tmp_path / "GM" / "IO2" / "gamma.plt"
    for path in (sc_file, ih_file, gm_file):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")

    def process_file(path):
        recorder_log = logging.getLogger("recorder.test_pipeline")
        recorder_log.setLevel(logging.DEBUG)
        recorder_log.debug("seen %r", Path(path).relative_to(tmp_path).as_posix())

    results = run_batwind_pipe(tmp_path, process_file=process_file)

    assert [path.relative_to(tmp_path).as_posix() for path in results.discovered_files] == [
        "GM/IO2/gamma.plt",
        "IH/IO2/beta.plt",
        "SC/IO2/alpha.plt",
    ]
    assert [path.relative_to(tmp_path).as_posix() for path in results.processed_files] == [
        "GM/IO2/gamma.plt",
        "IH/IO2/beta.plt",
        "SC/IO2/alpha.plt",
    ]
    assert sorted(results.computed_results) == [
        "GM/IO2/gamma.plt",
        "IH/IO2/beta.plt",
        "SC/IO2/alpha.plt",
    ]


def test_run_batwind_pipe_run_root_and_component_io2_share_state(tmp_path):
    file_path = tmp_path / "SC" / "IO2" / "alpha.plt"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("")

    def process_file(path):
        recorder_log = logging.getLogger("recorder.test_pipeline")
        recorder_log.setLevel(logging.DEBUG)
        recorder_log.debug("seen %r", Path(path).name)

    first = run_batwind_pipe(tmp_path, process_file=process_file)
    second = run_batwind_pipe(tmp_path / "SC" / "IO2", process_file=process_file)

    assert [path.relative_to(tmp_path).as_posix() for path in first.processed_files] == ["SC/IO2/alpha.plt"]
    assert second.processed_files == []
    assert [path.name for path in second.skipped_files] == ["alpha.plt"]
    assert (tmp_path / "SC" / "IO2" / "batwind-pipe.custom.processed.json").exists()
    assert not (tmp_path / "batwind-pipe.custom.processed.json").exists()


def test_run_batwind_pipe_continues_after_single_file_failure(tmp_path, caplog):
    (tmp_path / "alpha.plt").write_text("")
    (tmp_path / "beta.plt").write_text("")

    def process_file(path):
        if Path(path).name == "alpha.plt":
            raise RuntimeError("boom")
        recorder_log = logging.getLogger("recorder.test_pipeline")
        recorder_log.setLevel(logging.DEBUG)
        recorder_log.debug("ok %r", True)

    with caplog.at_level(logging.ERROR, logger="batwind.pipelines.batwind_pipe"):
        results = run_batwind_pipe(tmp_path, process_file=process_file)

    assert [path.name for path in results.failed_files] == ["alpha.plt"]
    assert [path.name for path in results.processed_files] == ["beta.plt"]
    assert results.computed_results["alpha.plt"]["meta"]["status"] == "failed"
    assert "boom" in results.computed_results["alpha.plt"]["meta"]["error"]
    assert results.computed_results["beta.plt"]["meta"]["status"] == "processed"
    assert results.computed_results["beta.plt"]["ok"]["value"] is True
    error_records = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert len(error_records) == 1
    assert "alpha.plt" in error_records[0].getMessage()
    assert "boom" in error_records[0].getMessage()
    assert error_records[0].exc_info is None


def test_run_batwind_pipe_fail_fast_raises_on_first_failure(tmp_path):
    (tmp_path / "alpha.plt").write_text("")
    (tmp_path / "beta.plt").write_text("")

    def process_file(path):
        if Path(path).name == "alpha.plt":
            raise RuntimeError("boom")
        recorder_log = logging.getLogger("recorder.test_pipeline")
        recorder_log.setLevel(logging.DEBUG)
        recorder_log.debug("ok %r", True)

    with pytest.raises(RuntimeError, match="boom"):
        run_batwind_pipe(tmp_path, process_file=process_file, fail_fast=True)


def test_run_batwind_pipe_offloads_large_numpy_array_to_artifact(tmp_path):
    (tmp_path / "alpha.plt").write_text("")

    def process_file(_path):
        recorder_log = logging.getLogger("recorder.test_pipeline")
        recorder_log.setLevel(logging.DEBUG)
        recorder_log.debug("big_array %r", np.arange(200_000, dtype=np.float64))

    results = run_batwind_pipe(tmp_path, process_file=process_file, array_offload_min_bytes=1_000_000)
    assert (tmp_path / "batwind-pipe.custom.processed.json").exists()
    entry = results.computed_results["alpha.plt"]["big_array"]
    value = entry["value"]
    assert sorted(value.keys()) == ["path"]
    artifact_path = tmp_path / value["path"]
    assert artifact_path.exists()
    loaded = np.load(artifact_path)
    assert loaded.shape == (200000,)
    assert float(loaded[0]) == 0.0
    assert float(loaded[-1]) == 199999.0
    assert entry["source"]["module"] == "test_pipeline"


def test_run_batwind_pipe_auto_routes_by_filename_prefix_and_records_failures(tmp_path):
    (tmp_path / "3d__one.plt").write_text("")
    (tmp_path / "3DALL_t000000_000000.bin").write_text("")
    (tmp_path / "shl_one.plt").write_text("")
    (tmp_path / "x=0_one.plt").write_text("")
    (tmp_path / "log_n000000.log").write_text("")
    (tmp_path / "misc.plt").write_text("")

    results = run_batwind_pipe(tmp_path)

    assert sorted(path.name for path in results.discovered_files) == [
        "3DALL_t000000_000000.bin",
        "3d__one.plt",
        "log_n000000.log",
        "shl_one.plt",
        "x=0_one.plt",
    ]
    assert sorted(path.name for path in results.failed_files) == [
        "3DALL_t000000_000000.bin",
        "3d__one.plt",
        "log_n000000.log",
        "shl_one.plt",
        "x=0_one.plt",
    ]
    assert sorted(results.computed_results.keys()) == [
        "3DALL_t000000_000000.bin",
        "3d__one.plt",
        "log_n000000.log",
        "shl_one.plt",
        "x=0_one.plt",
    ]
    assert (tmp_path / "batwind-pipe.log.processed.json").exists()
    assert (tmp_path / "batwind-pipe.ua.processed.json").exists()
    assert (tmp_path / "batwind-pipe.volume.processed.json").exists()
    assert (tmp_path / "batwind-pipe.shell.processed.json").exists()
    assert (tmp_path / "batwind-pipe.slice.processed.json").exists()
    assert not (tmp_path / "batwind-pipe.dummy.processed.json").exists()


def test_run_batwind_pipe_explicit_pipeline_only_processes_matching_prefixes(tmp_path):
    (tmp_path / "3d__one.plt").write_text("")
    (tmp_path / "x=0_one.plt").write_text("")
    (tmp_path / "y=0_one.dat").write_text("")

    results = run_batwind_pipe(tmp_path, pipeline="slice")

    assert sorted(path.name for path in results.discovered_files) == ["x=0_one.plt", "y=0_one.dat"]
    assert sorted(path.name for path in results.failed_files) == ["x=0_one.plt", "y=0_one.dat"]
    assert sorted(results.computed_results.keys()) == ["x=0_one.plt", "y=0_one.dat"]
    assert (tmp_path / "batwind-pipe.slice.processed.json").exists()
    assert not (tmp_path / "batwind-pipe.volume.processed.json").exists()


def test_batwind_pipe_main_accepts_builtin_pipeline_names_on_empty_directory(tmp_path):
    for pipeline_name in ("dummy", "log", "slice", "ua", "volume", "shell"):
        code = main([str(tmp_path), "--pipeline", pipeline_name, "--log-level", "ERROR", "--record-log-level", "ERROR"])
        state_file = tmp_path / f"batwind-pipe.{pipeline_name}.processed.json"
        payload = json.loads(state_file.read_text())
        assert code == 0
        assert payload["processed_files"] == []


def test_configure_logger_limits_info_to_pipeline_loggers():
    configure_logger("INFO")

    assert logging.getLogger("batwind.pipelines.slice").getEffectiveLevel() == logging.INFO
    assert logging.getLogger("batwind.smart_ds").getEffectiveLevel() == logging.WARNING
    assert logging.getLogger("batwind.recipes.batsrus").getEffectiveLevel() == logging.WARNING
    assert logging.getLogger("griblet.graph").getEffectiveLevel() == logging.WARNING


def test_configure_logger_debug_enables_internal_loggers():
    configure_logger("DEBUG")

    assert logging.getLogger("batwind.pipelines.slice").getEffectiveLevel() == logging.DEBUG
    assert logging.getLogger("batwind.smart_ds").getEffectiveLevel() == logging.DEBUG
    assert logging.getLogger("griblet.graph").getEffectiveLevel() == logging.DEBUG


def test_run_batwind_pipe_collects_movie_outputs(monkeypatch, tmp_path):
    (tmp_path / "alpha.plt").write_text("")

    expected_movie = tmp_path / "slice" / "alpha.slices.rho.mp4"
    monkeypatch.setattr(
        batwind_pipe_module,
        "write_recorded_png_movies",
        lambda directory, computed_results: [expected_movie],
    )

    def process_file(path):
        png_path = Path(path).parent / "slice" / "alpha.slices.rho.png"
        png_path.parent.mkdir(parents=True, exist_ok=True)
        png_path.write_text("", encoding="utf-8")
        recorder_log = logging.getLogger("recorder.test_pipeline")
        recorder_log.setLevel(logging.DEBUG)
        recorder_log.debug("slice_rho_png %r", str(png_path.relative_to(Path(path).parent)))

    results = run_batwind_pipe(tmp_path, process_file=process_file)

    assert results.movie_outputs == [expected_movie]


def test_run_batwind_pipe_skips_movie_outputs_for_log_pipeline(monkeypatch, tmp_path):
    (tmp_path / "log_n000000.log").write_text("", encoding="utf-8")

    movie_calls = []

    def fake_write_movies(directory, computed_results):
        movie_calls.append((directory, computed_results))
        return [tmp_path / "log" / "log_n000000.mp4"]

    monkeypatch.setattr(batwind_pipe_module, "write_recorded_png_movies", fake_write_movies)

    def fake_process_file(_path):
        recorder_log = logging.getLogger("recorder.test_pipeline")
        recorder_log.setLevel(logging.DEBUG)
        recorder_log.debug("log_all_columns_png %r", "log/log_n000000.all_columns.png")

    monkeypatch.setattr(
        batwind_pipe_module,
        "process_file_for_pipeline",
        lambda pipeline_name: (
            fake_process_file
            if pipeline_name == "log"
            else pytest.fail(f"unexpected pipeline {pipeline_name}")
        ),
    )

    results = run_batwind_pipe(tmp_path)

    assert [path.name for path in results.processed_files] == ["log_n000000.log"]
    assert results.movie_outputs == []
    assert movie_calls == []
