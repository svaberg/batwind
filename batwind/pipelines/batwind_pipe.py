"""Generic `batwind-pipe` orchestration CLI.
"""

# It discovers supported input files in a working directory and runs a
# per-file pipeline handler. Built-in handlers are `dummy`, `slice`, `shell`,
# `ua`, and `volume`.

from __future__ import annotations

import argparse
from datetime import datetime
from datetime import timezone
import logging
from pathlib import Path
import sys
from typing import Callable

import colorlog

from batwind.pipelines.recorder import DEFAULT_ARRAY_OFFLOAD_MIN_BYTES
from batwind.pipelines.recorder import DEFAULT_JSON_WARN_BYTES
from batwind.pipelines.recorder import BatwindPipeResults
from batwind.pipelines.recorder import BatwindRecordHandler
from batwind.pipelines.recorder import load_state
from batwind.pipelines.recorder import relative_file_key
from batwind.pipelines.recorder import save_state
from batwind.pipelines.recorder import sha256_file
from batwind.pipelines.recorder import state_file_path

log = logging.getLogger(__name__)
PIPELINE_LOG_FORMAT = "[%(levelname)s] %(pipeline_source)s %(message)s"
PIPELINE_COLOR_LOG_FORMAT = "%(log_color)s[%(levelname)s]%(reset)s %(pipeline_source)s %(message)s"


class PipelineSourceFilter(logging.Filter):
    """
    Add the shared `pipeline_source` field used by pipeline log formatters.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Populate the shared pipeline-source field for one log record.
        """
        if record.name.startswith("recorder."):
            record.pipeline_source = f"{record.name}.{record.funcName}:{record.lineno}"
        else:
            record.pipeline_source = record.name.rsplit(".", 1)[-1]
        return True


def configure_logger(level_name: str) -> None:
    """
    Configure the root logger for human-readable pipeline logs on stdout.
    """
    level = getattr(logging, str(level_name).upper())
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.addFilter(PipelineSourceFilter())

    if sys.stdout.isatty():
        handler.setFormatter(
            colorlog.ColoredFormatter(
                PIPELINE_COLOR_LOG_FORMAT,
                reset=True,
                style="%",
            )
        )
        log.debug("configure_logger using colorlog formatter")

    if handler.formatter is None:
        handler.setFormatter(logging.Formatter(PIPELINE_LOG_FORMAT))

    logger.addHandler(handler)

    # Pin noisy third-party libraries to saner defaults while letting batwind
    # follow the active root/capture configuration.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("griblet").setLevel(logging.INFO)


def configure_recorder(level_name: str = "WARNING") -> None:
    """
    Configure the dedicated recorder logger stream level.
    """
    recorder = logging.getLogger("recorder")
    recorder.setLevel(logging.DEBUG)
    recorder.handlers.clear()

    handler = logging.StreamHandler()
    handler.setLevel(getattr(logging, str(level_name).upper()))
    handler.addFilter(PipelineSourceFilter())
    handler.setFormatter(logging.Formatter(PIPELINE_LOG_FORMAT))

    recorder.addHandler(handler)
    recorder.propagate = False


def discover_input_files(directory: str | Path = ".", *, recursive: bool = False) -> list[Path]:
    """
    Discover supported input files in a directory.
    """
    base = Path(directory)
    paths = base.rglob("*") if recursive else base.iterdir()
    files = [path for path in paths if path.is_file() and path.suffix.lower() in {".plt", ".dat", ".bin"}]
    return sorted(files)


def pipeline_name_for_file(file_path: str | Path) -> str | None:
    """
    Infer built-in pipeline from the input filename prefix.
    """
    file_name = Path(file_path).name.lower()
    if file_name.startswith("3dall") and file_name.endswith(".bin"):
        return "ua"
    if file_name.startswith("3d"):
        return "volume"
    if file_name.startswith("shl"):
        return "shell"
    if file_name.startswith("x=0") or file_name.startswith("y=0") or file_name.startswith("z=0"):
        return "slice"
    return None


def process_file_for_pipeline(pipeline_name: str) -> Callable[[Path], None]:
    """
    Return the built-in per-file process function for one pipeline name.
    """
    if pipeline_name == "dummy":
        from batwind.pipelines.dummy_pipeline import process_plt_file

        return process_plt_file
    if pipeline_name == "slice":
        from batwind.pipelines.slice import process_plt_file

        return process_plt_file
    if pipeline_name == "shell":
        from batwind.pipelines.shell import process_plt_file

        return process_plt_file
    if pipeline_name == "ua":
        from batwind.pipelines.ua import process_bin_file

        return process_bin_file
    if pipeline_name == "volume":
        from batwind.pipelines.volume import process_plt_file

        return process_plt_file
    raise KeyError(f"Unknown pipeline '{pipeline_name}'")


def run_batwind_pipe(
    directory: str | Path = ".",
    *,
    pipeline: str | None = None,
    recursive: bool = False,
    noclobber: bool = False,
    include_file_hash: bool = False,
    array_offload_min_bytes: int = DEFAULT_ARRAY_OFFLOAD_MIN_BYTES,
    json_warn_bytes: int = DEFAULT_JSON_WARN_BYTES,
    fail_fast: bool = False,
    process_file: Callable[[Path], None] | None = None,
) -> BatwindPipeResults:
    """
    Run `batwind-pipe` over discovered input files in a directory.
    """
    log.info("run_batwind_pipe...")
    files = discover_input_files(directory, recursive=recursive)
    directory_path = Path(directory)
    pipeline_label = "auto" if pipeline is None else str(pipeline)

    state_files: dict[str, Path] = {}
    known_processed_by_pipeline: dict[str, set[str]] = {}
    known_computed_by_pipeline: dict[str, dict[str, dict[str, object]]] = {}
    process_functions: dict[str, Callable[[Path], None]] = {}
    selected: list[tuple[Path, str, Callable[[Path], None], str]] = []

    if process_file is not None:
        process_label = f"{process_file.__module__}.{process_file.__name__}"
        state_pipeline_name = "custom"
        state_files[state_pipeline_name] = state_file_path(directory_path, pipeline_name=state_pipeline_name)
        known_processed, known_computed = load_state(state_files[state_pipeline_name])
        known_processed_by_pipeline[state_pipeline_name] = known_processed
        known_computed_by_pipeline[state_pipeline_name] = known_computed
        for file_path in files:
            selected.append((file_path, process_label, process_file, state_pipeline_name))
    elif pipeline == "dummy":
        process_functions["dummy"] = process_file_for_pipeline("dummy")
        state_files["dummy"] = state_file_path(directory_path, pipeline_name="dummy")
        known_processed, known_computed = load_state(state_files["dummy"])
        known_processed_by_pipeline["dummy"] = known_processed
        known_computed_by_pipeline["dummy"] = known_computed
        for file_path in files:
            selected.append((file_path, "dummy", process_functions["dummy"], "dummy"))
    elif pipeline is not None:
        pipeline_name = str(pipeline)
        process_functions[pipeline_name] = process_file_for_pipeline(pipeline_name)
        state_files[pipeline_name] = state_file_path(directory_path, pipeline_name=pipeline_name)
        known_processed, known_computed = load_state(state_files[pipeline_name])
        known_processed_by_pipeline[pipeline_name] = known_processed
        known_computed_by_pipeline[pipeline_name] = known_computed
        for file_path in files:
            if pipeline_name_for_file(file_path) == pipeline_name:
                selected.append((file_path, pipeline_name, process_functions[pipeline_name], pipeline_name))
    else:
        for file_path in files:
            resolved_pipeline = pipeline_name_for_file(file_path)
            if resolved_pipeline is None:
                continue
            if resolved_pipeline not in process_functions:
                process_functions[resolved_pipeline] = process_file_for_pipeline(resolved_pipeline)
            if resolved_pipeline not in state_files:
                state_files[resolved_pipeline] = state_file_path(directory_path, pipeline_name=resolved_pipeline)
                known_processed, known_computed = load_state(state_files[resolved_pipeline])
                known_processed_by_pipeline[resolved_pipeline] = known_processed
                known_computed_by_pipeline[resolved_pipeline] = known_computed
            selected.append(
                (
                    file_path,
                    resolved_pipeline,
                    process_functions[resolved_pipeline],
                    resolved_pipeline,
                )
            )

    results = BatwindPipeResults(
        directory=directory_path,
        recursive=recursive,
        noclobber=noclobber,
        discovered_files=[item[0] for item in selected],
        computed_results={},
        state_file=None if len(state_files) != 1 else next(iter(state_files.values())),
    )

    for pipeline_name, pipeline_results in known_computed_by_pipeline.items():
        if pipeline_name == "custom":
            results.computed_results.update(pipeline_results)
            continue
        for file_key, payload in pipeline_results.items():
            results.computed_results[file_key] = payload

    log.debug(
        "run_batwind_pipe discovered=%s directory=%s noclobber=%s pipeline=%s recursive=%s",
        len(selected),
        directory_path,
        noclobber,
        pipeline_label,
        recursive,
    )

    if not selected:
        for state_pipeline_name, state_file in state_files.items():
            save_state(
                state_file,
                processed_keys=known_processed_by_pipeline[state_pipeline_name],
                computed_results=known_computed_by_pipeline[state_pipeline_name],
                json_warn_bytes=int(json_warn_bytes),
            )
        log.debug("run_batwind_pipe complete with no selected files")
        return results

    recorder = logging.getLogger("recorder")
    recorder.setLevel(logging.DEBUG)
    artifacts_root = directory_path / "batwind-pipe.artifacts"

    for file_path, process_label, active_process_file, state_pipeline_name in selected:
        file_key = relative_file_key(file_path, base_dir=directory_path)
        processed_keys = known_processed_by_pipeline[state_pipeline_name]

        if noclobber and file_key in processed_keys:
            results.skipped_files.append(file_path)
            log.debug("batwind_pipe.skip_processed | file=%s", file_path.name)
            continue

        file_results: dict[str, object] = {
            "meta": {
                "input_file": str(file_path.resolve()),
                "pipeline": process_label,
                "start_time_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }
        }

        if include_file_hash:
            file_results["meta"]["file_hash_sha256"] = sha256_file(file_path)

        recorder_handler = BatwindRecordHandler(
            file_results,
            file_key=file_key,
            artifacts_root=artifacts_root,
            array_offload_min_bytes=array_offload_min_bytes,
        )
        recorder.addHandler(recorder_handler)

        failure: Exception | None = None
        failure_traceback = None

        try:
            active_process_file(file_path)
        except Exception as exc:
            failure = exc
            failure_traceback = exc.__traceback__
            results.failed_files.append(file_path)
            file_results["meta"]["status"] = "failed"
            file_results["meta"]["error"] = str(exc)
            log.error("batwind-pipe file failed: %s (%s)", file_path.name, exc)
        else:
            file_results["meta"]["status"] = "processed"
            results.processed_files.append(file_path)
            processed_keys.add(file_key)
        finally:
            meta = file_results.get("meta")
            if isinstance(meta, dict):
                meta["end_time_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            recorder.removeHandler(recorder_handler)
            recorder_handler.close()
            results.computed_results[file_key] = file_results
            known_computed_by_pipeline[state_pipeline_name][file_key] = file_results
            save_state(
                state_files[state_pipeline_name],
                processed_keys=processed_keys,
                computed_results=known_computed_by_pipeline[state_pipeline_name],
                json_warn_bytes=int(json_warn_bytes),
            )

        if failure is not None and fail_fast:
            raise failure.with_traceback(failure_traceback)

    log.debug(
        "run_batwind_pipe complete processed=%d failed=%d skipped=%d",
        len(results.processed_files),
        len(results.failed_files),
        len(results.skipped_files),
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    """
    Build the `batwind-pipe` CLI argument parser.
    """
    parser = argparse.ArgumentParser(description="Run the batwind generic pipeline.")
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory to scan for input files (default: current directory).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search subdirectories for input files.",
    )
    parser.add_argument(
        "--pipeline",
        default=None,
        choices=("dummy", "slice", "shell", "ua", "volume"),
        help="Built-in per-file pipeline to run (default: auto by filename prefix).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logging level (default: INFO).",
    )
    parser.add_argument(
        "--record-log-level",
        default="WARNING",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Recorder logger level for stream output (default: WARNING).",
    )
    parser.add_argument(
        "--noclobber",
        action="store_true",
        help="Skip files already listed in the batwind-pipe state file.",
    )
    parser.add_argument(
        "--file-hash",
        action="store_true",
        help="Include SHA-256 hash of each input file in per-file metadata.",
    )
    parser.add_argument(
        "--array-offload-min-bytes",
        type=int,
        default=DEFAULT_ARRAY_OFFLOAD_MIN_BYTES,
        help="Offload recorded NumPy arrays at or above this byte size to .npy artifacts.",
    )
    parser.add_argument(
        "--json-warn-bytes",
        type=int,
        default=DEFAULT_JSON_WARN_BYTES,
        help="Warn if a per-pipeline state JSON is at or above this byte size (0 disables).",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop the run on the first per-file pipeline failure.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point for `batwind-pipe`.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logger(str(args.log_level))
    configure_recorder(str(args.record_log_level))

    run_batwind_pipe(
        args.directory,
        pipeline=args.pipeline,
        recursive=bool(args.recursive),
        noclobber=bool(args.noclobber),
        include_file_hash=bool(args.file_hash),
        array_offload_min_bytes=int(args.array_offload_min_bytes),
        json_warn_bytes=int(args.json_warn_bytes),
        fail_fast=bool(args.fail_fast),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
