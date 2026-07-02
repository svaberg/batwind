"""Generic `batwind-pipe` orchestration CLI.
"""

# It discovers supported input files in a working directory and runs a
# per-file pipeline handler. Built-in handlers are `dummy`, `log`, `slice`,
# `shell`, `ua`, and `volume`.

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
from batwind.pipelines.movie import write_recorded_png_movies
from batwind.pipelines.recorder import load_state
from batwind.pipelines.recorder import relative_file_key
from batwind.pipelines.recorder import save_state
from batwind.pipelines.recorder import sha256_file
from batwind.pipelines.recorder import state_file_path

log = logging.getLogger(__name__)
PIPELINE_LOG_FORMAT = "[%(levelname)s] %(pipeline_source)s %(message)s"
PIPELINE_COLOR_LOG_FORMAT = "%(log_color)s[%(levelname)s]%(reset)s %(pipeline_source)s %(message)s"
COMPONENT_NAMES = ("SC", "IH", "GM")
SUPPORTED_INPUT_SUFFIXES = {".plt", ".dat", ".bin", ".log"}


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
    internal_level = level if level <= logging.DEBUG else logging.WARNING
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

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

    # Keep human-facing pipeline progress at the requested level while hiding
    # internal graph/recipe chatter unless DEBUG was explicitly requested.
    logging.getLogger("batwind").setLevel(internal_level)
    logging.getLogger("batwind.pipelines").setLevel(level)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("griblet").setLevel(internal_level)


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
    if recursive:
        log.info("Scanning %s recursively...", base)
        paths = base.rglob("*")
    else:
        log.info("Scanning %s...", base)
        scan_dirs = [base]
        for component_name in COMPONENT_NAMES:
            io2_dir = base / component_name / "IO2"
            if io2_dir.is_dir():
                log.info("Entering %s...", io2_dir)
                scan_dirs.append(io2_dir)
        paths = [path for scan_dir in scan_dirs for path in scan_dir.iterdir()]
    files = [path for path in paths if path.is_file() and path.suffix.lower() in SUPPORTED_INPUT_SUFFIXES]
    return sorted(files)


def tracking_root_for_file(directory: str | Path, file_path: str | Path) -> Path:
    """
    Return the state/artifact root for one discovered file.
    """
    directory_path = Path(directory).resolve()
    path = Path(file_path).resolve()
    try:
        relative_path = path.relative_to(directory_path)
    except ValueError:
        return directory_path
    if len(relative_path.parts) >= 3 and relative_path.parts[0] in COMPONENT_NAMES and relative_path.parts[1] == "IO2":
        return directory_path / relative_path.parts[0] / relative_path.parts[1]
    return directory_path


def display_file_key_for_directory(
    file_key: str,
    *,
    tracking_root: str | Path,
    directory: str | Path,
) -> str:
    """
    Map one tracked file key onto the directory-level results namespace.
    """
    tracking_root_path = Path(tracking_root).resolve()
    directory_path = Path(directory).resolve()
    if tracking_root_path == directory_path:
        return str(file_key)
    return relative_file_key(tracking_root_path / str(file_key), base_dir=directory_path)


def parsed_utc_timestamp(value: object) -> datetime | None:
    """
    Parse one stored UTC timestamp string from recorder state.
    """
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def recorded_artifact_paths(recorded_value: object, *, tracking_root: Path) -> list[Path]:
    """
    Collect relative artifact paths recorded for one processed file.
    """
    out: list[Path] = []

    def artifact_path(text: str) -> Path | None:
        path = Path(text)
        if path.is_absolute():
            return None
        if path.suffix == "" and len(path.parts) == 1:
            return None
        return tracking_root / path

    def visit(value: object) -> None:
        if isinstance(value, dict):
            path_value = value.get("value")
            if isinstance(path_value, str):
                path = artifact_path(path_value)
                if path is not None:
                    out.append(path)
            artifact_value = value.get("path")
            if isinstance(artifact_value, str):
                path = artifact_path(artifact_value)
                if path is not None:
                    out.append(path)
            for item in value.values():
                visit(item)
            return
        if isinstance(value, list):
            for item in value:
                visit(item)

    visit(recorded_value)
    return out


def stale_processed_reason(
    file_path: Path,
    file_results: object,
    *,
    tracking_root: Path,
) -> str | None:
    """
    Return one reprocess reason when recorded state for a file is stale.
    """
    if not isinstance(file_results, dict):
        return "missing recorded result"
    meta = file_results.get("meta")
    if not isinstance(meta, dict):
        return "missing recorded metadata"
    if meta.get("status") != "processed":
        return "recorded status is not processed"
    processed_at = parsed_utc_timestamp(meta.get("end_time_utc"))
    if processed_at is None:
        return "missing recorded completion time"
    if file_path.stat().st_mtime_ns > int(processed_at.timestamp() * 1_000_000_000):
        return "source file is newer"
    for artifact_path in recorded_artifact_paths(file_results, tracking_root=tracking_root):
        if not artifact_path.exists():
            return f"missing recorded output {artifact_path.relative_to(tracking_root).as_posix()}"
    return None


def pipeline_name_for_file(file_path: str | Path) -> str | None:
    """
    Infer built-in pipeline from the input filename prefix.
    """
    file_name = Path(file_path).name.lower()
    if file_name.startswith("log_n") and file_name.endswith(".log"):
        return "log"
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
    if pipeline_name == "log":
        from batwind.pipelines.log import process_log_file

        return process_log_file
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
    noclobber: bool = True,
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
    directory_path = Path(directory).resolve()
    pipeline_label = "auto" if pipeline is None else str(pipeline)

    state_files: dict[tuple[Path, str], Path] = {}
    known_processed_by_state: dict[tuple[Path, str], set[str]] = {}
    known_computed_by_state: dict[tuple[Path, str], dict[str, dict[str, object]]] = {}
    process_functions: dict[str, Callable[[Path], None]] = {}
    selected: list[tuple[Path, str, Callable[[Path], None], Path, str]] = []

    def ensure_state_loaded(tracking_root: Path, state_pipeline_name: str) -> None:
        state_key = (tracking_root.resolve(), state_pipeline_name)
        if state_key in state_files:
            return
        state_files[state_key] = state_file_path(tracking_root, pipeline_name=state_pipeline_name)
        known_processed, known_computed = load_state(state_files[state_key])
        known_processed_by_state[state_key] = known_processed
        known_computed_by_state[state_key] = known_computed

    if process_file is not None:
        process_label = f"{process_file.__module__}.{process_file.__name__}"
        state_pipeline_name = "custom"
        for file_path in files:
            tracking_root = tracking_root_for_file(directory_path, file_path)
            ensure_state_loaded(tracking_root, state_pipeline_name)
            selected.append((file_path, process_label, process_file, tracking_root, state_pipeline_name))
        if not state_files:
            ensure_state_loaded(directory_path, state_pipeline_name)
    elif pipeline == "dummy":
        process_functions["dummy"] = process_file_for_pipeline("dummy")
        for file_path in files:
            tracking_root = tracking_root_for_file(directory_path, file_path)
            ensure_state_loaded(tracking_root, "dummy")
            selected.append((file_path, "dummy", process_functions["dummy"], tracking_root, "dummy"))
        if not state_files:
            ensure_state_loaded(directory_path, "dummy")
    elif pipeline is not None:
        pipeline_name = str(pipeline)
        process_functions[pipeline_name] = process_file_for_pipeline(pipeline_name)
        for file_path in files:
            if pipeline_name_for_file(file_path) == pipeline_name:
                tracking_root = tracking_root_for_file(directory_path, file_path)
                ensure_state_loaded(tracking_root, pipeline_name)
                selected.append((file_path, pipeline_name, process_functions[pipeline_name], tracking_root, pipeline_name))
        if not state_files:
            ensure_state_loaded(directory_path, pipeline_name)
    else:
        for file_path in files:
            resolved_pipeline = pipeline_name_for_file(file_path)
            if resolved_pipeline is None:
                continue
            if resolved_pipeline not in process_functions:
                process_functions[resolved_pipeline] = process_file_for_pipeline(resolved_pipeline)
            tracking_root = tracking_root_for_file(directory_path, file_path)
            ensure_state_loaded(tracking_root, resolved_pipeline)
            selected.append(
                (
                    file_path,
                    resolved_pipeline,
                    process_functions[resolved_pipeline],
                    tracking_root,
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

    for (tracking_root, _state_pipeline_name), pipeline_results in known_computed_by_state.items():
        for file_key, payload in pipeline_results.items():
            display_key = display_file_key_for_directory(file_key, tracking_root=tracking_root, directory=directory_path)
            results.computed_results[display_key] = payload

    log.debug(
        "run_batwind_pipe discovered=%s directory=%s noclobber=%s pipeline=%s recursive=%s",
        len(selected),
        directory_path,
        noclobber,
        pipeline_label,
        recursive,
    )

    if not selected:
        for state_key, state_file in state_files.items():
            tracking_root, state_pipeline_name = state_key
            save_state(
                state_file,
                processed_keys=known_processed_by_state[state_key],
                computed_results=known_computed_by_state[state_key],
                json_warn_bytes=int(json_warn_bytes),
            )
            if state_pipeline_name != "log":
                results.movie_outputs.extend(
                    write_recorded_png_movies(tracking_root, known_computed_by_state[state_key])
                )
        log.debug("run_batwind_pipe complete with no selected files")
        return results

    recorder = logging.getLogger("recorder")
    recorder.setLevel(logging.DEBUG)

    for file_path, process_label, active_process_file, tracking_root, state_pipeline_name in selected:
        state_key = (tracking_root.resolve(), state_pipeline_name)
        file_key = relative_file_key(file_path, base_dir=tracking_root)
        display_file_key = display_file_key_for_directory(file_key, tracking_root=tracking_root, directory=directory_path)
        processed_keys = known_processed_by_state[state_key]
        stale_reason = stale_processed_reason(
            file_path,
            known_computed_by_state[state_key].get(file_key),
            tracking_root=tracking_root,
        )

        if noclobber and file_key in processed_keys and stale_reason is None:
            results.skipped_files.append(file_path)
            log.debug("batwind_pipe.skip_processed | file=%s", file_path.name)
            continue
        if file_key in processed_keys:
            log.info("Reprocessing %s (%s)...", display_file_key, stale_reason)
        else:
            log.info("Processing %s...", display_file_key)

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
            artifacts_root=tracking_root / "batwind-pipe.artifacts",
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
            results.computed_results[display_file_key] = file_results
            known_computed_by_state[state_key][file_key] = file_results
            save_state(
                state_files[state_key],
                processed_keys=processed_keys,
                computed_results=known_computed_by_state[state_key],
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
    for (tracking_root, state_pipeline_name), pipeline_results in known_computed_by_state.items():
        if state_pipeline_name != "log":
            results.movie_outputs.extend(write_recorded_png_movies(tracking_root, pipeline_results))
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
        choices=("dummy", "log", "slice", "shell", "ua", "volume"),
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
        "--clobber",
        action="store_true",
        help="Reprocess files already listed in the batwind-pipe state file.",
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
        noclobber=not bool(args.clobber),
        include_file_hash=bool(args.file_hash),
        array_offload_min_bytes=int(args.array_offload_min_bytes),
        json_warn_bytes=int(args.json_warn_bytes),
        fail_fast=bool(args.fail_fast),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
