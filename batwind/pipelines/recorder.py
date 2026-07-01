"""Recorder-backed state persistence for pipeline runs."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import hashlib
import json
from json.encoder import INFINITY
from json.encoder import _make_iterencode
from json.encoder import encode_basestring
from json.encoder import encode_basestring_ascii
import logging
import math
import numpy as np
from pathlib import Path
import re

DEFAULT_ARRAY_OFFLOAD_MIN_BYTES = 1_000_000
DEFAULT_JSON_WARN_BYTES = 10_000_000

_record_pattern = re.compile(r"^([A-Za-z0-9_]+)\b")
_safe_name_pattern = re.compile(r"[^A-Za-z0-9_.-]+")

log = logging.getLogger(__name__)


class ScientificFloatEncoder(json.JSONEncoder):
    """
    JSON encoder that writes finite floats in scientific notation.
    Used by: `batwind/pipelines/recorder.py`
    """

    def iterencode(self, o, _one_shot=False):
        """
        Encode JSON while formatting finite floats as scientific literals.
        Used by: `batwind/pipelines/recorder.py`
        """
        markers = {} if self.check_circular else None
        encoder = encode_basestring_ascii if self.ensure_ascii else encode_basestring
        indent = self.indent
        if isinstance(indent, int):
            indent = " " * indent

        def floatstr(
            value,
            allow_nan=self.allow_nan,
            inf=INFINITY,
            neginf=-INFINITY,
        ):
            """Render finite floats in scientific notation for JSON encoding."""
            if math.isnan(value):
                text = "NaN"
            elif value == inf:
                text = "Infinity"
            elif value == neginf:
                text = "-Infinity"
            else:
                return format(value, ".16e")
            if not allow_nan:
                raise ValueError("Out of range float values are not JSON compliant: " + repr(value))
            return text

        iterencode = _make_iterencode(
            markers,
            self.default,
            encoder,
            indent,
            floatstr,
            self.key_separator,
            self.item_separator,
            self.sort_keys,
            self.skipkeys,
            _one_shot,
        )
        return iterencode(o, 0)


@dataclass
class BatwindPipeResults:
    """
    Minimal results container for one `batwind-pipe` run.
    Used by: `batwind/pipelines/batwind_pipe.py`
    """

    directory: Path
    recursive: bool
    noclobber: bool
    discovered_files: list[Path] = field(default_factory=list)
    processed_files: list[Path] = field(default_factory=list)
    failed_files: list[Path] = field(default_factory=list)
    skipped_files: list[Path] = field(default_factory=list)
    movie_outputs: list[Path] = field(default_factory=list)
    computed_results: dict[str, dict[str, object]] = field(default_factory=dict)
    state_file: Path | None = None


def safe_name(text: str) -> str:
    """
    Convert arbitrary text to a filesystem-safe token.
    Used by: `batwind/pipelines/recorder.py`, `batwind/pipelines/batwind_pipe.py`
    """

    return _safe_name_pattern.sub("_", str(text)).strip("._") or "value"


def relative_file_key(file_path: str | Path, *, base_dir: str | Path) -> str:
    """
    Stable relative key for a processed file inside the tracked directory.
    Used by: `batwind/pipelines/recorder.py`, `batwind/pipelines/batwind_pipe.py`
    """

    return Path(file_path).resolve().relative_to(Path(base_dir).resolve()).as_posix()


def state_file_path(directory: str | Path, *, pipeline_name: str) -> Path:
    """
    Default per-pipeline state-file path for processed input tracking.
    Used by: `batwind/pipelines/recorder.py`, `batwind/pipelines/batwind_pipe.py`
    """

    return Path(directory) / f"batwind-pipe.{safe_name(pipeline_name)}.processed.json"


def sha256_file(path: str | Path) -> str:
    """
    Compute SHA-256 for a file.
    Used by: `batwind/pipelines/recorder.py`, `batwind/pipelines/batwind_pipe.py`
    """

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_record_payload(record: logging.LogRecord) -> tuple[str, object] | None:
    """
    Parse `<key> ...` logger template and args into a key/value payload.
    Used by: `batwind/pipelines/recorder.py`
    """

    if not isinstance(record.msg, str):
        return None
    match = _record_pattern.match(record.msg)
    if match is None:
        return None
    key = match.group(1)
    args = record.args
    if isinstance(args, tuple):
        if len(args) == 0:
            return key, None
        if len(args) == 1:
            return key, args[0]
        return key, list(args)
    if isinstance(args, dict):
        return key, dict(args)
    if args is None:
        return key, None
    return key, args


def normalize_recorded_value(
    value: object,
    *,
    file_key: str,
    field_key: str,
    artifacts_root: str | Path,
    array_offload_min_bytes: int,
) -> object:
    """
    Convert recorded values into JSON-safe payloads with array offloading support.
    Used by: `batwind/pipelines/recorder.py`
    """

    if isinstance(value, np.ndarray):
        if value.nbytes >= int(array_offload_min_bytes):
            file_token = safe_name(str(file_key).replace("/", "__"))
            field_token = safe_name(field_key)
            rel_path = f"batwind-pipe.artifacts/{file_token}__{field_token}.npy"
            out_path = Path(artifacts_root).parent / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, value)
            return {"path": rel_path}
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, list):
        return [
            normalize_recorded_value(
                item,
                file_key=file_key,
                field_key=field_key,
                artifacts_root=artifacts_root,
                array_offload_min_bytes=array_offload_min_bytes,
            )
            for item in value
        ]
    if isinstance(value, tuple):
        return [
            normalize_recorded_value(
                item,
                file_key=file_key,
                field_key=field_key,
                artifacts_root=artifacts_root,
                array_offload_min_bytes=array_offload_min_bytes,
            )
            for item in value
        ]
    if isinstance(value, dict):
        return {
            str(key): normalize_recorded_value(
                item,
                file_key=file_key,
                field_key=field_key,
                artifacts_root=artifacts_root,
                array_offload_min_bytes=array_offload_min_bytes,
            )
            for key, item in value.items()
        }
    return value


class BatwindRecordHandler(logging.Handler):
    """
    Capture `sw_record` payloads from log records into a per-file results mapping.
    Used by: `batwind/pipelines/recorder.py`, `batwind/pipelines/batwind_pipe.py`
    """

    def __init__(
        self,
        target: dict[str, object],
        *,
        file_key: str,
        artifacts_root: str | Path,
        array_offload_min_bytes: int = DEFAULT_ARRAY_OFFLOAD_MIN_BYTES,
    ):
        """
        Initialize a handler that writes captured record payloads into `target`.
        Used by: `batwind/pipelines/recorder.py`, `batwind/pipelines/batwind_pipe.py`
        """

        super().__init__(level=logging.NOTSET)
        self.target = target
        self.file_key = file_key
        self.artifacts_root = Path(artifacts_root)
        self.array_offload_min_bytes = int(array_offload_min_bytes)

    def emit(self, record: logging.LogRecord) -> None:
        """
        Pull record payloads from logger template/args and store them by key.
        Used by: `batwind/pipelines/recorder.py`, `batwind/pipelines/batwind_pipe.py`
        """

        parsed = parse_record_payload(record)
        if parsed is None:
            return
        key, value = parsed
        module_name = record.name[len("recorder."):] if record.name.startswith("recorder.") else record.name
        normalized = normalize_recorded_value(
            value,
            file_key=self.file_key,
            field_key=key,
            artifacts_root=self.artifacts_root,
            array_offload_min_bytes=self.array_offload_min_bytes,
        )
        self.target[key] = {
            "value": normalized,
            "source": {
                "module": module_name,
                "function": record.funcName,
                "line": int(record.lineno),
            },
        }


def load_state(state_file: str | Path) -> tuple[set[str], dict[str, dict[str, object]]]:
    """
    Load processed-file keys and computed results from the recorder state file.
    Used by: `batwind/pipelines/recorder.py`, `batwind/pipelines/batwind_pipe.py`
    """

    path = Path(state_file)
    if not path.exists():
        log.debug("load_state missing path=%s", path)
        return set(), {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        log.info("Could not load recorder state %s (%s); starting from empty state.", path, exc)
        return set(), {}
    files = payload.get("processed_files", [])
    processed_keys = {str(item) for item in files} if isinstance(files, list) else set()
    computed = payload.get("computed_results", {})
    if isinstance(computed, dict):
        log.debug("load_state path=%s processed=%d computed=%d", path, len(processed_keys), len(computed))
        return processed_keys, {str(key): value for key, value in computed.items() if isinstance(value, dict)}
    return processed_keys, {}


def load_state_payload(state_file: str | Path) -> dict[str, object]:
    """
    Load a recorder state payload as a raw mapping for inspection tools.
    Used by: `batwind/pipelines/batwind_pipe_results.py`
    """

    path = Path(state_file)
    if not path.exists():
        log.debug("load_state_payload missing path=%s", path)
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        log.info("Could not load recorder payload %s (%s); returning empty payload.", path, exc)
        return {}
    if isinstance(payload, dict):
        log.debug("load_state_payload path=%s keys=%d", path, len(payload))
        return payload
    return {}


def save_state(
    state_file: str | Path,
    *,
    processed_keys: set[str],
    computed_results: dict[str, dict[str, object]],
    json_warn_bytes: int = DEFAULT_JSON_WARN_BYTES,
) -> None:
    """
    Save processed keys and computed results to the recorder state file.
    Used by: `batwind/pipelines/recorder.py`, `batwind/pipelines/batwind_pipe.py`
    """

    path = Path(state_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "processed_files": sorted(processed_keys),
        "computed_results": computed_results,
    }
    payload_text = json.dumps(payload, indent=2, cls=ScientificFloatEncoder)
    payload_size_bytes = len(payload_text.encode("utf-8"))
    if int(json_warn_bytes) > 0 and payload_size_bytes >= int(json_warn_bytes):
        log.warning(
            "batwind-pipe state file is large: %d bytes (threshold=%d bytes) at %s",
            payload_size_bytes,
            int(json_warn_bytes),
            path,
        )
    path.write_text(payload_text)
    log.debug(
        "save_state path=%s processed=%d computed=%d size_bytes=%d",
        path,
        len(processed_keys),
        len(computed_results),
        payload_size_bytes,
    )
