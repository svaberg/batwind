"""A small reader for SWMF-style `PARAM.in` files.
"""

# BATSRUS itself parses these files line by line. Non-command lines are ignored
# until a line starting with `#` is encountered, at which point BATSRUS switches
# into command-specific parsing and consumes a hard-coded number of following
# parameter lines for that command. Sessions are demarcated by `#END` or `#RUN`.
# In the SWMF layer, components are additional structure layered on top.
#
# This reader is intentionally more permissive: it flattens resolvable
# `#INCLUDE` statements, preserves sessions/components/duplicate commands, and
# stores command blocks for inspection without hard-coding command arity.


from __future__ import annotations

from collections import OrderedDict
import logging
from pathlib import Path

from scipy.constants import day
from batwind.constants import SOLAR_MASS_KG
from batwind.constants import SOLAR_RADIUS_M
log = logging.getLogger(__name__)


def _strip_lines(path) -> list[str]:
    """Read a text file and return stripped lines."""
    with open(path, encoding="utf-8") as stream:
        return [line.strip() for line in stream]


def flatten_includes(file_path) -> list[str]:
    """Flatten resolvable `#INCLUDE` directives into one flat line stream."""
    path = Path(file_path)
    content = _strip_lines(path)
    flat_lines: list[str] = []
    line_id = 0

    while line_id < len(content):
        line = content[line_id]
        if line.startswith("#INCLUDE") and line_id + 1 < len(content):
            child_name = content[line_id + 1].split()[0]
            child_path = path.parent / child_name
            if child_path.exists():
                log.debug("flatten_includes expanding %s -> %s", path.name, child_path.name)
                flat_lines.extend(flatten_includes(child_path))
                line_id += 2
                continue
        flat_lines.append(line)
        line_id += 1

    return flat_lines


def find_param_in(file_path):
    """Find the nearest `PARAM.in`/`param.in` beside or above a data file."""
    start = Path(file_path)
    search_root = start if start.is_dir() else start.parent
    search_dirs = [search_root]
    if search_root.parent != search_root:
        search_dirs.append(search_root.parent)
    if search_root.parent.parent != search_root.parent:
        search_dirs.append(search_root.parent.parent)
    for directory in search_root.parents:
        if directory not in search_dirs:
            search_dirs.append(directory)
    for directory in search_dirs:
        for name in ("PARAM.in", "param.in"):
            candidate = directory / name
            if candidate.exists():
                log.info("Using PARAM.in %s", candidate)
                return candidate
    log.debug("No nearby PARAM.in found for %s", file_path)
    return None


def _new_session():
    """Create one parsed session container."""
    return OrderedDict()


def parse_sessions(flat_lines) -> list[OrderedDict]:
    """Parse flat config lines into sessions, components, commands, and blocks."""
    sessions = [_new_session()]
    session = sessions[-1]
    component_name = "root"
    current_command = None

    for line in flat_lines:
        if not line:
            continue
        if line.startswith("!"):
            continue
        if line.lower().startswith("begin session:"):
            continue

        if line.startswith("#BEGIN_COMP"):
            tokens = line.split()
            component_name = tokens[1] if len(tokens) > 1 else "root"
            current_command = None
            session.setdefault(component_name, OrderedDict())
            continue

        if line.startswith("#END_COMP"):
            component_name = "root"
            current_command = None
            continue

        if line.startswith("#RUN") or line.startswith("#END"):
            if session:
                session = _new_session()
                sessions.append(session)
            component_name = "root"
            current_command = None
            continue

        if line.startswith("#"):
            current_command = line.split()[0]
            component = session.setdefault(component_name, OrderedDict())
            component.setdefault(current_command, []).append([])
            continue

        if current_command is None:
            continue

        component = session.setdefault(component_name, OrderedDict())
        component[current_command][-1].append(line)

    if sessions and not sessions[-1]:
        sessions.pop()
    return sessions


def _parse_scalar_token(text):
    """Parse one first-token scalar using SWMF-style `T/F/int/float/string` rules."""
    token = str(text).split()[0]
    upper = token.upper()
    if upper == "T":
        return True
    if upper == "F":
        return False
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        return token


def _parse_value_text(text):
    """Parse a full value field without discarding embedded spaces."""
    value_text = str(text).strip()
    upper = value_text.upper()
    if upper == "T":
        return True
    if upper == "F":
        return False
    try:
        return int(value_text)
    except ValueError:
        pass
    try:
        return float(value_text)
    except ValueError:
        return value_text


def _split_value_and_label(line: str) -> tuple[str, str]:
    """Split one SWMF parameter line into a value field and a trailing label."""
    text = str(line).strip()
    for index in range(len(text) - 1):
        if text[index].isspace() and text[index + 1].isspace():
            value = text[:index].strip()
            label = text[index + 1:].strip()
            while label and label[0].isspace():
                label = label[1:]
            return value, label
    tokens = text.split()
    if len(tokens) <= 1:
        return text, ""
    return tokens[0], " ".join(tokens[1:])


class ParamIn:
    """Read and query one `PARAM.in` file as sessions, components, and commands."""

    def __init__(self, file_path):
        """Parse a `PARAM.in` file immediately."""
        self.path = Path(file_path)
        self.flat_lines = flatten_includes(self.path)
        self.sessions = parse_sessions(self.flat_lines)
        log.debug(
            "ParamIn.__init__ path=%s flat_lines=%d sessions=%d",
            self.path,
            len(self.flat_lines),
            len(self.sessions),
        )

    @classmethod
    def from_file(cls, file_path):
        """Construct a parsed config from disk."""
        return cls(file_path)

    def num_sessions(self) -> int:
        """Return the number of parsed sessions."""
        return len(self.sessions)

    def get_commands(self, command, *, component="root", session=None) -> list[list[str]]:
        """Return all blocks for one command in one session/component."""
        if session is None:
            blocks: list[list[str]] = []
            for session_data in self.sessions:
                component_data = session_data.get(component, {})
                blocks.extend(component_data.get(command, ()))
            return blocks
        session_data = self.sessions[int(session)]
        component_data = session_data.get(component, {})
        return list(component_data.get(command, ()))

    def get_command(self, command, *, component="root", session=None, occurrence=-1) -> list[str] | None:
        """Return one command block, defaulting to the most recent occurrence."""
        blocks = self.get_commands(command, component=component, session=session)
        if not blocks:
            return None
        return blocks[occurrence]

    def get_param_line(
        self,
        command,
        index,
        *,
        component="root",
        session=None,
        occurrence=-1,
    ) -> str | None:
        """Return one raw parameter line from a command block."""
        block = self.get_command(command, component=component, session=session, occurrence=occurrence)
        if block is None:
            return None
        if index < 0 or index >= len(block):
            return None
        return block[index]

    def get_param(
        self,
        command,
        index,
        *,
        component="root",
        session=None,
        occurrence=-1,
    ):
        """Return one parsed parameter value from the first token on the line."""
        line = self.get_param_line(command, index, component=component, session=session, occurrence=occurrence)
        if line is None:
            return None
        return _parse_scalar_token(line)

    def get_named_params(self, command, *, component="root", session=None, occurrence=-1) -> OrderedDict:
        """Return an ordered mapping from trailing labels to parsed values."""
        block = self.get_command(command, component=component, session=session, occurrence=occurrence)
        out = OrderedDict()
        if block is None:
            return out
        for line in block:
            value_text, label = _split_value_and_label(line)
            if not value_text:
                continue
            key = label or f"param_{len(out)}"
            out[key] = _parse_value_text(value_text)
        return out

    def stellar_params(self) -> OrderedDict:
        """Return parsed stellar parameters from `#STAR`, if present."""
        params = self.get_named_params("#STAR")
        if not params:
            return OrderedDict()

        name = params.get("NameStar")
        unlabeled_numeric_values = [
            float(value)
            for key, value in params.items()
            if key.startswith("param_") and isinstance(value, (int, float)) and not isinstance(value, bool)
        ]

        def pick_numeric_value(index: int, *labels: str):
            for label in labels:
                value = params.get(label)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return float(value)
            if index < len(unlabeled_numeric_values):
                return unlabeled_numeric_values[index]
            return None

        radius_value = pick_numeric_value(0, "RadiusStar", "RadiusStar (in Solar radii)")
        mass_value = pick_numeric_value(1, "MassStar", "MassStar (in Solar masses)")
        period_value = pick_numeric_value(2, "RotationPeriodStar", "RotationPeriodStar (in days)")
        if radius_value is None or mass_value is None or period_value is None:
            return OrderedDict()

        out = OrderedDict()
        if isinstance(name, str):
            out["Star_name"] = name
        out["Star_radius_m"] = float(radius_value) * SOLAR_RADIUS_M
        out["Star_mass_kg"] = float(mass_value) * SOLAR_MASS_KG
        out["Star_rotational_period_s"] = float(period_value) * day
        out["Star_rotation_rate_rad_s"] = 2.0 * 3.141592653589793 / out["Star_rotational_period_s"]
        log.debug("stellar_params parsed keys=%s", tuple(out))
        return out

    def __str__(self) -> str:
        """Summarize the parsed config briefly."""
        component_count = sum(len(session) for session in self.sessions)
        command_count = 0
        for session in self.sessions:
            for component in session.values():
                command_count += sum(len(blocks) for blocks in component.values())
        return (
            f"ParamIn(path={self.path.name!r}, sessions={self.num_sessions()}, "
            f"components={component_count}, command_blocks={command_count})"
        )


def stellar_aux_from_nearby_param_in(file_path) -> OrderedDict:
    """Read stellar aux values from the nearest available `PARAM.in`."""
    param_path = find_param_in(file_path)
    if param_path is None:
        return OrderedDict()
    return ParamIn.from_file(param_path).stellar_params()
