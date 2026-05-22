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
    search_root = Path(file_path).parent
    for directory in (search_root, search_root.parent, search_root.parent.parent):
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


def parse_parameter_value(text):
    """Parse one PARAM.in value field using SWMF-style scalar rules."""
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
    for separator in ("\t", "   "):
        index = text.find(separator)
        if index >= 0:
            value = text[:index].strip()
            label = text[index + len(separator):].strip()
            return value, label
    return text, ""


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
        return parse_parameter_value(str(line).split()[0])

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
            out[key] = parse_parameter_value(value_text)
        return out

    def stellar_params(self) -> OrderedDict:
        """Return parsed stellar parameters from `#STAR`, if present."""
        block = self.get_command("#STAR")
        if block is None or len(block) < 4:
            return OrderedDict()

        name_text, _ = _split_value_and_label(block[0])
        radius_text, _ = _split_value_and_label(block[1])
        mass_text, _ = _split_value_and_label(block[2])
        period_text, _ = _split_value_and_label(block[3])

        name = parse_parameter_value(name_text)
        radius_rsun = float(parse_parameter_value(radius_text))
        mass_msun = float(parse_parameter_value(mass_text))
        period_days = float(parse_parameter_value(period_text))
        period_seconds = period_days * day

        return OrderedDict(
            [
                ("Star_name", name),
                ("Star_radius_m", radius_rsun * SOLAR_RADIUS_M),
                ("Star_mass_kg", mass_msun * SOLAR_MASS_KG),
                ("Star_rotational_period_s", period_seconds),
                ("Star_rotation_rate_rad_s", 2.0 * 3.141592653589793 / period_seconds),
            ]
        )

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
