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
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import ClassVar
from typing import Self

from scipy.constants import day
from batwind.constants import SOLAR_MASS_KG
from batwind.constants import SOLAR_RADIUS_M
log = logging.getLogger(__name__)


class ParamCommand:
    """Base class for typed PARAM.in command readers."""

    command: ClassVar[str]

    @classmethod
    def value_fields(
        cls,
        lines: list[str],
        *,
        exact: int | None = None,
        minimum: int | None = None,
    ) -> list[str]:
        """Return ordered value fields, split from any trailing descriptive text."""
        if exact is not None and len(lines) != exact:
            raise ValueError(f"{cls.command} expects exactly {exact} parameter lines, got {len(lines)}")
        if minimum is not None and len(lines) < minimum:
            raise ValueError(f"{cls.command} expects at least {minimum} parameter lines, got {len(lines)}")
        return [_split_value_and_label(line)[0] for line in lines]

    @classmethod
    def first_token_values(
        cls,
        lines: list[str],
        *,
        exact: int | None = None,
        minimum: int | None = None,
    ) -> list[object]:
        """Return ordered first-token parameter values."""
        value_fields = cls.value_fields(lines, exact=exact, minimum=minimum)
        return [parse_parameter_value(str(value_text).split()[0]) for value_text in value_fields]

    @classmethod
    def from_param_in(cls, config: ParamIn, *, component="root", session=None, occurrence=-1) -> Self | None:
        """Parse one command block from a `ParamIn` object."""
        command_line = config.get_command_header(cls.command, component=component, session=session, occurrence=occurrence)
        block = config.get_command(cls.command, component=component, session=session, occurrence=occurrence)
        if block is None:
            return None
        return cls.from_command_lines(command_line, block)

    @classmethod
    def from_command_lines(cls, command_line: str | None, lines: list[str]) -> Self | None:
        """Parse one command header plus payload block."""
        del command_line
        return cls.from_lines(lines)


@dataclass(frozen=True, slots=True)
class StarParams(ParamCommand):
    """Star parameters parsed from one ordered `#STAR` block."""

    command: ClassVar[str] = "#STAR"
    name: str | None
    radius: float
    mass: float
    rotational_period: float
    rotation_rate: float

    @classmethod
    def from_lines(cls, lines: list[str]) -> StarParams | None:
        """Parse one ordered `#STAR` block."""
        value_fields = cls.value_fields(lines, exact=4)
        name = parse_parameter_value(value_fields[0])
        radius_rsun = float(parse_parameter_value(value_fields[1]))
        mass_msun = float(parse_parameter_value(value_fields[2]))
        period_days = float(parse_parameter_value(value_fields[3]))
        period_seconds = period_days * day
        return cls(
            name=name,
            radius=radius_rsun * SOLAR_RADIUS_M,
            mass=mass_msun * SOLAR_MASS_KG,
            rotational_period=period_seconds,
            rotation_rate=2.0 * 3.141592653589793 / period_seconds,
        )

    @classmethod
    def from_command_lines(cls, command_line: str | None, lines: list[str]) -> StarParams | None:
        """Parse one `#STAR` command from either the old or new file form."""
        if len(lines) == 4:
            return cls.from_lines(lines)
        if len(lines) != 3:
            raise ValueError(f"{cls.command} expects either 3 or 4 parameter lines, got {len(lines)}")
        if command_line is None:
            raise ValueError(f"{cls.command} with 3 parameter lines requires a command header line")
        header_tokens = str(command_line).split(maxsplit=1)
        name = None if len(header_tokens) == 1 else parse_parameter_value(header_tokens[1])
        value_fields = cls.value_fields(lines, exact=3)
        radius_rsun = float(parse_parameter_value(value_fields[0]))
        mass_msun = float(parse_parameter_value(value_fields[1]))
        period_days = float(parse_parameter_value(value_fields[2]))
        period_seconds = period_days * day
        return cls(
            name=name,
            radius=radius_rsun * SOLAR_RADIUS_M,
            mass=mass_msun * SOLAR_MASS_KG,
            rotational_period=period_seconds,
            rotation_rate=2.0 * 3.141592653589793 / period_seconds,
        )


@dataclass(frozen=True, slots=True)
class TransitionRegionParams(ParamCommand):
    """Transition-region parameters parsed from one ordered `#TRANSITIONREGION` block."""

    command: ClassVar[str] = "#TRANSITIONREGION"
    do_extend: bool
    temperature: float
    delta_temperature: float | None

    @classmethod
    def from_lines(cls, lines: list[str]) -> TransitionRegionParams | None:
        """Parse one ordered `#TRANSITIONREGION` block."""
        values = cls.first_token_values(lines, minimum=2)

        do_extend = bool(values[0])
        temperature = float(values[1])
        delta_temperature = None
        if do_extend:
            if len(values) < 3:
                raise ValueError(f"{cls.command} with DoExtendTransitionRegion=T expects 3 parameter lines")
            delta_temperature = float(values[2])
        return cls(
            do_extend=do_extend,
            temperature=temperature,
            delta_temperature=delta_temperature,
        )


@dataclass(frozen=True, slots=True)
class PlasmaParams(ParamCommand):
    """Plasma parameters parsed from one single-fluid `#PLASMA` block."""

    command: ClassVar[str] = "#PLASMA"
    fluid_mass_amu: float
    ion_charge_e: float
    electron_temperature_ratio: float

    @classmethod
    def from_lines(cls, lines: list[str]) -> PlasmaParams | None:
        """Parse one single-fluid ordered `#PLASMA` block."""
        values = cls.first_token_values(lines, exact=3)
        return cls(
            fluid_mass_amu=float(values[0]),
            ion_charge_e=float(values[1]),
            electron_temperature_ratio=float(values[2]),
        )


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


def parse_sessions(flat_lines) -> tuple[list[OrderedDict], list[OrderedDict]]:
    """Parse flat config lines into sessions, components, command headers, and blocks."""
    sessions = [_new_session()]
    session_headers = [_new_session()]
    session = sessions[-1]
    header_session = session_headers[-1]
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
            header_session.setdefault(component_name, OrderedDict())
            continue

        if line.startswith("#END_COMP"):
            component_name = "root"
            current_command = None
            continue

        if line.startswith("#RUN") or line.startswith("#END"):
            if session:
                session = _new_session()
                header_session = _new_session()
                sessions.append(session)
                session_headers.append(header_session)
            component_name = "root"
            current_command = None
            continue

        if line.startswith("#"):
            current_command = line.split()[0]
            component = session.setdefault(component_name, OrderedDict())
            header_component = header_session.setdefault(component_name, OrderedDict())
            component.setdefault(current_command, []).append([])
            header_component.setdefault(current_command, []).append(line)
            continue

        if current_command is None:
            continue

        component = session.setdefault(component_name, OrderedDict())
        component[current_command][-1].append(line)

    if sessions and not sessions[-1]:
        sessions.pop()
        session_headers.pop()
    return sessions, session_headers


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
        self.sessions, self.session_headers = parse_sessions(self.flat_lines)
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

    def get_command_headers(self, command, *, component="root", session=None) -> list[str]:
        """Return all full command header lines for one command in one session/component."""
        if session is None:
            headers: list[str] = []
            for session_data in self.session_headers:
                component_data = session_data.get(component, {})
                headers.extend(component_data.get(command, ()))
            return headers
        session_data = self.session_headers[int(session)]
        component_data = session_data.get(component, {})
        return list(component_data.get(command, ()))

    def get_command_header(self, command, *, component="root", session=None, occurrence=-1) -> str | None:
        """Return one full command header line, defaulting to the most recent occurrence."""
        headers = self.get_command_headers(command, component=component, session=session)
        if not headers:
            return None
        return headers[occurrence]

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
