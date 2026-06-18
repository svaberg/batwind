from __future__ import annotations

from collections.abc import Sequence
import logging
import re

import griblet
import numpy as np
from scipy.constants import atomic_mass
from scipy.constants import electron_mass

log = logging.getLogger(__name__)


_UA_MARKUP_RE = re.compile(r"!(?:D|U)(.*?)!N")
_BRACKET_SPECIES_RE = re.compile(r"^\[(.+)\]$")
_SPECIES_INFO = {
    "H": ("neutral", 1.0 * atomic_mass),
    "He": ("neutral", 4.0 * atomic_mass),
    "C": ("neutral", 12.0 * atomic_mass),
    "N": ("neutral", 14.0 * atomic_mass),
    "N(2D)": ("neutral", 14.0 * atomic_mass),
    "O": ("neutral", 16.0 * atomic_mass),
    "Ar": ("neutral", 40.0 * atomic_mass),
    "CO": ("neutral", 28.0 * atomic_mass),
    "N2": ("neutral", 28.0 * atomic_mass),
    "NO": ("neutral", 30.0 * atomic_mass),
    "O2": ("neutral", 32.0 * atomic_mass),
    "CO2": ("neutral", 44.0 * atomic_mass),
    "C+": ("ion", 12.0 * atomic_mass),
    "O+": ("ion", 16.0 * atomic_mass),
    "CO+": ("ion", 28.0 * atomic_mass),
    "N2+": ("ion", 28.0 * atomic_mass),
    "NO+": ("ion", 30.0 * atomic_mass),
    "O2+": ("ion", 32.0 * atomic_mass),
    "CO2+": ("ion", 44.0 * atomic_mass),
    "e-": ("electron", electron_mass),
}


def build_ua_graph(variable_names: Sequence[str]):
    """Build a griblet graph for UA/GITM raw fields."""
    variable_names = tuple(variable_names)
    graph = griblet.Graph()

    _add_direct_alias(graph, variable_names, "Longitude", "Longitude [rad]")
    _add_direct_alias(graph, variable_names, "Latitude", "Latitude [rad]")
    _add_direct_alias(graph, variable_names, "Altitude", "Altitude [m]")
    _add_direct_alias(graph, variable_names, "Temperature", "Temperature [K]")
    _add_direct_alias(graph, variable_names, "eTemperature", "eTemperature [K]")
    _add_direct_alias(graph, variable_names, "iTemperature", "iTemperature [K]")
    _add_direct_alias(graph, variable_names, "Rho", "Rho [1/m^3]")
    _add_direct_alias(graph, variable_names, "Solar Zenith Angle", "Solar Zenith Angle [deg]")
    _add_direct_alias(graph, variable_names, "Local Time (hr)", "Local Time [hr]")

    _add_identity_alias(graph, "Temperature [K]", "Tn [K]")
    _add_identity_alias(graph, "eTemperature [K]", "Te [K]")
    _add_identity_alias(graph, "iTemperature [K]", "Ti [K]")
    _add_identity_alias(graph, "Rho [1/m^3]", "neutral_number_density [1/m^3]")

    _add_deg_to_rad(graph, "Longitude")
    _add_deg_to_rad(graph, "Latitude")
    _add_rad_to_deg(graph, "Longitude")
    _add_rad_to_deg(graph, "Latitude")
    _add_deg_to_rad(graph, "Solar Zenith Angle")

    neutral_mass_fields: list[str] = []
    ion_mass_fields: list[str] = []
    for raw_name in variable_names:
        if raw_name.startswith("[") and raw_name.endswith("]"):
            decoded = _decode_ua_text(raw_name)
            number_density_name = f"{decoded} [1/m^3]"
            _add_direct_alias(graph, variable_names, raw_name, number_density_name)
            kind, mass_kg = _SPECIES_INFO.get(decoded, (None, None))
            if mass_kg is not None:
                mass_density_name = f"{decoded} [kg/m^3]"
                graph.add(
                    mass_density_name,
                    lambda n, mass_kg=mass_kg: mass_kg * np.asarray(n),
                    needs=[number_density_name],
                    cost=0.02,
                    metadata={"description": f"Mass density for {decoded}"},
                )
                if kind == "neutral":
                    neutral_mass_fields.append(mass_density_name)
                elif kind == "ion":
                    ion_mass_fields.append(mass_density_name)
                elif kind == "electron":
                    _add_identity_alias(graph, mass_density_name, "electron_mass_density [kg/m^3]")
            if decoded == "e-":
                _add_identity_alias(graph, number_density_name, "Ne [1/m^3]")
            continue

        velocity_alias = _velocity_alias(raw_name)
        if velocity_alias is not None:
            _add_direct_alias(graph, variable_names, raw_name, f"{velocity_alias} [m/s]")

    if neutral_mass_fields:
        graph.add(
            "neutral_mass_density [kg/m^3]",
            _sum_fields,
            needs=tuple(neutral_mass_fields),
            cost=0.04,
            metadata={"description": "Total neutral mass density from species fields"},
        )
    if ion_mass_fields:
        graph.add(
            "ion_mass_density [kg/m^3]",
            _sum_fields,
            needs=tuple(ion_mass_fields),
            cost=0.04,
            metadata={"description": "Total ion mass density from species fields"},
        )

    return graph


def _add_direct_alias(graph: griblet.Graph, variable_names: Sequence[str], raw_name: str, alias_name: str) -> None:
    if raw_name not in variable_names:
        return
    if raw_name == alias_name:
        return
    graph.add(
        alias_name,
        lambda x: x,
        needs=[raw_name],
        cost=0.01,
        metadata={"description": f"UA alias for {raw_name}"},
    )


def _add_identity_alias(graph: griblet.Graph, source_name: str, alias_name: str) -> None:
    graph.add(
        alias_name,
        lambda x: x,
        needs=[source_name],
        cost=0.01,
        metadata={"description": f"UA alias for {source_name}"},
    )


def _add_rad_to_deg(graph: griblet.Graph, base_name: str) -> None:
    graph.add(
        f"{base_name} [deg]",
        lambda x: np.rad2deg(np.asarray(x)),
        needs=[f"{base_name} [rad]"],
        cost=0.02,
        metadata={"description": f"Convert {base_name} from radians to degrees"},
    )


def _add_deg_to_rad(graph: griblet.Graph, base_name: str) -> None:
    graph.add(
        f"{base_name} [rad]",
        lambda x: np.deg2rad(np.asarray(x)),
        needs=[f"{base_name} [deg]"],
        cost=0.02,
        metadata={"description": f"Convert {base_name} from degrees to radians"},
    )


def _decode_ua_text(text: str) -> str:
    cleaned = _UA_MARKUP_RE.sub(lambda match: match.group(1), str(text))
    bracketed = _BRACKET_SPECIES_RE.match(cleaned.strip())
    if bracketed is not None:
        cleaned = bracketed.group(1)
    return cleaned.strip()


def _velocity_alias(raw_name: str) -> str | None:
    if not raw_name.startswith("V"):
        return None
    decoded = _decode_ua_text(raw_name)
    match = re.fullmatch(r"(V[ni]) \(([^,()]+)(?:,([^()]+))?\)", decoded)
    if match is None:
        return None
    prefix, direction, species = match.groups()
    direction_name = direction.strip().replace(" ", "_")
    if species is None:
        return f"{prefix}_{direction_name}"
    species_name = species.strip().replace(" ", "")
    return f"{prefix}_{direction_name}_{species_name}"


def _sum_fields(*parts):
    return np.sum(np.stack([np.asarray(part) for part in parts], axis=0), axis=0)


__all__ = ["build_ua_graph"]
