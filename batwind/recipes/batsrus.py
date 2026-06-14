from __future__ import annotations

from collections.abc import Sequence
import logging
import re

import griblet
import numpy as np
from scipy.constants import atomic_mass, mu_0, proton_mass

from batwind.recipes.vectors import build_vector_graph

log = logging.getLogger(__name__)

_DEFAULT_GAMMA = 5.0 / 3.0


_UNIT_FACTORS = {
    "g/cm^3": ("kg/m^3", 1e3),
    "amu/cm^3": ("kg/m^3", atomic_mass * 1e6),
    "km/s": ("m/s", 1e3),
    "Gauss": ("T", 1e-4),
    "G": ("T", 1e-4),
    "nT": ("T", 1e-9),
    "erg/cm^3": ("J/m^3", 1e-1),
    "dyne/cm^2": ("Pa", 1e-1),
    "nPa": ("Pa", 1e-9),
    "`mA/m^2": ("A/m^2", 1e-6),
}


def build_batsrus_graph(
    variable_names: Sequence[str],
    *,
    gamma: float | None = None,
    body_radius_m: float | None = None,
):
    """
    Build a griblet graph for BATSRUS-style fields.

    Current scope:
    - canonical bracketed names for unbracketed unit strings
    - SI conversion recipes for common BATSRUS units
    - coordinate conversion X/Y/Z [R] -> [m] via ``RBODY [m]``
    - common derived fields: |U|, |B|, c_s, c_A, M_A
    """
    variable_names = tuple(variable_names)
    log.info("build_batsrus_graph...")
    log.debug(
        "build_batsrus_graph variables=%d gamma=%s body_radius_m=%s",
        len(variable_names),
        gamma is not None,
        body_radius_m is not None,
    )
    graph = griblet.Graph()
    unit_graph = build_unit_normalization_graph(variable_names, gamma=gamma, body_radius_m=body_radius_m)
    log.debug("build_batsrus_graph merging unit-normalization graph fields=%d", len(tuple(unit_graph.fields())))
    graph.merge(unit_graph)
    derived_graph = build_common_derived_graph()
    log.debug("build_batsrus_graph merging common-derived graph fields=%d", len(tuple(derived_graph.fields())))
    vector_graph = build_vector_graph(tuple(variable_names) + tuple(graph.fields()) + tuple(derived_graph.fields()))
    log.debug("build_batsrus_graph merging vector graph fields=%d", len(tuple(vector_graph.fields())))
    graph.merge(vector_graph)
    graph.merge(derived_graph)
    log.debug("build_batsrus_graph complete fields=%d", len(tuple(graph.fields())))
    return graph


def build_unit_normalization_graph(
    variable_names: Sequence[str],
    *,
    gamma: float | None = None,
    body_radius_m: float | None = None,
):
    variable_names = tuple(variable_names)
    log.debug(
        "build_unit_normalization_graph variables=%d gamma=%s body_radius_m=%s",
        len(variable_names),
        gamma is not None,
        body_radius_m is not None,
    )
    graph = griblet.Graph()
    n_canonicalized = 0
    n_converted = 0

    for raw_name in variable_names:
        parsed = _parse_var_name(raw_name)
        if parsed is None:
            continue
        base, unit = parsed

        canonical_name = f"{base} [{unit}]"
        if canonical_name != raw_name:
            n_canonicalized += 1
            graph.add(
                canonical_name,
                lambda x: x,
                needs=[raw_name],
                cost=0.01,
                metadata={"description": f"Canonicalize unit brackets for {raw_name}"},
            )
            source_name = canonical_name
        else:
            source_name = raw_name

        match = _UNIT_FACTORS.get(unit)
        if match is None:
            continue

        si_unit, factor = match
        target_name = f"{base} [{si_unit}]"
        if target_name == source_name and factor == 1:
            continue
        n_converted += 1
        graph.add(
            target_name,
            lambda x, factor=factor: factor * np.asarray(x),
            needs=[source_name],
            cost=0.05,
            metadata={"description": f"Unit conversion {unit}->{si_unit}"},
        )

    graph.merge(build_coordinate_scale_graph(body_radius_m))

    if gamma is not None:
        graph.add(
            "GAMMA [none]",
            lambda: float(gamma),
            needs=[],
            cost=0.01,
            metadata={"description": "Configured gamma"},
        )

    log.debug(
        "build_unit_normalization_graph canonicalized=%d converted=%d fields=%d",
        n_canonicalized,
        n_converted,
        len(tuple(graph.fields())),
    )
    return graph


def build_coordinate_scale_graph(body_radius_m: float | None = None):
    log.debug("build_coordinate_scale_graph body_radius_m=%s", body_radius_m is not None)
    graph = griblet.Graph()
    if body_radius_m is not None:
        log.debug("build_coordinate_scale_graph adding RBODY [m]")
        graph.add(
            "RBODY [m]",
            lambda: float(body_radius_m),
            needs=[],
            cost=0.0,
            metadata={"description": "Configured body radius"},
        )
    for axis in ("X", "Y", "Z"):
        source = f"{axis} [R]"
        target = f"{axis} [m]"
        graph.add(
            target,
            lambda x, rbody: np.asarray(rbody) * np.asarray(x),
            needs=[source, "RBODY [m]"],
            cost=0.05,
            metadata={"description": "Scale body-radius coordinates to meters"},
        )
    graph.add(
        "R [m]",
        lambda r, rbody: np.asarray(rbody) * np.asarray(r),
        needs=["R [R]", "RBODY [m]"],
        cost=0.05,
        metadata={"description": "Scale spherical radius to meters"},
    )
    log.debug("build_coordinate_scale_graph complete fields=%s", tuple(graph.fields()))
    return graph


def build_common_derived_graph():
    log.debug("build_common_derived_graph...")
    graph = griblet.Graph()

    graph.add(
        "E_mhd_x [V/m]",
        lambda uy, uz, by, bz: np.asarray(uz) * np.asarray(by) - np.asarray(uy) * np.asarray(bz),
        needs=["U_y [m/s]", "U_z [m/s]", "B_y [T]", "B_z [T]"],
        cost=0.15,
        metadata={"description": "Ideal-MHD electric field x-component: -(U x B)_x"},
    )
    graph.add(
        "E_mhd_y [V/m]",
        lambda ux, uz, bx, bz: np.asarray(ux) * np.asarray(bz) - np.asarray(uz) * np.asarray(bx),
        needs=["U_x [m/s]", "U_z [m/s]", "B_x [T]", "B_z [T]"],
        cost=0.15,
        metadata={"description": "Ideal-MHD electric field y-component: -(U x B)_y"},
    )
    graph.add(
        "E_mhd_z [V/m]",
        lambda ux, uy, bx, by: np.asarray(uy) * np.asarray(bx) - np.asarray(ux) * np.asarray(by),
        needs=["U_x [m/s]", "U_y [m/s]", "B_x [T]", "B_y [T]"],
        cost=0.15,
        metadata={"description": "Ideal-MHD electric field z-component: -(U x B)_z"},
    )

    # Sound speed c_s [m/s]
    graph.add(
        "c_s [m/s]",
        lambda P, rho: np.sqrt(_DEFAULT_GAMMA * np.asarray(P) / np.asarray(rho)),
        needs=["P [Pa]", "Rho [kg/m^3]"],
        cost=0.25,
        metadata={"description": "Adiabatic sound speed with fallback gamma=5/3"},
    )
    graph.add(
        "c_s [m/s]",
        lambda P, rho, gamma: np.sqrt(_safe_gamma(gamma) * np.asarray(P) / np.asarray(rho)),
        needs=["P [Pa]", "Rho [kg/m^3]", "GAMMA [none]"],
        cost=0.2,
        metadata={"description": "Adiabatic sound speed using GAMMA aux"},
    )

    # Alfven speed and Alfven Mach
    graph.add(
        "c_A [m/s]",
        lambda B, rho: np.asarray(B) / np.sqrt(mu_0 * np.asarray(rho)),
        needs=["B [T]", "Rho [kg/m^3]"],
        cost=0.2,
        metadata={"description": "Alfven speed"},
    )
    graph.add(
        "M_A [none]",
        lambda U, cA: np.asarray(U) / np.asarray(cA),
        needs=["U [m/s]", "c_A [m/s]"],
        cost=0.1,
        metadata={"description": "Alfven Mach number"},
    )
    graph.add(
        "Ma [none]",
        lambda U, cs: np.asarray(U) / np.asarray(cs),
        needs=["U [m/s]", "c_s [m/s]"],
        cost=0.1,
        metadata={"description": "Sonic Mach number"},
    )
    graph.add(
        "P_b [Pa]",
        lambda B: np.asarray(B) ** 2 / (2.0 * mu_0),
        needs=["B [T]"],
        cost=0.12,
        metadata={"description": "Magnetic pressure"},
    )
    graph.add(
        "ram_pressure [Pa]",
        lambda rho, U: np.asarray(rho) * (np.asarray(U) ** 2),
        needs=["Rho [kg/m^3]", "U [m/s]"],
        cost=0.12,
        metadata={"description": "Ram pressure"},
    )
    graph.add(
        "standoff_distance [m]",
        _standoff_distance_from_rho_u,
        needs=["Rho [kg/m^3]", "U [m/s]"],
        cost=0.2,
        metadata={"description": "Magnetospheric stand-off proxy from inertial ram pressure"},
    )
    graph.add(
        "beta [none]",
        lambda P, Pb: np.asarray(P) / np.asarray(Pb),
        needs=["P [Pa]", "P_b [Pa]"],
        cost=0.12,
        metadata={"description": "Plasma beta"},
    )
    graph.add(
        "Ne [1/m^3]",
        lambda rho: np.asarray(rho) / proton_mass,
        needs=["Rho [kg/m^3]"],
        cost=0.12,
        metadata={"description": "Electron number density assuming ionized hydrogen"},
    )
    graph.add(
        "Ne [1/cm^3]",
        lambda ne_m3: 1.0e-6 * np.asarray(ne_m3),
        needs=["Ne [1/m^3]"],
        cost=0.02,
        metadata={"description": "Electron number density in cgs units"},
    )

    # `unblocked_solid_angle [sr]` is the exterior visibility factor
    # `omega(r) = 2*pi*(1 + sqrt(1 - r^-2))` for `r >= 1` in stellar-radius units.
    graph.add(
        "unblocked_solid_angle [sr]",
        lambda r: 2.0 * np.pi * (1.0 + np.sqrt(np.clip(1.0 - np.asarray(r) ** -2, 0.0, None))),
        needs=["R [R]"],
        cost=0.08,
        metadata={"description": "Exterior unblocked solid angle outside one opaque stellar sphere"},
    )
    graph.add(
        "transition_region_emission_weight [none]",
        lambda te: np.ones_like(np.asarray(te), dtype=float),
        needs=["te [K]"],
        cost=0.2,
        metadata={"description": "No transition-region emission correction"},
    )
    graph.add(
        "transition_region_emission_weight [none]",
        lambda te, do_extend, tm, delta_tm: _transition_region_emission_weight(te, do_extend, tm, delta_tm),
        needs=["te [K]", "DoExtendTransitionRegion", "TeTransitionRegionSi", "DeltaTeModSi"],
        cost=0.12,
        metadata={"description": "BATSRUS transition-region optically thin emission correction"},
    )

    graph.add(
        "mass_flux [kg/m^2/s]",
        lambda rho, ur: np.asarray(rho) * np.asarray(ur),
        needs=["Rho [kg/m^3]", "U_r [m/s]"],
        cost=0.12,
        metadata={"description": "Radial mass flux density"},
    )
    graph.add(
        "energy_flux [W/m^2]",
        lambda e, ur: np.asarray(e) * np.asarray(ur),
        needs=["E [J/m^3]", "U_r [m/s]"],
        cost=0.12,
        metadata={"description": "Radial energy flux density"},
    )

    graph.add(
        "cylindrical_radius [R]",
        lambda x, y: np.sqrt(np.asarray(x) ** 2 + np.asarray(y) ** 2),
        needs=["X [R]", "Y [R]"],
        cost=0.1,
        metadata={"description": "Cylindrical radius from body-radius coordinates"},
    )
    graph.add(
        "cylindrical_radius [m]",
        lambda x, y: np.sqrt(np.asarray(x) ** 2 + np.asarray(y) ** 2),
        needs=["X [m]", "Y [m]"],
        cost=0.1,
        metadata={"description": "Cylindrical radius from SI coordinates"},
    )

    graph.add(
        "magnetic_torque_density [N/m]",
        lambda varpi, bphi, br: -np.asarray(varpi) * np.asarray(bphi) * np.asarray(br) / mu_0,
        needs=["cylindrical_radius [m]", "B_a [T]", "B_r [T]"],
        cost=0.2,
        metadata={"description": "Magnetic z-torque density (shell form)"},
    )
    graph.add(
        "dynamic_torque_density [N/m]",
        lambda varpi, rho, uphi, ur: np.asarray(varpi) * np.asarray(rho) * np.asarray(uphi) * np.asarray(ur),
        needs=["cylindrical_radius [m]", "Rho [kg/m^3]", "U_a [m/s]", "U_r [m/s]"],
        cost=0.2,
        metadata={"description": "Dynamic z-torque density (shell form)"},
    )
    graph.add(
        "total_torque_density [N/m]",
        lambda tmag, tdyn: np.asarray(tmag) + np.asarray(tdyn),
        needs=["magnetic_torque_density [N/m]", "dynamic_torque_density [N/m]"],
        cost=0.05,
        metadata={"description": "Total z-torque density (shell form)"},
    )

    graph.add(
        "B_meridional [T]",
        lambda bp: -np.asarray(bp),
        needs=["B_p [T]"],
        cost=0.05,
        metadata={"description": "Meridional magnetic component (northward)"},
    )
    graph.add(
        "B_tangential [T]",
        lambda ba, bmer: np.sqrt(np.asarray(ba) ** 2 + np.asarray(bmer) ** 2),
        needs=["B_a [T]", "B_meridional [T]"],
        cost=0.08,
        metadata={"description": "Tangential magnetic magnitude on spherical shell"},
    )

    log.debug(
        "build_common_derived_graph complete fields=%d names=%s",
        len(tuple(graph.fields())),
        tuple(graph.fields()),
    )
    return graph


def _transition_region_emission_weight(
    temperature_k: np.ndarray,
    do_extend: bool,
    transition_temperature_k: float,
    delta_temperature_k: float,
) -> np.ndarray:
    temperature_k = np.asarray(temperature_k, dtype=float)
    if not bool(do_extend):
        return np.ones_like(temperature_k, dtype=float)
    fraction_spitzer = 0.5 * (
        1.0 + np.tanh((temperature_k - float(transition_temperature_k)) / float(delta_temperature_k))
    )
    extension_factor = fraction_spitzer + (
        1.0 - fraction_spitzer
    ) * (float(transition_temperature_k) / temperature_k) ** 2.5
    return 1.0 / extension_factor


def _parse_var_name(name: str):
    """
    Parse BATSRUS variable names.

    Supports:
    - ``Foo [unit]``
    - ``Foo unit``  (legacy unbracketed style)
    """
    m = re.match(r"^(?P<base>.+?) \[(?P<unit>.+)\]$", name)
    if m:
        return m.group("base"), m.group("unit")

    # If there is a space and no brackets, interpret the final token as the unit.
    if " " in name and "[" not in name and "]" not in name:
        base, unit = name.rsplit(" ", 1)
        if "/" in unit or unit.isalpha() or any(ch.isdigit() for ch in unit):
            return base, unit
    return None


def _safe_gamma(gamma):
    g = float(gamma)
    if not np.isfinite(g) or g <= 0:
        return _DEFAULT_GAMMA
    return g


def _standoff_distance_from_rho_u(rho, U):
    p_ram = np.asarray(rho) * (np.asarray(U) ** 2)
    numer = (0.7e-4**2) / (2.0 * mu_0)
    return np.power(numer / p_ram, 1.0 / 6.0)


__all__ = [
    "build_batsrus_graph",
    "build_common_derived_graph",
    "build_coordinate_scale_graph",
    "build_unit_normalization_graph",
]
