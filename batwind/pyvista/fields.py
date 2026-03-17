from __future__ import annotations

import numpy as np


_RHO_CGS_TO_SI = 1e3
_KM_TO_M = 1e3
_GAUSS_TO_T = 1e-4

_DENSITY_NAME = "Rho [g/cm^3]"
_WIND_COMPONENT_NAMES = ("U_x [km/s]", "U_y [km/s]", "U_z [km/s]")
_MAGNETIC_COMPONENT_NAMES = ("B_x [Gauss]", "B_y [Gauss]", "B_z [Gauss]")


def resolve_density_si(sds) -> np.ndarray:
    return _RHO_CGS_TO_SI * np.asarray(_variable(sds, _DENSITY_NAME), dtype=float)


def resolve_wind_vector_si(sds) -> np.ndarray:
    return _KM_TO_M * _stack_components(sds, _WIND_COMPONENT_NAMES)


def resolve_wind_speed_si(sds) -> np.ndarray:
    return np.linalg.norm(resolve_wind_vector_si(sds), axis=1)


def resolve_magnetic_vector_si(sds) -> np.ndarray:
    return _GAUSS_TO_T * _stack_components(sds, _MAGNETIC_COMPONENT_NAMES)


def resolve_body_radius(sds) -> float:
    value = sds.aux.get("RBODY")
    if value is not None:
        radius = float(value)
        if radius > 0.0:
            return radius

    radii = np.linalg.norm(np.asarray(sds.points, dtype=float)[:, :3], axis=1)
    return float(np.min(radii[radii > 0.0]))


def radial_component(vectors: np.ndarray, points: np.ndarray) -> np.ndarray:
    radii = np.linalg.norm(points, axis=1)
    rhat = np.divide(points, radii[:, None], out=np.zeros_like(points), where=radii[:, None] > 0.0)
    return np.sum(vectors * rhat, axis=1)


def _stack_components(sds, names: tuple[str, str, str]) -> np.ndarray:
    return np.column_stack([np.asarray(_variable(sds, name), dtype=float) for name in names])


def _variable(sds, name: str):
    variable = getattr(sds, "variable", None)
    if variable is not None:
        return variable(name)
    return sds[name]


__all__ = [
    "radial_component",
    "resolve_body_radius",
    "resolve_density_si",
    "resolve_magnetic_vector_si",
    "resolve_wind_speed_si",
    "resolve_wind_vector_si",
]
