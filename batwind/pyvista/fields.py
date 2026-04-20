from __future__ import annotations

import numpy as np

from batwind.smart_ds import SmartDs


def resolve_density_si(smart_ds: SmartDs) -> np.ndarray:
    return np.asarray(smart_ds["Rho [kg/m^3]"], dtype=float)


def resolve_wind_vector_si(smart_ds: SmartDs) -> np.ndarray:
    return np.asarray(smart_ds["U_xyz [m/s]"], dtype=float)


def resolve_wind_speed_si(smart_ds: SmartDs) -> np.ndarray:
    return np.asarray(smart_ds["U [m/s]"], dtype=float)


def resolve_magnetic_vector_si(smart_ds: SmartDs) -> np.ndarray:
    return np.asarray(smart_ds["B_xyz [T]"], dtype=float)


def resolve_body_radius(smart_ds: SmartDs) -> float:
    value = smart_ds.raw.aux.get("RBODY")
    if value is not None:
        radius = float(value)
        if radius > 0.0:
            return radius

    radii = np.linalg.norm(np.asarray(smart_ds.raw.points, dtype=float)[:, :3], axis=1)
    return float(np.min(radii[radii > 0.0]))


def radial_component(vectors: np.ndarray, points: np.ndarray) -> np.ndarray:
    radii = np.linalg.norm(points, axis=1)
    rhat = np.divide(points, radii[:, None], out=np.zeros_like(points), where=radii[:, None] > 0.0)
    return np.sum(vectors * rhat, axis=1)


__all__ = [
    "radial_component",
    "resolve_body_radius",
    "resolve_density_si",
    "resolve_magnetic_vector_si",
    "resolve_wind_speed_si",
    "resolve_wind_vector_si",
]
