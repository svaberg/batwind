from __future__ import annotations

from collections.abc import Sequence
import logging
import re

import griblet
import numpy as np

from batwind.data.field_names import DEFAULT_XYZ_NAMES

SPHERICAL_COMPONENTS = ("r", "p", "a")
log = logging.getLogger(__name__)


def cartesian_to_spherical_angles(x, y, z):
    """
    Convert Cartesian coordinates to spherical coordinates.

    Conventions:
    - ``r >= 0``
    - ``polar`` is colatitude in ``[0, pi]`` (NaN at ``r == 0``)
    - ``azimuth`` is azimuth from ``atan2(y, x)`` in ``[-pi, pi]``
      (NaN where ``x == y == 0``)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    r = np.sqrt(x * x + y * y + z * z)
    rho_xy = np.sqrt(x * x + y * y)

    polar = np.full_like(r, np.nan, dtype=float)
    azimuth = np.full_like(r, np.nan, dtype=float)

    with np.errstate(invalid="ignore", divide="ignore"):
        mask_r = r > 0
        cos_polar = np.empty_like(r, dtype=float)
        cos_polar.fill(np.nan)
        cos_polar[mask_r] = np.clip(z[mask_r] / r[mask_r], -1.0, 1.0)
        polar[mask_r] = np.arccos(cos_polar[mask_r])

        mask_azimuth = rho_xy > 0
        azimuth[mask_azimuth] = np.arctan2(y[mask_azimuth], x[mask_azimuth])

    return r, polar, azimuth


def spherical_radius_lat_lon_to_cartesian(radius, latitude_deg, longitude_deg):
    """Convert spherical radius + latitude/longitude in degrees to Cartesian coordinates."""
    radius = np.asarray(radius, dtype=float)
    latitude_rad = np.deg2rad(np.asarray(latitude_deg, dtype=float))
    longitude_rad = np.deg2rad(np.asarray(longitude_deg, dtype=float))
    cos_lat = np.cos(latitude_rad)
    x = radius * cos_lat * np.cos(longitude_rad)
    y = radius * cos_lat * np.sin(longitude_rad)
    z = radius * np.sin(latitude_rad)
    return x, y, z


def radial_component(vx, vy, vz, x, y, z):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)
    vz = np.asarray(vz, dtype=float)

    r = np.sqrt(x * x + y * y + z * z)
    out = np.full_like(r, np.nan, dtype=float)
    mask = r > 0
    out[mask] = (vx[mask] * x[mask] + vy[mask] * y[mask] + vz[mask] * z[mask]) / r[mask]
    return out


def spherical_vector_components(vx, vy, vz, x, y, z):
    """
    Return ``(v_r, v_p, v_a)`` using physics convention ``polar=colatitude``.

    ``v_p`` and ``v_a`` are undefined on the polar axis (``x=y=0``) and are
    returned as NaN there. All components are NaN at ``r=0``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)
    vz = np.asarray(vz, dtype=float)

    r = np.sqrt(x * x + y * y + z * z)
    rho_xy = np.sqrt(x * x + y * y)

    v_r = np.full_like(r, np.nan, dtype=float)
    v_p = np.full_like(r, np.nan, dtype=float)
    v_a = np.full_like(r, np.nan, dtype=float)

    mask_r = r > 0
    v_r[mask_r] = (vx[mask_r] * x[mask_r] + vy[mask_r] * y[mask_r] + vz[mask_r] * z[mask_r]) / r[mask_r]

    mask_axis = mask_r & (rho_xy > 0)
    if np.any(mask_axis):
        rho = rho_xy[mask_axis]
        rr = r[mask_axis]
        xx = x[mask_axis]
        yy = y[mask_axis]
        zz = z[mask_axis]
        vxx = vx[mask_axis]
        vyy = vy[mask_axis]
        vzz = vz[mask_axis]

        # e_p = (cos(polar)cos(azimuth), cos(polar)sin(azimuth), -sin(polar))
        # Derived directly in Cartesian coordinates to avoid angle singularities.
        v_p[mask_axis] = (zz * (xx * vxx + yy * vyy) - (xx * xx + yy * yy) * vzz) / (rr * rho)
        # e_a = (-sin(azimuth), cos(azimuth), 0)
        v_a[mask_axis] = (-yy * vxx + xx * vyy) / rho

    return v_r, v_p, v_a


def build_spherical_graph(
    variable_names: Sequence[str],
    coord_fields: Sequence[str] = DEFAULT_XYZ_NAMES,
):
    """
    Build a griblet graph for spherical geometry and auto-detected vector components.

    This requires ``griblet``.
    """
    log.info("build_spherical_graph...")
    x_name, y_name, z_name = coord_fields
    match = re.match(r"^X \[(.+)\]$", x_name)
    if match is None:
        raise ValueError(f"could not infer radius name from coordinate field {x_name!r}")
    r_name = f"R [{match.group(1)}]"
    log.debug("build_spherical_graph coord_fields=%s radius_field=%s", coord_fields, r_name)

    graph = griblet.Graph()
    deps = [x_name, y_name, z_name]
    graph.add(
        r_name,
        lambda x, y, z: cartesian_to_spherical_angles(x, y, z)[0],
        needs=deps,
        cost=0.2,
        metadata={"description": "Cartesian->spherical radius"},
    )
    graph.add(
        "polar [rad]",
        lambda x, y, z: cartesian_to_spherical_angles(x, y, z)[1],
        needs=deps,
        cost=0.2,
        metadata={"description": "Cartesian->spherical colatitude"},
    )
    graph.add(
        "azimuth [rad]",
        lambda x, y, z: cartesian_to_spherical_angles(x, y, z)[2],
        needs=deps,
        cost=0.2,
        metadata={"description": "Cartesian->spherical azimuth"},
    )
    graph.add(
        "Lat [deg]",
        lambda polar: np.rad2deg((0.5 * np.pi) - np.asarray(polar, dtype=float)),
        needs=["polar [rad]"],
        cost=0.05,
        metadata={"description": "Spherical colatitude -> latitude"},
    )
    graph.add(
        "Lon [deg]",
        lambda azimuth: np.rad2deg(np.asarray(azimuth, dtype=float)),
        needs=["azimuth [rad]"],
        cost=0.05,
        metadata={"description": "Spherical azimuth -> longitude"},
    )
    graph.add(
        x_name,
        lambda r, lon, lat: spherical_radius_lat_lon_to_cartesian(r, lat, lon)[0],
        needs=[r_name, "Lon [deg]", "Lat [deg]"],
        cost=0.2,
        metadata={"description": "Spherical radius/longitude/latitude -> Cartesian x"},
    )
    graph.add(
        y_name,
        lambda r, lon, lat: spherical_radius_lat_lon_to_cartesian(r, lat, lon)[1],
        needs=[r_name, "Lon [deg]", "Lat [deg]"],
        cost=0.2,
        metadata={"description": "Spherical radius/longitude/latitude -> Cartesian y"},
    )
    graph.add(
        z_name,
        lambda r, lon, lat: spherical_radius_lat_lon_to_cartesian(r, lat, lon)[2],
        needs=[r_name, "Lon [deg]", "Lat [deg]"],
        cost=0.2,
        metadata={"description": "Spherical radius/longitude/latitude -> Cartesian z"},
    )
    pattern = re.compile(r"^(?P<prefix>.+)_(?P<comp>[xyz]) \[(?P<unit>.+)\]$")
    by_prefix: dict[tuple[str, str], set[str]] = {}
    for name in variable_names:
        m = pattern.match(name)
        if not m:
            continue
        by_prefix.setdefault((m.group("prefix"), m.group("unit")), set()).add(m.group("comp"))

    n_vectors = 0
    detected_vectors: list[str] = []
    for (prefix, unit), found in sorted(by_prefix.items()):
        if found != {"x", "y", "z"}:
            continue
        n_vectors += 1
        detected_vectors.append(f"{prefix} [{unit}]")
        deps = [f"{prefix}_x [{unit}]", f"{prefix}_y [{unit}]", f"{prefix}_z [{unit}]", x_name, y_name, z_name]

        if "r" in SPHERICAL_COMPONENTS:
            graph.add(
                f"{prefix}_r [{unit}]",
                lambda vx, vy, vz, x, y, z: spherical_vector_components(vx, vy, vz, x, y, z)[0],
                needs=deps,
                cost=0.4,
                metadata={"description": f"{prefix} radial component"},
            )
        if "p" in SPHERICAL_COMPONENTS:
            graph.add(
                f"{prefix}_p [{unit}]",
                lambda vx, vy, vz, x, y, z: spherical_vector_components(vx, vy, vz, x, y, z)[1],
                needs=deps,
                cost=0.5,
                metadata={"description": f"{prefix} polar component"},
            )
        if "a" in SPHERICAL_COMPONENTS:
            graph.add(
                f"{prefix}_a [{unit}]",
                lambda vx, vy, vz, x, y, z: spherical_vector_components(vx, vy, vz, x, y, z)[2],
                needs=deps,
                cost=0.5,
                metadata={"description": f"{prefix} azimuthal component"},
            )
    log.debug(
        "build_spherical_graph vectors=%d detected=%s components=%s",
        n_vectors,
        tuple(detected_vectors),
        SPHERICAL_COMPONENTS,
    )
    log.debug("build_spherical_graph complete fields=%d", len(tuple(graph.fields())))
    return graph
