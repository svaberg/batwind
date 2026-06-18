from __future__ import annotations

import numpy as np
from batcamp import camera_rays
from batcamp import Octree
from batcamp import OctreeInterpolator
from batcamp import OctreeRayTracer

from batwind.smart_ds import SmartDs


def view_direction_from_inclination_phase(inclination_deg: float, phase_deg: float) -> np.ndarray:
    """
    Return one unit observer direction from stellar inclination and rotation phase.

    Conventions:
    - the stellar rotation axis is ``+Z``
    - ``inclination_deg = 0`` means pole-on along ``+Z``
    - ``inclination_deg = 90`` and ``phase_deg = 0`` means a ``+Y`` line of sight
    - increasing phase rotates the observer direction around ``+Z``
    """
    inclination_rad = np.deg2rad(float(inclination_deg))
    phase_rad = np.deg2rad(float(phase_deg))
    view_direction = np.array(
        [
            np.sin(inclination_rad) * np.sin(phase_rad),
            np.sin(inclination_rad) * np.cos(phase_rad),
            np.cos(inclination_rad),
        ],
        dtype=float,
    )
    return view_direction / np.linalg.norm(view_direction)


def camera_rays_from_view_direction(
    smart_ds: SmartDs,
    view_direction: np.ndarray,
    *,
    image_n: int = 128,
    side_length_r: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """
    Build one parallel image-plane ray bundle for one observer direction.

    The returned image extent is expressed in image-plane coordinates in ``R_*``.
    """
    view_direction = np.asarray(view_direction, dtype=float)
    view_direction = view_direction / np.linalg.norm(view_direction)

    preferred_up = np.array([0.0, 0.0, 1.0], dtype=float)
    if np.isclose(np.abs(np.dot(preferred_up, view_direction)), 1.0):
        preferred_up = np.array([1.0, 0.0, 0.0], dtype=float)
    up = preferred_up - np.dot(preferred_up, view_direction) * view_direction
    up = up / np.linalg.norm(up)

    x = np.asarray(smart_ds["X [R]"], dtype=float)
    y = np.asarray(smart_ds["Y [R]"], dtype=float)
    z = np.asarray(smart_ds["Z [R]"], dtype=float)
    center_r = np.array(
        [
            0.5 * (float(np.min(x)) + float(np.max(x))),
            0.5 * (float(np.min(y)) + float(np.max(y))),
            0.5 * (float(np.min(z)) + float(np.max(z))),
        ],
        dtype=float,
    )
    half_diagonal_r = 0.5 * np.linalg.norm(
        [
            float(np.max(x) - np.min(x)),
            float(np.max(y) - np.min(y)),
            float(np.max(z) - np.min(z)),
        ]
    )
    camera_distance_r = half_diagonal_r + 1.0
    origin_r = center_r - camera_distance_r * view_direction
    target_r = center_r + camera_distance_r * view_direction
    origins, directions = camera_rays(
        origin=tuple(origin_r),
        target=tuple(target_r),
        up=tuple(up),
        nx=int(image_n),
        ny=int(image_n),
        width=float(side_length_r),
        height=float(side_length_r),
        projection="parallel",
    )
    half_side_r = 0.5 * float(side_length_r)
    extent_r = (-half_side_r, half_side_r, -half_side_r, half_side_r)
    return origins, directions, extent_r


def _render_trilinear_scalar_image(
    tree: Octree,
    point_values: np.ndarray,
    origins: np.ndarray,
    directions: np.ndarray,
    *,
    length_scale: float,
) -> np.ndarray:
    """
    Integrate one point-valued scalar field along traced rays.

    With local emissivity units ``W m^-3 sr^-1`` and ``length_scale`` in metres,
    the returned image has units ``W m^-2 sr^-1``.
    """
    interpolator = OctreeInterpolator(tree, np.asarray(point_values, dtype=float))
    tracer = OctreeRayTracer(tree)
    image_native, _ = tracer.trilinear_image(interpolator, origins, directions)
    return np.asarray(image_native, dtype=float) * float(length_scale)


def _visible_point_mask(smart_ds: SmartDs, view_direction: np.ndarray) -> np.ndarray:
    """
    Return the pointwise visibility mask outside one opaque unit-radius star.

    Dataset coordinates are assumed to be expressed in ``R_*``.
    """
    direction_hat = np.asarray(view_direction, dtype=float)
    direction_hat = direction_hat / np.linalg.norm(direction_hat)
    positions_r = np.column_stack(
        [
            np.asarray(smart_ds["X [R]"], dtype=float),
            np.asarray(smart_ds["Y [R]"], dtype=float),
            np.asarray(smart_ds["Z [R]"], dtype=float),
        ]
    )
    parallel_r = positions_r @ direction_hat
    radial_distance_sq_r = np.sum(positions_r**2, axis=1)
    perpendicular_sq_r = np.clip(radial_distance_sq_r - parallel_r**2, 0.0, None)
    near_limb_r = -np.sqrt(np.clip(1.0 - perpendicular_sq_r, 0.0, None))
    return (perpendicular_sq_r >= 1.0) | (parallel_r <= near_limb_r)


def band_intensity_image_si(
    smart_ds: SmartDs,
    point_emissivity_w_m3_sr: np.ndarray,
    *,
    inclination_deg: float,
    phase_deg: float,
    image_n: int = 128,
    side_length_r: float = 4.0,
    tree: Octree | None = None,
) -> dict[str, np.ndarray | tuple[float, float, float, float]]:
    """
    Render one band intensity image for one inclination and stellar phase.

    Units:
    - emissivity: ``W m^-3 sr^-1``
    - path length: ``m``
    - image intensity: ``W m^-2 sr^-1``
    """
    if tree is None:
        tree = Octree.from_ds(smart_ds.raw)
    view_direction = view_direction_from_inclination_phase(inclination_deg, phase_deg)
    origins, directions, extent_r = camera_rays_from_view_direction(
        smart_ds,
        view_direction,
        image_n=image_n,
        side_length_r=side_length_r,
    )
    body_radius_m = float(smart_ds["RBODY [m]"])
    visible_point_emissivity_w_m3_sr = np.asarray(point_emissivity_w_m3_sr, dtype=float) * np.asarray(
        _visible_point_mask(smart_ds, view_direction),
        dtype=float,
    )
    image_w_m2_sr = _render_trilinear_scalar_image(
        tree,
        visible_point_emissivity_w_m3_sr,
        origins,
        directions,
        length_scale=body_radius_m,
    )
    return {
        "image": image_w_m2_sr,
        "extent_r": extent_r,
        "origins": np.asarray(origins, dtype=float),
        "directions": np.asarray(directions, dtype=float),
        "view_direction": view_direction,
    }


def integrate_image_radiant_intensity_si(
    image_w_m2_sr: np.ndarray,
    extent_r: tuple[float, float, float, float],
    body_radius_m: float,
) -> float:
    """
    Integrate one band intensity image over projected image-plane area.

    Units:
    - image intensity: ``W m^-2 sr^-1``
    - projected area: ``m^2``
    - returned radiant intensity: ``W sr^-1``
    """
    x_min, x_max, y_min, y_max = extent_r
    pixel_area_m2 = (
        (float(x_max - x_min) * float(body_radius_m)) * (float(y_max - y_min) * float(body_radius_m))
        / float(np.asarray(image_w_m2_sr).size)
    )
    return float(np.sum(np.asarray(image_w_m2_sr, dtype=float)) * pixel_area_m2)


def band_light_curve_si(
    smart_ds: SmartDs,
    point_emissivity_w_m3_sr: np.ndarray,
    phase_deg: np.ndarray,
    *,
    inclination_deg: float,
    image_n: int = 128,
    side_length_r: float = 4.0,
    tree: Octree | None = None,
) -> dict[str, np.ndarray]:
    """
    Return one band light curve as radiant intensity versus stellar phase.

    The returned ordinate has units ``W sr^-1`` because it is the image-plane
    integral of a band intensity image in ``W m^-2 sr^-1``.
    """
    if tree is None:
        tree = Octree.from_ds(smart_ds.raw)
    phase_deg = np.asarray(phase_deg, dtype=float)
    body_radius_m = float(smart_ds["RBODY [m]"])
    radiant_intensity_w_sr = np.empty_like(phase_deg, dtype=float)
    for phase_id, phase in enumerate(phase_deg):
        image = band_intensity_image_si(
            smart_ds,
            point_emissivity_w_m3_sr,
            inclination_deg=inclination_deg,
            phase_deg=float(phase),
            image_n=image_n,
            side_length_r=side_length_r,
            tree=tree,
        )
        radiant_intensity_w_sr[phase_id] = integrate_image_radiant_intensity_si(
            image["image"],
            image["extent_r"],
            body_radius_m,
        )
    return {
        "phase_deg": phase_deg,
        "radiant_intensity_w_sr": radiant_intensity_w_sr,
    }
