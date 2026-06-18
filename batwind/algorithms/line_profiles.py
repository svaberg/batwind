from __future__ import annotations

import numpy as np

from batcamp import Octree
from batcamp import OctreeInterpolator
from batcamp import OctreeRayTracer
from batcamp import TracedRays

from batwind.algorithms.octree_integration import compute_octree_leaf_centers_and_volumes
from batwind.algorithms.octree_integration import leaf_point_mean


def normalize_view_direction(view_direction: np.ndarray) -> np.ndarray:
    """
    Return one unit observer direction vector.
    """
    direction = np.asarray(view_direction, dtype=float)
    if direction.shape != (3,):
        raise ValueError(f"view_direction must have shape (3,), got {direction.shape}")
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        raise ValueError("view_direction must be non-zero")
    return direction / norm


def project_los_velocity(velocity_vectors: np.ndarray, view_direction: np.ndarray) -> np.ndarray:
    """
    Return observer-parallel velocity for one array of 3-vectors.
    """
    vectors = np.asarray(velocity_vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError(f"velocity_vectors must have shape (n, 3), got {vectors.shape}")
    direction_hat = normalize_view_direction(view_direction)
    return vectors @ direction_hat


def histogram_leaf_los_velocity(
    tree: Octree,
    point_velocity_vectors: np.ndarray,
    velocity_edges: np.ndarray,
    *,
    view_direction: np.ndarray,
    length_scale: float = 1.0,
    point_weights: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Histogram one octree-resolved LOS velocity distribution.

    The histogram is built from leaf-cell mean velocities. Each leaf contributes
    its full volume, optionally weighted by one point-valued scalar field averaged
    to the same leaf. The first and last bins are open-ended overflow bins, so
    values below the first interior edge go into bin 0 and values above the last
    interior edge go into the final bin.

    Units:
    - if ``length_scale`` is given in metres, the leaf volumes are in ``m^3``
    - if ``point_weights`` is omitted, the histogram therefore carries ``m^3``
    - if ``point_weights`` carries a local emissivity in ``W m^-3 sr^-1``, the
      histogram carries ``W sr^-1`` per velocity bin because this path is a
      volume integral, not a line-of-sight image integral
    """
    velocity_edges = np.asarray(velocity_edges, dtype=float)
    if velocity_edges.ndim != 1 or velocity_edges.size < 2:
        raise ValueError("velocity_edges must be one-dimensional with at least two entries")
    if not np.all(np.diff(velocity_edges) > 0.0):
        raise ValueError("velocity_edges must be strictly increasing")

    point_velocity_vectors = np.asarray(point_velocity_vectors, dtype=float)
    if point_velocity_vectors.ndim != 2 or point_velocity_vectors.shape[1] != 3:
        raise ValueError(
            f"point_velocity_vectors must have shape (n_points, 3), got {point_velocity_vectors.shape}"
        )

    leaf_centers, leaf_volumes = compute_octree_leaf_centers_and_volumes(tree, length_scale=length_scale)
    leaf_velocity_vectors = np.column_stack(
        [
            leaf_point_mean(tree, point_velocity_vectors[:, 0]),
            leaf_point_mean(tree, point_velocity_vectors[:, 1]),
            leaf_point_mean(tree, point_velocity_vectors[:, 2]),
        ]
    )
    los_velocity = project_los_velocity(leaf_velocity_vectors, view_direction)
    leaf_weights = np.asarray(leaf_volumes, dtype=float)
    if point_weights is not None:
        leaf_weights = leaf_weights * leaf_point_mean(tree, np.asarray(point_weights, dtype=float))
    n_bins = int(velocity_edges.size - 1)
    bin_ids = np.searchsorted(velocity_edges, los_velocity, side="right") - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    histogram = np.zeros(n_bins, dtype=float)
    np.add.at(histogram, bin_ids, leaf_weights)
    bin_centers = 0.5 * (velocity_edges[:-1] + velocity_edges[1:])
    return {
        "velocity_edges": velocity_edges,
        "velocity_centers": bin_centers,
        "histogram": histogram,
        "leaf_los_velocity": los_velocity,
        "leaf_weights": leaf_weights,
        "leaf_centers": leaf_centers,
        "leaf_volumes": leaf_volumes,
    }


def histogram_traced_los_velocity(
    tree: Octree,
    segments: TracedRays,
    point_velocity_vectors: np.ndarray,
    velocity_edges: np.ndarray,
    *,
    length_scale: float = 1.0,
    point_weights: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Build one per-ray LOS velocity histogram cube from traced octree segments.

    Each crossed segment contributes `segment_length * segment_weight` to one
    velocity bin. Segment velocities and weights are sampled at the segment
    midpoint from one trilinear octree interpolator, so the image can vary
    within one leaf cell instead of collapsing to one block value per cell.
    The first and last bins are open-ended overflow bins.

    Units:
    - if ``length_scale`` is given in metres, traced segment lengths are in ``m``
    - if ``point_weights`` carries a local emissivity in ``W m^-3 sr^-1``, the
      returned spectral cube carries LOS intensity in ``W m^-2 sr^-1`` per
      velocity bin because this path integrates emissivity over path length
    """
    velocity_edges = np.asarray(velocity_edges, dtype=float)
    if velocity_edges.ndim != 1 or velocity_edges.size < 2:
        raise ValueError("velocity_edges must be one-dimensional with at least two entries")
    if not np.all(np.diff(velocity_edges) > 0.0):
        raise ValueError("velocity_edges must be strictly increasing")

    point_velocity_vectors = np.asarray(point_velocity_vectors, dtype=float)
    if point_velocity_vectors.ndim != 2 or point_velocity_vectors.shape[1] != 3:
        raise ValueError(
            f"point_velocity_vectors must have shape (n_points, 3), got {point_velocity_vectors.shape}"
        )

    velocity_interpolator = OctreeInterpolator(tree, point_velocity_vectors)
    scalar_weight_interpolator = None
    if point_weights is not None:
        scalar_weight_interpolator = OctreeInterpolator(tree, np.asarray(point_weights, dtype=float))

    cell_counts = np.diff(segments.ray_offsets)
    n_rays = int(cell_counts.size)
    n_bins = int(velocity_edges.size - 1)
    histogram_flat = np.zeros((n_rays, n_bins), dtype=float)
    if segments.cell_ids.size == 0:
        return {
            "velocity_edges": velocity_edges,
            "velocity_centers": 0.5 * (velocity_edges[:-1] + velocity_edges[1:]),
            "spectral_cube": histogram_flat.reshape(segments.ray_shape + (n_bins,)),
            "ray_segment_counts": cell_counts.reshape(segments.ray_shape),
        }

    if segments.directions.ndim == 1:
        ray_directions = np.broadcast_to(np.asarray(segments.directions, dtype=float), (n_rays, 3)).copy()
    else:
        ray_directions = np.asarray(segments.directions, dtype=float).reshape(n_rays, 3)
    ray_direction_norm = np.linalg.norm(ray_directions, axis=1)
    ray_direction_hat = ray_directions / ray_direction_norm[:, None]
    segment_ray_ids = np.repeat(np.arange(n_rays, dtype=int), cell_counts)
    segment_lengths = np.empty(segments.cell_ids.size, dtype=float)
    segment_midpoints = np.empty((segments.cell_ids.size, 3), dtype=float)
    segment_lo = 0
    for ray_id, cell_count in enumerate(cell_counts):
        if cell_count == 0:
            continue
        time_lo = int(segments.time_offsets[ray_id])
        time_hi = int(segments.time_offsets[ray_id + 1])
        segment_hi = segment_lo + int(cell_count)
        ray_times = segments.times[time_lo:time_hi]
        segment_mid_t = 0.5 * (ray_times[:-1] + ray_times[1:])
        segment_lengths[segment_lo:segment_hi] = (
            np.diff(ray_times) * ray_direction_norm[ray_id] * float(length_scale)
        )
        segment_midpoints[segment_lo:segment_hi] = (
            segments.origins.reshape(n_rays, 3)[ray_id]
            + segment_mid_t[:, None] * ray_directions[ray_id]
        )
        segment_lo = segment_hi
    segment_velocity_vectors = np.asarray(velocity_interpolator(segment_midpoints), dtype=float)
    segment_los_velocity = np.einsum(
        "ij,ij->i",
        segment_velocity_vectors,
        ray_direction_hat[segment_ray_ids],
    )
    segment_weights = segment_lengths.copy()
    if scalar_weight_interpolator is not None:
        segment_weights *= np.asarray(scalar_weight_interpolator(segment_midpoints), dtype=float)
    segment_bin_ids = np.searchsorted(velocity_edges, segment_los_velocity, side="right") - 1
    segment_bin_ids = np.clip(segment_bin_ids, 0, n_bins - 1)
    np.add.at(
        histogram_flat.ravel(),
        segment_ray_ids * n_bins + segment_bin_ids,
        segment_weights,
    )

    return {
        "velocity_edges": velocity_edges,
        "velocity_centers": 0.5 * (velocity_edges[:-1] + velocity_edges[1:]),
        "spectral_cube": histogram_flat.reshape(segments.ray_shape + (n_bins,)),
        "ray_segment_counts": cell_counts.reshape(segments.ray_shape),
    }


def render_los_velocity_histogram_cube(
    tree: Octree,
    point_velocity_vectors: np.ndarray,
    velocity_edges: np.ndarray,
    origins: np.ndarray,
    directions: np.ndarray,
    *,
    length_scale: float = 1.0,
    point_weights: np.ndarray | None = None,
    t_min: float = 0.0,
    t_max: float = np.inf,
) -> dict[str, np.ndarray]:
    """
    Trace one image-plane ray bundle and return one per-ray velocity histogram cube.

    The output cube has the same units as ``histogram_traced_los_velocity``:
    with emissivity weights in ``W m^-3 sr^-1`` and ``length_scale`` in metres,
    each velocity bin stores ``W m^-2 sr^-1``.
    """
    tracer = OctreeRayTracer(tree)
    segments = tracer.trace(origins, directions, t_min=t_min, t_max=t_max)
    out = histogram_traced_los_velocity(
        tree,
        segments,
        point_velocity_vectors,
        velocity_edges,
        length_scale=length_scale,
        point_weights=point_weights,
    )
    out["origins"] = np.asarray(origins, dtype=float)
    out["directions"] = np.asarray(directions, dtype=float)
    return out


def summarize_spectral_cube(
    spectral_cube: np.ndarray,
    velocity_centers: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Return simple per-pixel summary maps for one LOS spectral cube.

    The returned maps are:
    - total intensity
    - intensity-weighted mean LOS velocity
    - line-shape concentration from one normalized entropy measure

    Units:
    - ``total_intensity`` has the same units as one spectral bin in the input cube
    - ``mean_velocity`` has the same units as ``velocity_centers``
    - ``concentration`` is dimensionless
    """
    cube = np.asarray(spectral_cube, dtype=float)
    velocity_centers = np.asarray(velocity_centers, dtype=float)
    if cube.ndim < 2:
        raise ValueError(f"spectral_cube must have at least two dimensions, got {cube.shape}")
    if cube.shape[-1] != velocity_centers.size:
        raise ValueError(
            f"spectral_cube last axis must match velocity_centers, got {cube.shape[-1]} and {velocity_centers.size}"
        )

    total_intensity = np.sum(cube, axis=-1)
    weighted_velocity_sum = np.sum(cube * velocity_centers, axis=-1)
    mean_velocity = np.divide(
        weighted_velocity_sum,
        total_intensity,
        out=np.zeros_like(total_intensity, dtype=float),
        where=total_intensity > 0.0,
    )
    probability = np.divide(
        cube,
        total_intensity[..., None],
        out=np.zeros_like(cube, dtype=float),
        where=total_intensity[..., None] > 0.0,
    )
    entropy_terms = np.zeros_like(probability, dtype=float)
    positive_probability = probability > 0.0
    entropy_terms[positive_probability] = probability[positive_probability] * np.log(probability[positive_probability])
    entropy = -np.sum(entropy_terms, axis=-1)
    max_entropy = np.log(float(velocity_centers.size))
    concentration = 1.0 - entropy / max_entropy if max_entropy > 0.0 else np.ones_like(total_intensity, dtype=float)
    concentration = np.clip(concentration, 0.0, 1.0)
    return {
        "total_intensity": total_intensity,
        "mean_velocity": mean_velocity,
        "concentration": concentration,
    }
