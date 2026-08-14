"""Wedge-specific 3D BATSRUS readers, axisymmetric adapters, and quicklooks."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import replace
import logging
from pathlib import Path
import re

import griblet
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.tri import Triangulation
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from scipy.spatial import cKDTree

from batread import Dataset
from batread.read_plt import read_plt

from batwind.data.field_names import DEFAULT_XYZ_NAMES
from batwind.param_in import find_param_in
from batwind.param_in import ParamIn
from batwind.param_in import StarParams
from batwind.param_in import TransitionRegionParams
from batwind.pipelines.utils import iteration_token_from_path
from batwind.recipes.batsrus import build_batsrus_graph
from batwind.recipes.spherical import build_spherical_graph
from batwind.smart_ds import SmartDs

log = logging.getLogger(__name__)

_VECTOR_COMPONENT_RE = re.compile(r"^(?P<base>.+)_(?P<comp>[xyz]) (?P<unit>\[[^\]]+\])$")
_VECTOR_X_RE = re.compile(r"^(?P<base>.+)_x (?P<unit>\[[^\]]+\])$")


@dataclass(frozen=True, slots=True)
class WedgeMeridianData:
    """Raw wedge pair plus one longitude-collapsed cell-centre dataset."""

    geometry_path: Path
    cell_center_path: Path
    geometry: Dataset
    cell_centers: Dataset
    reduced_cell_centers: Dataset
    reference_azimuth_rad: float
    point_group_ids: np.ndarray
    sample_counts: np.ndarray


def _infer_param_component(param_in: ParamIn, file_path: str | Path, command: str, *, default="root") -> str:
    candidates: list[str] = []
    seen: set[str] = set()
    for session in param_in.sessions:
        for component_name, component_data in session.items():
            if component_name in seen or command not in component_data:
                continue
            candidates.append(component_name)
            seen.add(component_name)
    if not candidates:
        return str(default)
    parts = {part.casefold() for part in Path(file_path).resolve().parent.parts}
    for component_name in candidates:
        if str(component_name).casefold() in parts:
            return component_name
    if default in candidates:
        return str(default)
    return candidates[0]


class AxisymmetricWedgeDs(SmartDs):
    """
    SmartDs-like adapter that reconstructs 3D axisymmetric values from one wedge meridian.

    The stored raw dataset is the longitude-collapsed meridional cell-centre slice.
    Resampling onto arbitrary 3D points converts the query points to ``(r, polar)``,
    interpolates there, and rotates Cartesian vector components back into the query
    longitude.
    """

    _MERIDIAN_COORD_FIELDS = ("R [R]", "polar [rad]")
    _RPA_COORD_FIELDS = ("R [R]", "polar [rad]", "azimuth [rad]")

    def __init__(
        self,
        wedge_data: WedgeMeridianData,
        *,
        cache_enabled: bool = True,
        computation_graph: griblet.Graph | None = None,
    ) -> None:
        super().__init__(
            wedge_data.reduced_cell_centers,
            cache_enabled=cache_enabled,
            computation_graph=computation_graph,
        )
        self._wedge_data = wedge_data
        x = np.asarray(self.raw["X [R]"], dtype=float)
        y = np.asarray(self.raw["Y [R]"], dtype=float)
        z = np.asarray(self.raw["Z [R]"], dtype=float)
        radius, polar, _azimuth = _cartesian_to_rpa(x, y, z)
        self._meridian_source_coords = np.column_stack((radius, polar))
        self._meridian_coord_mask = np.isfinite(self._meridian_source_coords).all(axis=1)
        self._meridian_spatial_cache: dict[str, object | None] = {
            "nearest_tree": None,
            "linear_triangulation": None,
        }

    @classmethod
    def from_file(
        cls,
        file: str | Path,
        *,
        pair_path: str | Path | None = None,
        reference_azimuth_rad: float = 0.0,
        group_decimals: int = 6,
        batsrus: bool = True,
        spherical: bool = True,
        body_radius_m: float | None = None,
        cache_enabled: bool = True,
    ) -> "AxisymmetricWedgeDs":
        wedge_data = read_wedge_meridian(
            file,
            pair_path=pair_path,
            reference_azimuth_rad=reference_azimuth_rad,
            group_decimals=group_decimals,
        )
        reduced = _merge_nearby_param_in_aux(wedge_data.reduced_cell_centers, wedge_data.cell_center_path)
        if reduced is not wedge_data.reduced_cell_centers:
            wedge_data = replace(wedge_data, reduced_cell_centers=reduced)
        if body_radius_m is None:
            radius_from_aux = reduced.aux.get("Star_radius_m")
            if radius_from_aux is not None:
                body_radius_m = float(radius_from_aux)
        out = cls(wedge_data, cache_enabled=cache_enabled)
        if batsrus:
            out.merge_computation_graph(
                build_batsrus_graph(
                    out.raw.variables,
                    gamma=out.raw.aux.get("GAMMA"),
                    body_radius_m=body_radius_m,
                )
            )
        if spherical:
            out.merge_computation_graph(build_spherical_graph(tuple(out)))
        return out

    @property
    def wedge_data(self) -> WedgeMeridianData:
        return self._wedge_data

    @property
    def reference_azimuth_rad(self) -> float:
        return float(self._wedge_data.reference_azimuth_rad)

    @property
    def title(self) -> str:
        return str(self.raw.title)

    @property
    def zone(self) -> str:
        return str(self.raw.zone)

    def resample(
        self,
        sample_points,
        *,
        coordinate_fields: Sequence[str] | None = None,
        fields: Sequence[str] | None = None,
        method: str = "auto",
        fill_value: float = np.nan,
        corners=None,
        copy_aux: bool = True,
        title: str | None = None,
        zone: str | None = None,
    ) -> SmartDs:
        sample_points = np.asarray(sample_points, dtype=float)
        if sample_points.ndim == 1:
            sample_points = sample_points[np.newaxis, :]
        if sample_points.ndim < 2:
            raise ValueError("sample_points must have shape (..., ndim)")
        ndim = int(sample_points.shape[-1])
        sample_shape = sample_points.shape[:-1]
        flat_sample_points = sample_points.reshape(-1, ndim)

        if coordinate_fields is None:
            coordinate_fields = _infer_coordinate_fields(self.raw.variables, ndim)
        coordinate_fields = tuple(coordinate_fields)
        query_xyz, query_radius, query_polar, query_azimuth = _query_xyz_rpa(flat_sample_points, coordinate_fields)
        query_meridian = np.column_stack((query_radius, query_polar))
        resolved_method = _resolve_meridian_resample_method(method)
        output_variables = self._resolve_output_variables(coordinate_fields, fields)
        out_points = np.full((flat_sample_points.shape[0], len(output_variables)), np.nan, dtype=float)
        out_index = {name: index for index, name in enumerate(output_variables)}
        self._populate_resampled_points(
            out_points,
            out_index,
            output_variables,
            query_xyz,
            query_radius,
            query_polar,
            query_azimuth,
            query_meridian,
            method=resolved_method,
            fill_value=fill_value,
        )
        dataset = self._build_sampled_dataset(
            out_points,
            sample_shape,
            output_variables,
            corners=corners,
            copy_aux=copy_aux,
            title=title,
            zone=zone,
        )
        return SmartDs(
            dataset,
            cache_enabled=self._cache_enabled,
            computation_graph=self.recipe_graph,
        )

    def _resolve_output_variables(
        self,
        coordinate_fields: Sequence[str],
        fields: Sequence[str] | None,
    ) -> list[str]:
        if fields is None:
            return list(self.raw.variables)
        output_variables = list(coordinate_fields)
        for name in fields:
            if name not in output_variables:
                output_variables.append(name)
        return output_variables

    def _populate_resampled_points(
        self,
        out_points: np.ndarray,
        out_index: dict[str, int],
        output_variables: Sequence[str],
        query_xyz: np.ndarray,
        query_radius: np.ndarray,
        query_polar: np.ndarray,
        query_azimuth: np.ndarray,
        query_meridian: np.ndarray,
        *,
        method: str,
        fill_value: float,
    ) -> None:
        _assign_direct_query_coordinates(
            out_points,
            out_index,
            query_xyz,
            query_radius,
            query_polar,
            query_azimuth,
            body_radius_m=_body_radius_from_ds(self),
        )
        handled_fields = _direct_query_field_names(out_index)
        handled_fields |= self._populate_rotated_vector_fields(
            out_points,
            out_index,
            output_variables,
            query_azimuth,
            query_meridian,
            method=method,
            fill_value=fill_value,
        )
        for name in output_variables:
            if name in handled_fields:
                continue
            out_points[:, out_index[name]] = self._interpolate_meridional_field(
                name,
                query_meridian,
                method=method,
                fill_value=fill_value,
            )

    def _populate_rotated_vector_fields(
        self,
        out_points: np.ndarray,
        out_index: dict[str, int],
        output_variables: Sequence[str],
        query_azimuth: np.ndarray,
        query_meridian: np.ndarray,
        *,
        method: str,
        fill_value: float,
    ) -> set[str]:
        handled_fields: set[str] = set()
        delta_azimuth = query_azimuth - self.reference_azimuth_rad
        cos_delta = np.cos(delta_azimuth)
        sin_delta = np.sin(delta_azimuth)
        available_names = set(self)
        for x_name, y_name, z_name in _requested_cartesian_vector_triplets(output_variables, available_names):
            vx_ref = self._interpolate_meridional_field(
                x_name,
                query_meridian,
                method=method,
                fill_value=fill_value,
            )
            vy_ref = self._interpolate_meridional_field(
                y_name,
                query_meridian,
                method=method,
                fill_value=fill_value,
            )
            vz = self._interpolate_meridional_field(
                z_name,
                query_meridian,
                method=method,
                fill_value=fill_value,
            )
            if x_name in out_index:
                out_points[:, out_index[x_name]] = cos_delta * vx_ref - sin_delta * vy_ref
            if y_name in out_index:
                out_points[:, out_index[y_name]] = sin_delta * vx_ref + cos_delta * vy_ref
            if z_name in out_index:
                out_points[:, out_index[z_name]] = vz
            handled_fields.update({x_name, y_name, z_name})
        return handled_fields

    def _build_sampled_dataset(
        self,
        out_points: np.ndarray,
        sample_shape: tuple[int, ...],
        output_variables: Sequence[str],
        *,
        corners,
        copy_aux: bool,
        title: str | None,
        zone: str | None,
    ) -> Dataset:
        if corners is None:
            corners_arr = np.empty((0, 0), dtype=int)
        else:
            corners_arr = np.asarray(corners)
        aux = deepcopy(self.raw.aux) if copy_aux else self.raw.aux
        out_title = self.raw.title if title is None else title
        out_zone = f"{self.raw.zone} (resampled)" if zone is None else zone
        return Dataset(
            out_points.reshape(*sample_shape, len(output_variables)),
            corners_arr,
            aux,
            out_title,
            list(output_variables),
            out_zone,
        )

    def _interpolate_meridional_field(
        self,
        name: str,
        query_meridian: np.ndarray,
        *,
        method: str,
        fill_value: float,
    ) -> np.ndarray:
        values = np.asarray(self[name], dtype=float).reshape(-1)
        if values.shape[0] != self._meridian_source_coords.shape[0]:
            raise ValueError(
                f"Field {name!r} has length {values.shape[0]} but meridional coordinates "
                f"have length {self._meridian_source_coords.shape[0]}"
            )
        valid = self._meridian_coord_mask & np.isfinite(values)
        if not np.any(valid):
            return np.full(query_meridian.shape[0], float(fill_value), dtype=float)
        if method == "nearest":
            tree = self._meridian_spatial_cache["nearest_tree"]
            if tree is None or not np.array_equal(valid, self._meridian_coord_mask):
                tree = cKDTree(self._meridian_source_coords[valid])
                if np.array_equal(valid, self._meridian_coord_mask):
                    self._meridian_spatial_cache["nearest_tree"] = tree
            nearest_indices = tree.query(query_meridian)[1]
            return values[valid][nearest_indices]
        if method == "linear":
            if np.array_equal(valid, self._meridian_coord_mask):
                triangulation = self._meridian_spatial_cache["linear_triangulation"]
                if triangulation is None:
                    triangulation = Delaunay(self._meridian_source_coords[valid])
                    self._meridian_spatial_cache["linear_triangulation"] = triangulation
                interpolator = LinearNDInterpolator(triangulation, values[valid], fill_value=fill_value)
            else:
                interpolator = LinearNDInterpolator(
                    self._meridian_source_coords[valid],
                    values[valid],
                    fill_value=fill_value,
                )
            out = np.asarray(interpolator(query_meridian), dtype=float)
            if out.ndim == 0:
                out = out[np.newaxis]
            return out
        raise ValueError(f"Unsupported wedge resample method {method!r}")


def revolve_axisymmetric_wedge_to_volume(
    smart_ds: AxisymmetricWedgeDs,
    *,
    n_radius: int = 48,
    n_polar: int = 48,
    n_azimuth: int = 96,
    radius_range: tuple[float, float] | None = None,
    polar_range: tuple[float, float] | None = None,
    fields: Sequence[str] | None = None,
    method: str = "auto",
    fill_value: float = np.nan,
    title: str | None = None,
    zone: str | None = None,
) -> SmartDs:
    """
    Revolve one axisymmetric wedge adapter onto a structured periodic 3D volume.

    The logical grid is structured in ``(radius, azimuth, polar)`` so the
    longitude sweep is the backend periodic axis. The nodes are converted to
    Cartesian coordinates only at the mesh nodes. This preserves axisymmetry
    far better than voxelizing onto one Cartesian box grid.
    """
    if int(n_radius) < 2 or int(n_polar) < 2 or int(n_azimuth) < 3:
        raise ValueError("n_radius >= 2, n_polar >= 2, and n_azimuth >= 3 are required")

    default_radius_range, default_polar_range = infer_axisymmetric_spherical_ranges(smart_ds)
    radius_range = default_radius_range if radius_range is None else radius_range
    polar_range = default_polar_range if polar_range is None else polar_range

    radius_nodes = np.linspace(float(radius_range[0]), float(radius_range[1]), int(n_radius))
    polar_nodes = np.linspace(float(polar_range[0]), float(polar_range[1]), int(n_polar))
    azimuth_nodes = np.linspace(0.0, 2.0 * np.pi, int(n_azimuth), endpoint=False)
    rr, aa, pp = np.meshgrid(radius_nodes, azimuth_nodes, polar_nodes, indexing="ij")
    xx, yy, zz = _rpa_to_cartesian(rr, pp, aa)
    sample_points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    corners = _swept_hex_corners(int(n_radius), int(n_azimuth), int(n_polar))

    if fields is None:
        source_fields = tuple(smart_ds.raw.variables)
    else:
        source_fields = smart_ds.source_fields(tuple(dict.fromkeys(fields)))

    out_title = smart_ds.raw.title if title is None else title
    out_zone = f"{smart_ds.raw.zone} (revolved axisymmetric volume)" if zone is None else zone
    volume_ds = smart_ds.resample(
        sample_points,
        coordinate_fields=DEFAULT_XYZ_NAMES,
        fields=source_fields,
        method=method,
        fill_value=fill_value,
        corners=corners,
        title=out_title,
        zone=out_zone,
    )
    value_columns = [
        index for index, name in enumerate(volume_ds.raw.variables) if name not in DEFAULT_XYZ_NAMES
    ]
    if value_columns:
        value_points = np.asarray(volume_ds.raw.points, dtype=float)
        missing = ~np.isfinite(value_points[:, value_columns]).all(axis=1)
        if np.any(missing):
            nearest_ds = smart_ds.resample(
                sample_points[missing],
                coordinate_fields=DEFAULT_XYZ_NAMES,
                fields=source_fields,
                method="nearest",
                fill_value=fill_value,
                title=out_title,
                zone=out_zone,
            )
            filled_points = np.array(volume_ds.raw.points, copy=True)
            filled_points[missing] = nearest_ds.raw.points
            volume_ds = SmartDs(
                Dataset(
                    filled_points,
                    np.asarray(corners, dtype=int),
                    dict(volume_ds.raw.aux),
                    out_title,
                    list(volume_ds.raw.variables),
                    out_zone,
                ),
                cache_enabled=smart_ds._cache_enabled,
                computation_graph=smart_ds.recipe_graph,
            )
    return volume_ds


def infer_axisymmetric_spherical_ranges(
    smart_ds: AxisymmetricWedgeDs,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Infer one safe structured ``(radius, polar)`` sweep domain from the meridian.

    Exact poles are excluded so the revolved hexahedra do not collapse there.
    """
    radius = np.asarray(smart_ds["R [R]"], dtype=float)
    polar = np.asarray(smart_ds["polar [rad]"], dtype=float)
    finite_radius = radius[np.isfinite(radius)]
    finite_polar = polar[np.isfinite(polar)]
    if finite_radius.size == 0 or finite_polar.size == 0:
        raise ValueError("Axisymmetric wedge data have no finite spherical coordinates")

    positive_polar = np.sort(np.unique(finite_polar[finite_polar > 0.0]))
    below_pi_polar = np.sort(np.unique(finite_polar[finite_polar < np.pi]))
    if positive_polar.size == 0 or below_pi_polar.size == 0:
        raise ValueError("Axisymmetric wedge data do not span a usable non-degenerate polar range")

    polar_min = 0.5 * float(positive_polar[0])
    polar_max = np.pi - 0.5 * float(np.pi - below_pi_polar[-1])
    return (float(np.min(finite_radius)), float(np.max(finite_radius))), (polar_min, polar_max)


def _merge_nearby_param_in_aux(dataset: Dataset, file_path: str | Path) -> Dataset:
    param_path = find_param_in(file_path)
    if param_path is None:
        return dataset
    param_in = ParamIn.from_file(param_path)
    star_component = _infer_param_component(param_in, file_path, StarParams.command)
    transition_region_component = _infer_param_component(
        param_in,
        file_path,
        TransitionRegionParams.command,
    )
    star_aux = StarParams.from_param_in(param_in, component=star_component)
    transition_region = TransitionRegionParams.from_param_in(
        param_in,
        component=transition_region_component,
    )
    aux_values = {}
    if star_aux is not None:
        if star_aux.name is not None:
            aux_values["Star_name"] = star_aux.name
        aux_values |= {
            "Star_radius_m": star_aux.radius,
            "Star_mass_kg": star_aux.mass,
            "Star_rotational_period_s": star_aux.rotational_period,
            "Star_rotation_rate_rad_s": star_aux.rotation_rate,
        }
    if transition_region is not None:
        aux_values |= {
            "DoExtendTransitionRegion": transition_region.do_extend,
            "TeTransitionRegionSi": transition_region.temperature,
            "DeltaTeModSi": transition_region.delta_temperature,
        }
    if not aux_values:
        return dataset
    return Dataset(
        dataset.points,
        dataset.corners,
        dict(dataset.aux) | aux_values,
        dataset.title,
        list(dataset.variables),
        dataset.zone,
    )


def _infer_coordinate_fields(variable_names: Sequence[str], ndim: int) -> tuple[str, ...]:
    coordinate_fields = tuple(name for name in DEFAULT_XYZ_NAMES if name in variable_names)
    if len(coordinate_fields) < ndim:
        raise ValueError("Could not infer coordinate fields. Pass coordinate_fields explicitly.")
    return coordinate_fields[:ndim]


def _query_xyz_rpa(
    sample_points: np.ndarray,
    coordinate_fields: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coordinate_fields = tuple(coordinate_fields)
    if coordinate_fields == tuple(DEFAULT_XYZ_NAMES):
        xyz = np.asarray(sample_points, dtype=float)
        radius, polar, azimuth = _cartesian_to_rpa(xyz[:, 0], xyz[:, 1], xyz[:, 2])
        return xyz, radius, polar, azimuth
    if coordinate_fields == AxisymmetricWedgeDs._RPA_COORD_FIELDS:
        radius = np.asarray(sample_points[:, 0], dtype=float)
        polar = np.asarray(sample_points[:, 1], dtype=float)
        azimuth = np.asarray(sample_points[:, 2], dtype=float)
        x, y, z = _rpa_to_cartesian(radius, polar, azimuth)
        return np.column_stack((x, y, z)), radius, polar, azimuth
    raise ValueError(
        "AxisymmetricWedgeDs.resample requires coordinate_fields to be "
        "('X [R]', 'Y [R]', 'Z [R]') or ('R [R]', 'polar [rad]', 'azimuth [rad]')."
    )


def _resolve_meridian_resample_method(method: str) -> str:
    if method == "auto":
        return "linear"
    if method == "octree":
        log.debug("Axisymmetric wedge source maps method='octree' onto meridional linear interpolation")
        return "linear"
    if method in {"linear", "nearest"}:
        return method
    raise ValueError("method must be one of ('auto', 'octree', 'linear', 'nearest')")


def _body_radius_from_ds(smart_ds) -> float | None:
    if "RBODY [m]" not in smart_ds:
        return None
    value = np.asarray(smart_ds["RBODY [m]"], dtype=float).reshape(-1)
    if value.size == 0:
        return None
    return float(value[0])


def _assign_direct_query_coordinates(
    out_points: np.ndarray,
    out_index: dict[str, int],
    query_xyz: np.ndarray,
    query_radius: np.ndarray,
    query_polar: np.ndarray,
    query_azimuth: np.ndarray,
    *,
    body_radius_m: float | None,
) -> None:
    direct_fields = {
        "X [R]": query_xyz[:, 0],
        "Y [R]": query_xyz[:, 1],
        "Z [R]": query_xyz[:, 2],
        "R [R]": query_radius,
        "polar [rad]": query_polar,
        "azimuth [rad]": query_azimuth,
        "Lat [deg]": np.rad2deg((0.5 * np.pi) - query_polar),
        "Lon [deg]": np.rad2deg(query_azimuth),
    }
    if body_radius_m is not None:
        direct_fields |= {
            "X [m]": body_radius_m * query_xyz[:, 0],
            "Y [m]": body_radius_m * query_xyz[:, 1],
            "Z [m]": body_radius_m * query_xyz[:, 2],
            "R [m]": body_radius_m * query_radius,
        }
    for name, values in direct_fields.items():
        index = out_index.get(name)
        if index is not None:
            out_points[:, index] = values


def _direct_query_field_names(out_index: dict[str, int]) -> set[str]:
    return set(out_index).intersection(
        {
            "X [R]",
            "Y [R]",
            "Z [R]",
            "R [R]",
            "polar [rad]",
            "azimuth [rad]",
            "Lat [deg]",
            "Lon [deg]",
            "X [m]",
            "Y [m]",
            "Z [m]",
            "R [m]",
        }
    )


def _requested_cartesian_vector_triplets(
    requested_names: Sequence[str],
    available_names: set[str],
) -> list[tuple[str, str, str]]:
    requested_triplets: dict[tuple[str, str], set[str]] = {}
    for name in requested_names:
        match = _VECTOR_COMPONENT_RE.match(name)
        if match is None:
            continue
        key = (match.group("base"), match.group("unit"))
        requested_triplets.setdefault(key, set()).add(match.group("comp"))

    triplets: list[tuple[str, str, str]] = []
    for (base, unit), _components in requested_triplets.items():
        x_name = f"{base}_x {unit}"
        y_name = f"{base}_y {unit}"
        z_name = f"{base}_z {unit}"
        if {x_name, y_name, z_name} <= available_names:
            triplets.append((x_name, y_name, z_name))
    return triplets


def _structured_hex_corners(nx: int, ny: int, nz: int) -> np.ndarray:
    corners = np.empty(((nx - 1) * (ny - 1) * (nz - 1), 8), dtype=int)

    def point_id(ix: int, iy: int, iz: int) -> int:
        return (ix * ny + iy) * nz + iz

    cell_id = 0
    for ix in range(nx - 1):
        for iy in range(ny - 1):
            for iz in range(nz - 1):
                corners[cell_id] = [
                    point_id(ix, iy, iz),
                    point_id(ix + 1, iy, iz),
                    point_id(ix + 1, iy + 1, iz),
                    point_id(ix, iy + 1, iz),
                    point_id(ix, iy, iz + 1),
                    point_id(ix + 1, iy, iz + 1),
                    point_id(ix + 1, iy + 1, iz + 1),
                    point_id(ix, iy + 1, iz + 1),
                ]
                cell_id += 1
    return corners


def _swept_hex_corners(n_radius: int, n_azimuth: int, n_polar: int) -> np.ndarray:
    corners = np.empty(((n_radius - 1) * n_azimuth * (n_polar - 1), 8), dtype=int)

    def point_id(ir: int, ia: int, ip: int) -> int:
        return (ir * n_azimuth + ia) * n_polar + ip

    cell_id = 0
    for ir in range(n_radius - 1):
        for ia in range(n_azimuth):
            for ip in range(n_polar - 1):
                next_ia = (ia + 1) % n_azimuth
                corners[cell_id] = [
                    point_id(ir, ia, ip),
                    point_id(ir + 1, ia, ip),
                    point_id(ir + 1, next_ia, ip),
                    point_id(ir, next_ia, ip),
                    point_id(ir, ia, ip + 1),
                    point_id(ir + 1, ia, ip + 1),
                    point_id(ir + 1, next_ia, ip + 1),
                    point_id(ir, next_ia, ip + 1),
                ]
                cell_id += 1
    return corners


def read_wedge_meridian(
    path: str | Path,
    *,
    pair_path: str | Path | None = None,
    reference_azimuth_rad: float = 0.0,
    group_decimals: int = 6,
) -> WedgeMeridianData:
    """
    Read one dual-3D wedge timestep and collapse its cell-centre data over longitude.

    The larger-point file is treated as the geometry/corner source and the smaller
    file as the cell-centre value source. Vector triplets named like ``U_x/U_y/U_z``
    or ``B_x/B_y/B_z`` are rotated around ``+z`` onto one common longitude before
    averaging.
    """
    geometry_path, cell_center_path = resolve_wedge_pair_paths(path, pair_path=pair_path)
    geometry = _read_plt_dataset(geometry_path)
    cell_centers = _read_plt_dataset(cell_center_path)
    _validate_wedge_pair(geometry_path, geometry, cell_center_path, cell_centers)

    reduced_points, point_group_ids, sample_counts = _reduce_cell_center_points(
        cell_centers,
        reference_azimuth_rad=float(reference_azimuth_rad),
        group_decimals=int(group_decimals),
    )
    reduced = Dataset(
        reduced_points,
        np.empty((0, 0), dtype=int),
        dict(cell_centers.aux),
        cell_centers.title,
        list(cell_centers.variables),
        f"{cell_centers.zone} (lon-averaged meridian)",
    )
    return WedgeMeridianData(
        geometry_path=geometry_path,
        cell_center_path=cell_center_path,
        geometry=geometry,
        cell_centers=cell_centers,
        reduced_cell_centers=reduced,
        reference_azimuth_rad=float(reference_azimuth_rad),
        point_group_ids=point_group_ids,
        sample_counts=sample_counts,
    )


def plot_wedge_meridional_field(
    wedge_data: WedgeMeridianData,
    *,
    field: str = "Rho [g/cm^3]",
    output_path: str | Path | None = None,
    title: str | None = None,
    cmap: str = "viridis",
    log_scale: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot one longitude-collapsed meridional field in the ``(R_cyl, Z)`` plane.

    The reduced wedge data currently exist at cell centres, so this quicklook uses
    one triangulated pseudocolour plot rather than one strict rectilinear mesh.
    """
    reduced = wedge_data.reduced_cell_centers
    if field not in reduced.variables:
        raise ValueError(f"Field {field!r} not present in reduced wedge data")

    x = np.asarray(reduced["X [R]"], dtype=float)
    y = np.asarray(reduced["Y [R]"], dtype=float)
    z = np.asarray(reduced["Z [R]"], dtype=float)
    values = np.asarray(reduced[field], dtype=float)
    cyl_radius = np.sqrt(x * x + y * y)

    mask = np.isfinite(cyl_radius) & np.isfinite(z) & np.isfinite(values)
    cyl_radius = cyl_radius[mask]
    z = z[mask]
    values = values[mask]
    positive = values[values > 0.0]
    norm = None
    if bool(log_scale) and positive.size > 0:
        norm = LogNorm(vmin=float(np.min(positive)), vmax=float(np.max(positive)))

    figure, axis = plt.subplots(figsize=(6.8, 8.0), constrained_layout=True)
    if values.size >= 3:
        triangulation = Triangulation(cyl_radius, z)
        image = axis.tripcolor(triangulation, values, shading="gouraud", cmap=cmap, norm=norm, rasterized=True)
    else:
        image = axis.scatter(cyl_radius, z, c=values, cmap=cmap, norm=norm, s=36.0)
    axis.set_aspect("equal")
    axis.set_xlabel(r"$R_{\mathrm{cyl}}$ $(R_\star)$")
    axis.set_ylabel(r"$z$ $(R_\star)$")
    axis.set_title(field if title is None else str(title))
    axis.grid(True, color="0.9", linewidth=0.6)

    star_z = np.linspace(-1.0, 1.0, 401, dtype=float)
    star_r = np.sqrt(np.clip(1.0 - star_z * star_z, 0.0, None))
    axis.plot(star_r, star_z, color="white", linewidth=1.8)
    axis.plot(star_r, star_z, color="black", linewidth=0.8)

    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label(field)
    if output_path is not None:
        figure.savefig(output_path, dpi=180)
    return figure, axis


def resolve_wedge_pair_paths(
    path: str | Path,
    *,
    pair_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """
    Resolve the two same-timestep wedge files and return ``(geometry, cell_centres)``.
    """
    first_path = Path(path)
    if pair_path is not None:
        second_path = Path(pair_path)
        if first_path.resolve() == second_path.resolve():
            raise ValueError("Wedge pair paths must refer to two distinct files")
        pair = (first_path, second_path)
    else:
        pair = find_wedge_pair_paths(first_path)

    first = _read_plt_dataset(pair[0])
    second = _read_plt_dataset(pair[1])
    return _classify_wedge_pair(pair[0], first, pair[1], second)


def find_wedge_pair_paths(path: str | Path) -> tuple[Path, Path]:
    """
    Find the two ``3d__var_*`` wedge files for one iteration in the same directory.
    """
    base_path = Path(path)
    token = iteration_token_from_path(base_path)
    if token is None:
        raise ValueError(f"Could not extract one BATSRUS iteration token from {base_path}")
    siblings = sorted(
        candidate
        for candidate in base_path.parent.glob(f"3d__var_*_{token}.plt")
        if candidate.is_file()
    )
    if len(siblings) != 2:
        raise ValueError(
            f"Expected exactly two same-timestep wedge files for {token} beside {base_path.name}, found {len(siblings)}"
        )
    return siblings[0], siblings[1]


def _read_plt_dataset(path: str | Path) -> Dataset:
    points, corners, aux, title, variables, zone = read_plt(str(path))
    return Dataset(points, corners, aux, title, variables, zone)


def _classify_wedge_pair(
    first_path: Path,
    first: Dataset,
    second_path: Path,
    second: Dataset,
) -> tuple[Path, Path]:
    first_points = int(first.points.shape[0])
    second_points = int(second.points.shape[0])
    if first_points == second_points:
        raise ValueError(
            f"Wedge pair {first_path.name} and {second_path.name} have the same point count {first_points}; "
            "cannot choose geometry vs cell-centre file"
        )
    if first_points > second_points:
        return first_path, second_path
    return second_path, first_path


def _validate_wedge_pair(
    geometry_path: Path,
    geometry: Dataset,
    cell_center_path: Path,
    cell_centers: Dataset,
) -> None:
    if geometry.corners is None:
        raise ValueError(f"Geometry file {geometry_path.name} has no corners")
    if cell_centers.corners is None:
        raise ValueError(f"Cell-centre file {cell_center_path.name} has no corners array")
    if list(geometry.variables) != list(cell_centers.variables):
        raise ValueError("Geometry and cell-centre wedge files must define the same variables")
    if geometry.corners.shape[0] != cell_centers.points.shape[0]:
        raise ValueError(
            f"Geometry file {geometry_path.name} has {geometry.corners.shape[0]} cells but "
            f"cell-centre file {cell_center_path.name} has {cell_centers.points.shape[0]} points"
        )


def _reduce_cell_center_points(
    cell_centers: Dataset,
    *,
    reference_azimuth_rad: float,
    group_decimals: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(cell_centers.points, dtype=float)
    variables = list(cell_centers.variables)
    x = np.asarray(cell_centers["X [R]"], dtype=float)
    y = np.asarray(cell_centers["Y [R]"], dtype=float)
    z = np.asarray(cell_centers["Z [R]"], dtype=float)
    radius, polar, azimuth = _cartesian_to_rpa(x, y, z)

    group_keys = np.column_stack([np.round(radius, group_decimals), np.round(polar, group_decimals)])
    _unique_keys, point_group_ids = np.unique(group_keys, axis=0, return_inverse=True)
    sample_counts = np.bincount(point_group_ids, minlength=int(np.max(point_group_ids)) + 1).astype(int, copy=False)
    n_groups = int(sample_counts.size)

    reduced = np.zeros((n_groups, points.shape[1]), dtype=float)
    mean_radius = _group_mean(radius, point_group_ids, n_groups)
    mean_polar = _group_mean(polar, point_group_ids, n_groups)
    ref_azimuth = np.full(n_groups, float(reference_azimuth_rad), dtype=float)
    mean_x, mean_y, mean_z = _rpa_to_cartesian(mean_radius, mean_polar, ref_azimuth)

    variable_index = {name: index for index, name in enumerate(variables)}
    vector_triplets = _cartesian_vector_triplets(variables)
    handled = set()

    reduced[:, variable_index["X [R]"]] = mean_x
    reduced[:, variable_index["Y [R]"]] = mean_y
    reduced[:, variable_index["Z [R]"]] = mean_z
    handled.update({"X [R]", "Y [R]", "Z [R]"})

    delta = float(reference_azimuth_rad) - azimuth
    cos_delta = np.cos(delta)
    sin_delta = np.sin(delta)

    for x_name, y_name, z_name in vector_triplets:
        vx = np.asarray(cell_centers[x_name], dtype=float)
        vy = np.asarray(cell_centers[y_name], dtype=float)
        vz = np.asarray(cell_centers[z_name], dtype=float)
        rotated_x = cos_delta * vx - sin_delta * vy
        rotated_y = sin_delta * vx + cos_delta * vy
        reduced[:, variable_index[x_name]] = _group_mean(rotated_x, point_group_ids, n_groups)
        reduced[:, variable_index[y_name]] = _group_mean(rotated_y, point_group_ids, n_groups)
        reduced[:, variable_index[z_name]] = _group_mean(vz, point_group_ids, n_groups)
        handled.update({x_name, y_name, z_name})

    for name in variables:
        if name in handled:
            continue
        reduced[:, variable_index[name]] = _group_mean(np.asarray(cell_centers[name], dtype=float), point_group_ids, n_groups)

    return reduced, point_group_ids, sample_counts


def _cartesian_vector_triplets(variables: list[str]) -> list[tuple[str, str, str]]:
    triplets: list[tuple[str, str, str]] = []
    variable_set = set(variables)
    for name in variables:
        match = _VECTOR_X_RE.match(name)
        if match is None:
            continue
        y_name = f"{match.group('base')}_y {match.group('unit')}"
        z_name = f"{match.group('base')}_z {match.group('unit')}"
        if y_name in variable_set and z_name in variable_set:
            triplets.append((name, y_name, z_name))
    return triplets


def _group_mean(values: np.ndarray, group_ids: np.ndarray, n_groups: int) -> np.ndarray:
    sums = np.zeros(n_groups, dtype=float)
    np.add.at(sums, group_ids, np.asarray(values, dtype=float))
    counts = np.bincount(group_ids, minlength=n_groups).astype(float, copy=False)
    return sums / counts


def _cartesian_to_rpa(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radius = np.sqrt(x * x + y * y + z * z)
    safe_radius = np.maximum(radius, np.finfo(float).tiny)
    polar = np.arccos(np.clip(z / safe_radius, -1.0, 1.0))
    azimuth = np.arctan2(y, x)
    return radius, polar, azimuth


def _rpa_to_cartesian(radius: np.ndarray, polar: np.ndarray, azimuth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sin_polar = np.sin(polar)
    x = radius * sin_polar * np.cos(azimuth)
    y = radius * sin_polar * np.sin(azimuth)
    z = radius * np.cos(polar)
    return x, y, z
