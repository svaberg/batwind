from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
import logging
from time import perf_counter

import griblet
import numpy as np

from batread import Dataset
from batwind.data.field_names import DEFAULT_XYZ_NAMES
from batwind.param_in import find_param_in
from batwind.param_in import ParamIn
from batwind.param_in import StarParams
from batwind._smart_ds_resample import resample_smart_ds
from batwind.recipes.batsrus import build_batsrus_graph
from batwind.recipes.spherical import build_spherical_graph

log = logging.getLogger(__name__)


class SmartDs:
    """
    Lightweight wrapper around ``batread.Dataset`` with graph-backed derived fields.
    """

    DEFAULT_COORD_FIELDS = DEFAULT_XYZ_NAMES

    @staticmethod
    def _resolve_resample_method(sample_points, method: str) -> str:
        ndim = np.asarray(sample_points, dtype=float).shape[-1]
        if method == "auto":
            if ndim == 3:
                return "octree"
            if ndim == 2:
                return "linear"
            raise ValueError(f"method='auto' does not support ndim={ndim}")
        return method

    def __init__(
        self,
        dataset: Dataset,
        *,
        cache_enabled: bool = True,
        computation_graph: griblet.Graph | None = None,
    ) -> None:
        self._dataset = dataset
        self._cache_enabled = bool(cache_enabled)
        self._cache: dict[str, np.ndarray] = {}
        self._resample_spatial_cache: dict[tuple[str, ...], dict[str, object]] = {}
        self._source_graph = self._build_source_graph()
        self._recipe_graph = griblet.Graph()
        self._computation_graph_cache: griblet.Graph | None = None
        if computation_graph is not None:
            self.merge_computation_graph(computation_graph)
        log.debug(
            "SmartDs.__init__ points=%s variables=%d cache_enabled=%s",
            np.shape(self._dataset.points),
            len(self._dataset.variables),
            self._cache_enabled,
        )

    @classmethod
    def from_file(
        cls,
        file: str,
        *,
        batsrus: bool = True,
        spherical: bool = True,
        body_radius_m: float | None = None,
        **kwargs,
    ) -> "SmartDs":
        log.info("SmartDs.from_file...")
        stage_start = perf_counter()
        log.debug(
            "SmartDs.from_file file=%s batsrus=%s spherical=%s body_radius_arg=%s",
            file,
            batsrus,
            spherical,
            body_radius_m is not None,
        )
        raw = Dataset.from_file(str(file))
        star_aux = None
        param_path = find_param_in(file)
        if param_path is not None:
            star_aux = StarParams.from_param_in(ParamIn.from_file(param_path))
        if star_aux is not None:
            aux_values = {
                "Star_name": star_aux.name,
                "Star_radius_m": star_aux.radius,
                "Star_mass_kg": star_aux.mass,
                "Star_rotational_period_s": star_aux.rotational_period,
                "Star_rotation_rate_rad_s": star_aux.rotation_rate,
            }
            log.debug("SmartDs.from_file merged nearby PARAM.in star aux keys=%d", len(aux_values))
            raw = Dataset(
                raw.points,
                raw.corners,
                dict(raw.aux) | aux_values,
                raw.title,
                raw.variables,
                raw.zone,
            )
        else:
            log.debug("SmartDs.from_file found no nearby PARAM.in star aux")
        if body_radius_m is None:
            radius_from_param = raw.aux.get("Star_radius_m")
            if radius_from_param is not None:
                body_radius_m = float(radius_from_param)
                log.debug("SmartDs.from_file using Star_radius_m as body_radius_m")
            else:
                log.debug("SmartDs.from_file has no body_radius_m source")
        else:
            log.debug("SmartDs.from_file using explicit body_radius_m")
        sds = cls(raw, **kwargs)
        if batsrus:
            log.debug("SmartDs.from_file merging BATSRUS graph")
            sds.merge_computation_graph(
                build_batsrus_graph(
                    sds.raw.variables,
                    gamma=sds.raw.aux.get("GAMMA"),
                    body_radius_m=body_radius_m,
                )
            )
        if spherical:
            log.debug("SmartDs.from_file merging spherical graph")
            sds.merge_computation_graph(build_spherical_graph(tuple(sds)))
        log.debug("SmartDs.from_file complete in %.2f s.", perf_counter() - stage_start)
        return sds

    @property
    def raw(self) -> Dataset:
        return self._dataset

    def __repr__(self) -> str:
        return (
            f"SmartDs(title={self._dataset.title!r}, zone={self._dataset.zone!r}, "
            f"points={np.shape(self._dataset.points)}, variables={len(self._dataset.variables)})"
        )

    def __str__(self) -> str:
        return "\n".join(
            (
                "SmartDs",
                f"  Title: {self._dataset.title}",
                f"  Zone : {self._dataset.zone}",
                f"  Points: {np.shape(self._dataset.points)}",
                f"  Variables: {len(self._dataset.variables)}",
            )
        )

    @property
    def computation_graph(self):
        if self._computation_graph_cache is None:
            graph = griblet.Graph(self._source_graph)
            graph.merge(self._recipe_graph)
            self._computation_graph_cache = graph
        return self._computation_graph_cache

    @property
    def recipe_graph(self) -> griblet.Graph:
        return self._recipe_graph

    def __iter__(self):
        names = list(self._dataset.variables)
        for name in self.computation_graph.fields():
            if name not in names:
                names.append(name)
        return iter(names)

    def __getitem__(self, index_or_name):
        if not isinstance(index_or_name, str):
            return self._dataset[index_or_name]

        name = index_or_name
        if self._cache_enabled and name in self._cache:
            log.debug("SmartDs.__getitem__ cache hit field=%s", name)
            return self._cache[name]

        if name in self._dataset.variables:
            log.debug("SmartDs.__getitem__ raw field=%s", name)
            value = self._dataset[name]
            if self._cache_enabled:
                self._cache[name] = value
                log.debug("SmartDs.__getitem__ cached raw field=%s", name)
            return value

        try:
            path = self.computation_graph.path(name)
        except (KeyError, griblet.NoPathError) as e:
            log.debug("SmartDs.__getitem__ unresolved field=%s", name)
            raise IndexError(
                f"Field '{name}' not available. Raw fields: {self._dataset.variables}."
            ) from e
        cost = path.cost
        if not np.isfinite(cost):
            log.debug("SmartDs.__getitem__ non-finite resolve cost field=%s cost=%s", name, cost)
            raise IndexError(
                f"Field '{name}' not available. Raw fields: {self._dataset.variables}."
            )
        log.debug("SmartDs.__getitem__ graph resolve field=%s cost=%s", name, cost)
        value = self.computation_graph.compute(path)
        if self._cache_enabled:
            self._cache[name] = value
            log.debug("SmartDs.__getitem__ cached derived field=%s", name)
        return value

    def _build_source_graph(self) -> griblet.Graph:
        graph = griblet.Graph()
        for raw_name in self._dataset.variables:
            graph.add(
                raw_name,
                lambda raw_name=raw_name: self._dataset[raw_name],
                needs=[],
                cost=0.0,
                metadata={"description": "Dataset raw field", "loader": True},
            )
        for key, value in self._dataset.aux.items():
            graph.add(
                key,
                lambda value=value: value,
                needs=[],
                cost=0.0,
                metadata={"description": "Dataset aux", "loader": True},
            )
        return graph

    def _invalidate_computation_graph(self) -> None:
        self._computation_graph_cache = None

    def clear_computation_graph(self):
        log.debug(
            "SmartDs.clear_computation_graph recipe_fields=%d",
            len(self._recipe_graph.fields()),
        )
        self._recipe_graph = griblet.Graph()
        self._invalidate_computation_graph()
        return self

    def merge_computation_graph(self, graph):
        self._recipe_graph.merge(graph)
        self._invalidate_computation_graph()
        log.debug(
            "SmartDs.merge_computation_graph recipe_fields=%d",
            len(self._recipe_graph.fields()),
        )
        return self

    def clear_cache(self, *names: str) -> None:
        if not names:
            self._cache.clear()
            self._resample_spatial_cache.clear()
            log.debug("SmartDs.clear_cache cleared all caches")
            return
        for name in names:
            self._cache.pop(name, None)
        log.debug("SmartDs.clear_cache cleared fields=%s", names)

    def source_fields(self, fields: Sequence[str]) -> tuple[str, ...]:
        unique_fields = tuple(dict.fromkeys(fields))
        log.debug("SmartDs.source_fields requested_fields=%d", len(unique_fields))
        base_fields: list[str] = []
        graph = self.computation_graph

        for field in unique_fields:
            if field in self._dataset.variables:
                base_fields.append(field)
                log.debug("SmartDs.source_fields raw field=%s", field)
                continue
            try:
                path = graph.path(field)
            except (KeyError, griblet.NoPathError):
                if field not in base_fields:
                    base_fields.append(field)
                log.debug("SmartDs.source_fields unresolved passthrough=%s", field)
                continue
            stack = [path]
            leaves: list[str] = []
            while stack:
                node = stack.pop()
                if node.needs:
                    stack.extend(reversed(node.needs))
                    continue
                if node.name in self._dataset.variables and node.name not in leaves:
                    leaves.append(node.name)
            for name in leaves or [field]:
                if name not in base_fields:
                    base_fields.append(name)
            log.debug("SmartDs.source_fields field=%s leaves=%s", field, tuple(leaves) or (field,))

        log.debug("SmartDs.source_fields complete source_fields=%d", len(base_fields))
        return tuple(base_fields)

    def resample(
        self,
        sample_points,
        *,
        coordinate_fields: Sequence[str] | None = None,
        fields: Sequence[str] | None = None,
        # Method guidance:
        # - `auto` resolves to `octree` for 3D resampling.
        # - `auto` resolves to `linear` for 2D resampling and structured datasets.
        # - `nearest` is mainly for exposing the underlying grid resolution.
        method: str = "auto",
        fill_value: float = np.nan,
        corners=None,
        copy_aux: bool = True,
        title: str | None = None,
        zone: str | None = None,
    ) -> "SmartDs":
        log.info("SmartDs.resample...")
        stage_start = perf_counter()
        inferred_coordinate_fields = coordinate_fields is None
        if coordinate_fields is None:
            preferred = DEFAULT_XYZ_NAMES
            ndim = np.asarray(sample_points, dtype=float).shape[-1]
            coordinate_fields = tuple(name for name in preferred if name in self._dataset.variables)
            if len(coordinate_fields) < ndim:
                raise ValueError(
                    "Could not infer coordinate fields. Pass coordinate_fields explicitly."
                )
            coordinate_fields = coordinate_fields[:ndim]
            log.debug("SmartDs.resample inferred coordinate_fields=%s ndim=%d", coordinate_fields, ndim)
        resolved_method = self._resolve_resample_method(sample_points, method)
        log.debug(
            "SmartDs.resample method=%s resolved_method=%s coordinate_fields=%s explicit_fields=%s inferred_coordinate_fields=%s",
            method,
            resolved_method,
            coordinate_fields,
            fields is not None,
            inferred_coordinate_fields,
        )
        out = resample_smart_ds(
            self,
            sample_points,
            coordinate_fields=coordinate_fields,
            fields=fields,
            method=resolved_method,
            fill_value=fill_value,
            corners=corners,
            copy_aux=copy_aux,
            title=title,
            zone=zone,
        )
        log.debug("SmartDs.resample complete in %.2f s.", perf_counter() - stage_start)
        return out

    def append_fields(
        self,
        extra_fields: Mapping[str, np.ndarray],
        *,
        zone_suffix: str = "derived fields",
    ) -> "SmartDs":
        if not extra_fields:
            return self
        log.debug("SmartDs.append_fields fields=%d zone_suffix=%s", len(extra_fields), zone_suffix)

        base_points = np.asarray(self.raw.points)
        if base_points.ndim < 2:
            raise ValueError("Expected raw points to have shape (..., nvars)")
        base_shape = base_points.shape[:-1]

        arrays = []
        names = []
        for name, values in extra_fields.items():
            arr = np.asarray(values)
            if arr.shape != base_shape:
                raise ValueError(
                    f"Extra field '{name}' shape {arr.shape} does not match dataset grid shape {base_shape}"
                )
            arrays.append(arr[..., None])
            names.append(name)

        new_points = np.concatenate([base_points, *arrays], axis=-1)
        new_dataset = Dataset(
            new_points,
            self.raw.corners,
            self.raw.aux,
            self.raw.title,
            list(self.raw.variables) + names,
            f"{self.raw.zone} ({zone_suffix})",
        )
        out = type(self)(
            new_dataset,
            cache_enabled=self._cache_enabled,
            computation_graph=self._recipe_graph,
        )
        log.debug(
            "SmartDs.append_fields complete new_variables=%d",
            len(out.raw.variables),
        )
        return out


__all__ = ["SmartDs"]
