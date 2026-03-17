from __future__ import annotations

from collections.abc import Mapping
from os import PathLike

import numpy as np
import pyvista as pv
from batread.dataset import Dataset

DataLike = object


def coerce_smart_ds(data: DataLike):
    if isinstance(data, Dataset):
        return data
    if isinstance(data, (str, PathLike)):
        return Dataset.from_file(str(data))
    if _is_dataset_like(data):
        return data
    raise TypeError(
        "Expected a dataset-like object, batread.Dataset, or file path for PyVista conversion; "
        f"got {type(data).__name__}"
    )


def to_unstructured_grid(
    data: DataLike,
    *,
    point_data: Mapping[str, object] | None = None,
) -> pv.UnstructuredGrid:
    sds = coerce_smart_ds(data)
    points = np.asarray(sds.points)
    corners = np.asarray(sds.corners)

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("Expected dataset points with at least three coordinate columns")
    if corners.ndim != 2 or corners.size == 0 or corners.shape[1] == 0:
        raise ValueError("Expected cell connectivity to build a PyVista unstructured grid")

    grid = pv.UnstructuredGrid({pv.CellType.HEXAHEDRON: corners}, points[:, :3])

    for name, values in (point_data or {}).items():
        arr = np.asarray(values)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        grid.point_data[name] = arr

    for name, value in sds.aux.items():
        grid.add_field_data([value], name)

    return grid


def _is_dataset_like(data) -> bool:
    return all(hasattr(data, name) for name in ("variable", "points", "corners", "aux"))


__all__ = ["DataLike", "coerce_smart_ds", "to_unstructured_grid"]
