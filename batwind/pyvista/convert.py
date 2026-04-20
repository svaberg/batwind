from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pyvista as pv
from batwind.smart_ds import SmartDs


def to_unstructured_grid(
    smart_ds: SmartDs,
    *,
    point_data: Mapping[str, object] | None = None,
) -> pv.UnstructuredGrid:
    points = np.asarray(smart_ds.raw.points)
    corners = np.asarray(smart_ds.raw.corners)

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

    for name, value in smart_ds.raw.aux.items():
        grid.add_field_data([value], name)

    return grid


__all__ = ["to_unstructured_grid"]
