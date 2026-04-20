from __future__ import annotations

import numpy as np
from batread.dataset import Dataset
from batwind.recipes.batsrus import build_batsrus_graph
from batwind.recipes.spherical import build_spherical_graph
from batwind.smart_ds import SmartDs


def hex_corners(nx: int, ny: int, nz: int) -> np.ndarray:
    corners = []

    def idx(i: int, j: int, k: int) -> int:
        return (i * ny + j) * nz + k

    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                corners.append(
                    [
                        idx(i, j, k),
                        idx(i + 1, j, k),
                        idx(i + 1, j + 1, k),
                        idx(i, j + 1, k),
                        idx(i, j, k + 1),
                        idx(i + 1, j, k + 1),
                        idx(i + 1, j + 1, k + 1),
                        idx(i, j + 1, k + 1),
                    ]
                )
    return np.asarray(corners, dtype=int)


def make_structured_smart_ds(
    columns,
    *,
    n: int,
    variables,
    aux: dict[str, object] | None = None,
    title: str,
    zone: str,
) -> SmartDs:
    smart_ds = SmartDs(
        Dataset(
            np.column_stack(columns),
            hex_corners(n, n, n),
            aux={} if aux is None else dict(aux),
            title=title,
            variables=list(variables),
            zone=zone,
        )
    )
    smart_ds.merge_computation_graph(build_batsrus_graph(smart_ds.raw.variables))
    smart_ds.merge_computation_graph(build_spherical_graph(tuple(smart_ds)))
    return smart_ds


def scalar_mesh_actor(plotter):
    return next(
        actor
        for actor in plotter.actors.values()
        if getattr(getattr(actor, "mapper", None), "scalar_visibility", False)
    )


def scalar_bar_actor(plotter):
    return next(iter(plotter.scalar_bars.values()))
