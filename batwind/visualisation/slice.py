import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from batwind.utils import auto_coords
from batwind.utils import triangles

log = logging.getLogger(__name__)


def plot_xz_slice_tripcolor_with_marginals(
    ds,
    *,
    var="FNEUTRAL [none]",
    bins_x=200,
    bins_z=200,
    figsize=(9, 7),
    tripcolor_kwargs=None,
):
    log.debug("plot_xz_slice_tripcolor_with_marginals var=%s bins=(%d,%d)", var, bins_x, bins_z)
    if tripcolor_kwargs is None:
        tripcolor_kwargs = {"shading": "flat"}
    norm = tripcolor_kwargs.get("norm")

    u_name, v_name = auto_coords(ds)
    tris = triangles(ds, u_name, v_name)

    w = np.asarray(ds[var]).ravel()

    if w.size != tris.x.size:
        raise ValueError(f"{var} length {w.size} != tris.x length {tris.x.size}")

    x = np.asarray(tris.x)
    z = np.asarray(tris.y)

    m = np.isfinite(x) & np.isfinite(z) & np.isfinite(w)
    x, z, w = x[m], z[m], w[m]

    # remap triangles to masked vertex indexing if needed
    if m.all():
        tris_m = tris
    else:
        old2new = -np.ones(m.size, dtype=int)
        old2new[np.where(m)[0]] = np.arange(m.sum())
        tri = old2new[tris.triangles]
        tri = tri[(tri >= 0).all(axis=1)]
        tris_m = type(tris)(x, z, triangles=tri)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1.2], width_ratios=[1.2, 4])
    ax_left = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[0, 1])
    ax_bottom = fig.add_subplot(gs[1, 1], sharex=ax_main)

    img = ax_main.tripcolor(tris_m, w[m] if not m.all() else w, **tripcolor_kwargs)
    ax_main.set_xlabel(u_name)
    ax_main.set_ylabel(v_name)
    ax_main.set_title(var)

    # binned means vs x and z
    x_edges = np.linspace(x.min(), x.max(), bins_x + 1)
    z_edges = np.linspace(z.min(), z.max(), bins_z + 1)

    xb = np.digitize(x, x_edges) - 1
    zb = np.digitize(z, z_edges) - 1

    vx = (xb >= 0) & (xb < bins_x)
    vz = (zb >= 0) & (zb < bins_z)

    sum_x = np.bincount(xb[vx], weights=w[m][vx] if not m.all() else w[vx], minlength=bins_x)
    cnt_x = np.bincount(xb[vx], minlength=bins_x)
    mean_x = np.full(bins_x, np.nan)
    mean_x[cnt_x > 0] = sum_x[cnt_x > 0] / cnt_x[cnt_x > 0]
    x_cent = 0.5 * (x_edges[:-1] + x_edges[1:])

    sum_z = np.bincount(zb[vz], weights=w[m][vz] if not m.all() else w[vz], minlength=bins_z)
    cnt_z = np.bincount(zb[vz], minlength=bins_z)
    mean_z = np.full(bins_z, np.nan)
    mean_z[cnt_z > 0] = sum_z[cnt_z > 0] / cnt_z[cnt_z > 0]
    z_cent = 0.5 * (z_edges[:-1] + z_edges[1:])

    ax_bottom.plot(x_cent, mean_x)
    ax_bottom.set_ylabel("mean")
    ax_bottom.set_xlabel(u_name)

    ax_left.plot(mean_z, z_cent)
    ax_left.set_xlabel("mean")
    ax_left.set_ylabel(v_name)
    ax_left.invert_xaxis()
    plt.setp(ax_left.get_yticklabels(), visible=False)
    ax_left.tick_params(axis="y", length=0)

    cbar = fig.colorbar(img, ax=[ax_main, ax_left, ax_bottom], location="right")
    cbar.set_label(var)

    log.debug("plot_xz_slice_tripcolor_with_marginals complete")
    return fig, (ax_main, ax_left, ax_bottom), cbar


def plot_xz_slice_tripcolor_with_cross_quantiles(
    ds,
    *,
    var="FNEUTRAL [none]",
    qlevels=(0.16, 0.5, 0.84),
    cut_frac=0.01,
    min_per_bin=50,
    figsize=(10, 8),
    tripcolor_kwargs=None,
):
    log.debug("plot_xz_slice_tripcolor_with_cross_quantiles var=%s qlevels=%s", var, qlevels)
    if tripcolor_kwargs is None:
        tripcolor_kwargs = {"shading": "flat"}
    norm = tripcolor_kwargs.get("norm")

    u_name, v_name = auto_coords(ds)
    tris = triangles(ds, u_name, v_name)
    x = np.asarray(tris.x).ravel()
    z = np.asarray(tris.y).ravel()
    w_full = np.asarray(ds[var]).ravel()

    if w_full.size != x.size:
        raise ValueError(f"{var}: {w_full.size} values, triangulation has {x.size} points")

    # finite mask for *cross* computations
    m = np.isfinite(x) & np.isfinite(z) & np.isfinite(w_full)
    xf, zf, wf = x[m], z[m], w_full[m]

    def _select_near_zero(coord, frac):
        if coord.size == 0:
            return np.zeros(0, dtype=bool)
        k = max(1, int(np.ceil(frac * coord.size)))
        tol = np.partition(np.abs(coord), k - 1)[k - 1]
        return np.abs(coord) <= tol

    def _edges_midway_drophalf(v, min_per_bin_):
        v = np.asarray(v)
        v = v[np.isfinite(v)]
        if v.size < 2:
            return None

        v_sorted = np.sort(v)
        v_unique = np.unique(v_sorted)
        if v_unique.size < 2:
            return None

        n_points = v_sorted.size
        candidates = v_unique

        # drop half of candidate split locations until average occupancy is OK
        while (candidates.size > 2) and (n_points / max(1, (candidates.size - 1)) < min_per_bin_):
            candidates = candidates[::2]

        mids = 0.5 * (candidates[:-1] + candidates[1:])
        edges = np.empty(mids.size + 2, dtype=float)
        edges[0] = candidates[0]
        edges[1:-1] = mids
        edges[-1] = np.nextafter(candidates[-1], np.inf)  # include max
        return edges

    def _binned_quantiles(coord, val, edges, qlevels_):
        nb = edges.size - 1
        centers = 0.5 * (edges[:-1] + edges[1:])
        bi = np.digitize(coord, edges) - 1
        ok = (bi >= 0) & (bi < nb)

        out = []
        for q in qlevels_:
            y = np.full(nb, np.nan)
            for b in range(nb):
                vv = val[ok & (bi == b)]
                if vv.size:
                    y[b] = np.quantile(vv, q)
            out.append(y)
        return centers, out

    # layout
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1.2], width_ratios=[1.2, 4])
    ax_left = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[0, 1])
    ax_bottom = fig.add_subplot(gs[1, 1], sharex=ax_main)

    # main: use original tris + original values (don’t remap triangles here)
    img = ax_main.tripcolor(tris, w_full, **tripcolor_kwargs)
    ax_main.set_xlabel(u_name)
    ax_main.set_ylabel(v_name)
    ax_main.set_title(var)

    # bottom marginal: v≈0 cut, bins in u using your edge rule
    sel_z0 = _select_near_zero(zf, cut_frac)
    xb, wb = xf[sel_z0], wf[sel_z0]
    x_edges = _edges_midway_drophalf(xb, min_per_bin)
    if x_edges is not None:
        x_cent, yqs = _binned_quantiles(xb, wb, x_edges, qlevels)
        for yq in yqs:
            ax_bottom.plot(x_cent, yq)
    ax_bottom.set_xlabel(u_name)
    ax_bottom.set_ylabel("quantiles")
    if isinstance(norm, LogNorm):
        ax_bottom.set_yscale("log")

    # left marginal: u≈0 cut, bins in v using your edge rule
    sel_x0 = _select_near_zero(xf, cut_frac)
    zl, wl = zf[sel_x0], wf[sel_x0]
    z_edges = _edges_midway_drophalf(zl, min_per_bin)
    if z_edges is not None:
        z_cent, xqs = _binned_quantiles(zl, wl, z_edges, qlevels)
        for xq in xqs:
            ax_left.plot(xq, z_cent)
    ax_left.set_xlabel("quantiles")
    ax_left.set_ylabel(v_name)
    if isinstance(norm, LogNorm):
        ax_left.set_xscale("log")
    ax_left.invert_xaxis()
    plt.setp(ax_left.get_yticklabels(), visible=False)
    ax_left.tick_params(axis="y", length=0)

    cbar = fig.colorbar(img, ax=[ax_main, ax_left, ax_bottom], location="right")
    cbar.set_label(var)

    log.debug("plot_xz_slice_tripcolor_with_cross_quantiles complete")
    return fig, (ax_main, ax_left, ax_bottom), cbar


def plot_xz_slice_with_marginal_points(
    ds,
    *,
    var="FNEUTRAL [none]",
    figsize=(10, 8),
    tripcolor_kwargs=None,
    scatter_kwargs=None,
):
    log.debug("plot_xz_slice_with_marginal_points var=%s", var)
    if tripcolor_kwargs is None:
        tripcolor_kwargs = {"shading": "flat"}
    if scatter_kwargs is None:
        scatter_kwargs = dict(s=2, alpha=0.3)

    u_name, v_name = auto_coords(ds)
    tris = triangles(ds, u_name, v_name)
    x = np.asarray(tris.x).ravel()
    z = np.asarray(tris.y).ravel()
    w = np.asarray(ds[var]).ravel()

    if w.size != x.size:
        raise ValueError(f"{var} length {w.size} != tris.x length {x.size}")

    m = np.isfinite(x) & np.isfinite(z) & np.isfinite(w)
    x, z, w = x[m], z[m], w[m]

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1.2], width_ratios=[1.2, 4])
    ax_left = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[0, 1])
    ax_bottom = fig.add_subplot(gs[1, 1], sharex=ax_main)

    img = ax_main.tripcolor(tris, ds[var], **tripcolor_kwargs)
    ax_main.set_xlabel(u_name)
    ax_main.set_ylabel(v_name)
    ax_main.set_title(var)

    ax_bottom.scatter(x, w, **scatter_kwargs)
    ax_bottom.set_xlabel(u_name)
    ax_bottom.set_ylabel(var)

    ax_left.scatter(w, z, **scatter_kwargs)
    ax_left.set_xlabel(var)
    ax_left.set_ylabel(v_name)
    ax_left.invert_xaxis()
    plt.setp(ax_left.get_yticklabels(), visible=False)
    ax_left.tick_params(axis="y", length=0)
    log.debug("plot_xz_slice_with_marginal_points complete")

    cbar = fig.colorbar(img, ax=[ax_main, ax_left, ax_bottom], location="right")
    cbar.set_label(var)

    return fig, (ax_main, ax_left, ax_bottom), cbar


def plot_xz_slice_tripcolor_with_marginal_quantiles_by_unique_coords(
    ds,
    *,
    var="FNEUTRAL [none]",
    qlevels=(0.16, 0.5, 0.84),
    figsize=(10, 8),
    tripcolor_kwargs=None,
):
    if tripcolor_kwargs is None:
        tripcolor_kwargs = {"shading": "flat"}

    u_name, v_name = auto_coords(ds)
    tris = triangles(ds, u_name, v_name)
    x = np.asarray(tris.x).ravel()
    z = np.asarray(tris.y).ravel()
    w = np.asarray(ds[var]).ravel()

    if w.size != x.size:
        raise ValueError(f"{var}: {w.size} values, triangulation has {x.size} points")

    m = np.isfinite(x) & np.isfinite(z) & np.isfinite(w)
    x, z, w = x[m], z[m], w[m]

    def quantiles_by_coord(coord, vals, qs):
        order = np.argsort(coord)
        c = coord[order]
        v = vals[order]

        uc, start, counts = np.unique(c, return_index=True, return_counts=True)
        out = np.full((len(qs), uc.size), np.nan)

        for i, (s, n) in enumerate(zip(start, counts)):
            seg = v[s: s + n]
            out[:, i] = np.quantile(seg, qs)

        return uc, out

    x_u, x_q = quantiles_by_coord(x, w, qlevels)  # (nq, nx)
    z_u, z_q = quantiles_by_coord(z, w, qlevels)  # (nq, nz)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1.2], width_ratios=[1.2, 4])
    ax_left = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[0, 1])
    ax_bottom = fig.add_subplot(gs[1, 1], sharex=ax_main)

    img = ax_main.tripcolor(tris, ds[var], **tripcolor_kwargs)
    ax_main.set_xlabel(u_name)
    ax_main.set_ylabel(v_name)
    ax_main.set_title(var)

    for k in range(len(qlevels)):
        ax_bottom.plot(x_u, x_q[k])
    ax_bottom.set_xlabel(u_name)
    ax_bottom.set_ylabel(var)

    for k in range(len(qlevels)):
        ax_left.plot(z_q[k], z_u)
    ax_left.set_xlabel(var)
    ax_left.set_ylabel(v_name)
    ax_left.invert_xaxis()
    plt.setp(ax_left.get_yticklabels(), visible=False)
    ax_left.tick_params(axis="y", length=0)

    cbar = fig.colorbar(img, ax=[ax_main, ax_left, ax_bottom], location="right")
    cbar.set_label(var)

    return fig, (ax_main, ax_left, ax_bottom), cbar
