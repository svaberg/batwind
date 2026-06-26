import logging
import re

import numpy as np
from matplotlib import tri

from batwind.data.field_names import DEFAULT_XYZ_NAMES

log = logging.getLogger(__name__)


def auto_coords(ds, names=None):

    if names is None:
        names = DEFAULT_XYZ_NAMES
    log.debug("auto_coords names=%s", names)

    if np.allclose(ds["X [R]"], 0):
        log.debug("auto_coords selected Y/Z plane")
        return "Y [R]", "Z [R]"
    if np.allclose(ds["Y [R]"], 0):
        log.debug("auto_coords selected X/Z plane")
        return "X [R]", "Z [R]"
    if np.allclose(ds["Z [R]"], 0):
        log.debug("auto_coords selected X/Y plane")
        return "X [R]", "Y [R]"


def triangles(ds, uname=None, vname=None):
    """ """

    if uname is None and vname is None:
        uname, vname = auto_coords(ds)

    pu = ds[uname]
    pv = ds[vname]
    raw = getattr(ds, "raw", ds)
    corners = raw.corners

    if corners.shape[1] != 4:
        raise ValueError("Can only triangulate a 2D dataset with 4 corners per element")

    triangles = np.vstack((corners[:, [0, 1, 2]], corners[:, [2, 3, 0]]))
    log.debug("triangles u=%s v=%s triangles=%d", uname, vname, triangles.shape[0])
    return tri.Triangulation(pu, pv, triangles)


def extract_index(p):
    m = re.search(r"_n(\d+)\.dat$", p.name)
    return int(m.group(1)) if m else -1


def sort_key(p):
    m = re.search(r"_n(\d+)\.dat$", p.name)
    num_str = m.group(1)
    num = int(num_str)

    # count trailing zeros
    trailing_zeros = len(num_str) - len(num_str.rstrip("0"))

    # minus for descending trailing-zero priority
    return (-trailing_zeros, num)
