from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.io import FortranFile

from batread import Dataset


def _record_endian(path: str | Path) -> str:
    raw = Path(path).read_bytes()[:4]
    if len(raw) != 4:
        raise ValueError(f"Empty or truncated UA/GITM file: {path}")
    big = int(np.frombuffer(raw, dtype=">i4")[0])
    if 0 < big < 10000:
        return ">"
    little = int(np.frombuffer(raw, dtype="<i4")[0])
    if 0 < little < 10000:
        return "<"
    raise ValueError(f"Could not determine Fortran record endianness for {path}")


def _decode_name(record: np.ndarray) -> str:
    return bytes(np.asarray(record, dtype=np.uint8)).decode("ascii", errors="replace").strip()


def read_ua_gitm_bin(path: str | Path) -> Dataset:
    """
    Read one UA/MGITM `3DALL*.bin` file into an in-memory `Dataset`.

    The binary file stores a structured `(nLon, nLat, nAlt)` grid, with each
    variable written as one Fortran unformatted record in column-major order.
    """
    file_path = Path(path)
    endian = _record_endian(file_path)
    i4 = np.dtype(f"{endian}i4")
    f8 = np.dtype(f"{endian}f8")
    u1 = np.dtype(np.uint8)

    with FortranFile(file_path, "r", header_dtype=np.dtype(f"{endian}u4")) as fh:
        version = float(fh.read_record(f8)[0])
        n_lon, n_lat, n_alt = (int(value) for value in fh.read_record(i4))
        n_vars = int(fh.read_record(i4)[0])
        variables = [_decode_name(fh.read_record(u1)) for _ in range(n_vars)]
        yy, mm, dd, hh, minute, ss, ms = (int(value) for value in fh.read_record(i4))
        time = datetime(yy, mm, dd, hh, minute, ss, ms * 1000)

        points = np.empty((n_lon, n_lat, n_alt, n_vars), dtype=float)
        for i_var in range(n_vars):
            points[..., i_var] = fh.read_record(f8).reshape((n_lon, n_lat, n_alt), order="F")

    aux = {
        "UA_TIME": time,
        "UA_VERSION": version,
        "UA_ENDIAN": "big" if endian == ">" else "little",
        "UA_NLON": n_lon,
        "UA_NLAT": n_lat,
        "UA_NALT": n_alt,
    }
    corners = np.empty((0, 0), dtype=int)
    return Dataset(points, corners, aux=aux, title=file_path.name, variables=variables, zone="ua")


__all__ = ["read_ua_gitm_bin"]
