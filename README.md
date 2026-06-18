<h1>
<!-- <img src="https://raw.githubusercontent.com/svaberg/batcamp/main/assets/batcamp.png" alt="batwind"> -->
<img src="assets/batwind.png" alt="batwind" height="128"> batwind
</h1>

[![Python package](https://github.com/svaberg/batwind/actions/workflows/python-package.yml/badge.svg)](https://github.com/svaberg/batwind/actions/workflows/python-package.yml)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/7b25b26a82334daf9c6224c25a40d4bb)](https://app.codacy.com/gh/svaberg/batwind/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/6236f7130d7f4b0caab5ee221430e74b)](https://app.codacy.com/gh/svaberg/batwind/dashboard?utm_source=github.com&utm_medium=referral&utm_content=svaberg/batwind&utm_campaign=Badge_Coverage)
[![DOI](https://zenodo.org/badge/634176378.svg)](https://doi.org/10.5281/zenodo.19064165)

`batwind` lets you explore three-dimensional SWMF/BATSRUS results with a simple workflow and a minimum of costly dependencies.

## Example workflow

Start by loading a file:

```python
from batwind import SmartDs  # "Smart" dataset class

# Load a BATSRUS result with three-dimensional data
sds = SmartDs.from_file(
    "3d__var_1_n00000000.plt",
    body_radius_m=6.96e8,  # Stellar radius in meters
)
print(sds)
```

If your data already lives close to a suitable `PARAM.in` file, `SmartDs` can also pick up the stellar radius from there. The stellar radius is required to compute distances, fluxes, etc., in physical units.

Data loaded, you can now ask for the quantities you want to work with. Raw quantities include:

```python
x = sds["X [R]"]  # X coordinate in stellar radii
y = sds["Y [R]"]  # Y coordinate in stellar radii
z = sds["Z [R]"]  # Z coordinate in stellar radii

rho = sds["Rho [g/cm^3]"]  # Density

bx = sds["B_x [Gauss]"]    # Magnetic field x components
by = sds["B_y [Gauss]"]    # Magnetic field y components
bz = sds["B_z [Gauss]"]    # Magnetic field z components
```

Derived quantities are computed from raw quantities using `griblet`; this is, however, transparent to the `SmartDS` user:

```python
x = sds["X [m]"]  # X coordinate in metres
b_r = sds["B_r [T]"]  # Radial magnetic field component in Tesla
u_a = sds["U_a [km/s]"]  # Azimuthal wind component in km/s
c_s = sds["c_s [km/s]"]  # Sound speed in m/s
```
These data points are at the original locations of the SWMF/BATSRUS output, which are not regular, being the leaf nodes of an Octree grid.  To obtain values at the chosen coordinates, interpolate the data onto e.g. a spherical shell, an orbital trajectory, or a regular grid. For 3D data, the default interpolation method is `OctreeInterpolator` from the `batcamp` package.  Note: This uses coordinate units of stellar radii.

```python
import numpy as np

x = np.array([1.5, 3.0, 5.0, 8.0])  # Stellar radii
y = np.zeros_like(x)
z = np.zeros_like(x)

sample_points = np.column_stack([x, y, z])

# Uses an octree-aware interpolator
sample = sds.interpolate(
    sample_points,
    fields=["Rho [kg/m^3]", "B_r [Gauss]", "U_a [km/s]"],
)
```
The `SmartDs` object caches the interpolator, which makes repeat interpolation operations very quick. 
```python
# Repeat calls to sds.interpolate uses the cached interpolator object. 
different_points = ...
sample = sds.interpolate(
    different_points, 
    fields=["Rho [kg/m^3]"])
```

## Conventions
Conventions are largely inherited from SWMF/BATSRUS.
- Customary MHD quantity names are used;
- Variable names comprise a quantity and a unit in brackets, i.e. density `Rho [kg/m^3]`;
- For many SWMF/BATSRUS output files, the default coordinates are `X [R]`, `Y [R]`, and `Z [R]`, which are cartesian coordinates in stellar-radius units;
- The coordinates `X [m]`, `Y [m]`, `Z [m]`, and `R [m]` (with unit metres) are available when the stellar radius is provided;
- Subscripts `xyz` in e.g. `U_x [m/s]`,  `U_y [m/s]`,  `U_z [m/s]` refer to cartesian vector components;
- Subscripts `rpa` in `U_r [m/s]`,  `U_p [m/s]`,  `U_a [m/s]` refer to the radial, polar, and azimuthal vector components (i.e. the components in spherical coordinates);
- The `polar [rad]` coordinate is measured from the positive `z` axis and ranges `[0, pi]` (also called colatitude);
- The azimuth `azimuth [rad]` coordinate matches the definition of `atan2(y, x)` and ranges `[-pi, pi]`.

## Examples

The `examples/` directory contains notebooks and scripts showing typical
workflows for loading data, computing quantities, and sampling 3D solutions.
