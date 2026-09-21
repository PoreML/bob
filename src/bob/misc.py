"""Validation helpers: initial states + host-side shape metrics.

Decoupled from ``bob.color3d`` (which owns the physics step). ``init_sphere`` /
``init_box`` build red-in-blue ``State``s for relaxation tests; ``sphericity``,
``pressure_jump`` (Laplace law) and ``contact_angle`` (spherical-cap fit on the
bottom z-wall) are NumPy metrics, host-side by design — pure-JAX solver
functions stay jit/scan-composable, measurements do not need to.
"""

import jax.numpy as jnp
import numpy as np

from bob import d3q19
from bob.color3d import State, densities

__all__ = [
    "contact_angle",
    "init_box",
    "init_sphere",
    "pressure_jump",
    "sphericity",
]

W = d3q19.W
CS2 = d3q19.CS2


def init_sphere(nz, ny, nx, R, cz=None, cy=None, cx=None):
    """A red ball of radius R (unit density) in a blue background."""
    cz = (nz - 1) / 2.0 if cz is None else cz
    cy = (ny - 1) / 2.0 if cy is None else cy
    cx = (nx - 1) / 2.0 if cx is None else cx
    zz, yy, xx = jnp.meshgrid(jnp.arange(nz), jnp.arange(ny), jnp.arange(nx), indexing="ij")
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2 <= R**2
    rhoR = jnp.where(inside, 1.0, 0.0)
    return State(W[:, None, None, None] * rhoR[None], W[:, None, None, None] * (1.0 - rhoR)[None])


def init_box(nz, ny, nx, d, h, w):
    """A centered red box (d deep in z, h tall in y, w wide in x) in blue."""
    zz, yy, xx = jnp.meshgrid(jnp.arange(nz), jnp.arange(ny), jnp.arange(nx), indexing="ij")
    inside = (
        (jnp.abs(zz - (nz - 1) / 2.0) <= d / 2.0)
        & (jnp.abs(yy - (ny - 1) / 2.0) <= h / 2.0)
        & (jnp.abs(xx - (nx - 1) / 2.0) <= w / 2.0)
    )
    rhoR = jnp.where(inside, 1.0, 0.0)
    return State(W[:, None, None, None] * rhoR[None], W[:, None, None, None] * (1.0 - rhoR)[None])


def sphericity(mass):
    """Inertia-tensor aspect ratio sqrt(lambda_max/lambda_min) of a 3D mass field.

    ~1 for a ball; grows as the shape elongates. NumPy metric.
    """
    mass = np.asarray(mass, dtype=float)
    total = mass.sum()
    coords = np.stack(np.indices(mass.shape)).reshape(3, -1)
    m = mass.reshape(-1)
    centre = (coords * m).sum(axis=1) / total
    d = coords - centre[:, None]
    inertia = np.einsum("ak,bk,k->ab", d, d, m) / total
    ev = np.linalg.eigvalsh(inertia)
    return float(np.sqrt(ev.max() / max(ev.min(), 1e-30)))


def pressure_jump(state, R, margin=5):
    """Laplace pressure jump dP = cs2*(rho_core - rho_far) for a centered droplet.

    Core: cells within R/2 of the centre; far field: cells beyond R+margin.
    Host-side NumPy metric.
    """
    rho = np.asarray((state.fR + state.fB).sum(0))
    nz, ny, nx = rho.shape
    zs, ys, xs = np.indices(rho.shape)
    r = np.sqrt((xs - (nx - 1) / 2.0) ** 2 + (ys - (ny - 1) / 2.0) ** 2 + (zs - (nz - 1) / 2.0) ** 2)
    return float(CS2 * (rho[r < R / 2.0].mean() - rho[r > R + margin].mean()))


def contact_angle(state, wall_plane=0):
    """Contact angle (degrees) of a sessile red droplet on the bottom z-wall, via
    the spherical-cap relation theta = 2*atan(H/b): b = base radius from the red
    area in the first fluid plane, H = cap height. NumPy metric.

    The base radius is the area-equivalent radius b = sqrt(area/pi) of the red
    disk in the first fluid plane (one layer above the wall).
    """
    rhoR, rhoB = densities(state)
    red = np.asarray(rhoR > rhoB)
    base = red[wall_plane + 1]
    if not base.any():
        return 0.0
    b = float(np.sqrt(base.sum() / np.pi))
    planes = np.where(red[wall_plane + 1 :].any(axis=(1, 2)))[0]
    H = float(planes.max() - planes.min() + 1)
    return float(np.degrees(2.0 * np.arctan(H / b)))
