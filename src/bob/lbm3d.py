"""Single-phase D3Q19 LBM core.

BGK (or MRT, via ``rates``) collision with optional Guo forcing, periodic
streaming, full-way bounce-back. All functions are pure ``arrays -> arrays``
and compose under ``jax.jit`` / ``jax.lax.scan``.

Shapes (see ``bob.d3q19`` for the lattice convention):
  f     (19, nz, ny, nx)  distribution functions
  rho   (nz, ny, nx)      density
  u     (3, nz, ny, nx)   velocity, ordered (ux, uy, uz)
  solid (nz, ny, nx)      bool mask, True = solid
"""

import jax.numpy as jnp

from bob import d3q19, mrt

_C = d3q19.C.astype(jnp.result_type(float))  # (19, 3) velocities as floats (canonical dtype)

# (z, y, x) roll shifts for streaming: direction i moves +cx along x (axis 3),
# +cy along y (axis 2), +cz along z (axis 1). Static Python ints computed once
# at import, so no concretization happens inside a jit/scan trace.
_STREAM_SHIFTS = tuple((int(d3q19.C[i, 2]), int(d3q19.C[i, 1]), int(d3q19.C[i, 0])) for i in range(19))


def _as_field(force):
    """A (3,) uniform body force broadcast to the (3, nz, ny, nx) field shape
    (full fields pass through)."""
    return force[:, None, None, None] if jnp.ndim(force) == 1 else force


def density(f):
    """Density rho (nz, ny, nx) = sum of populations over directions."""
    return f.sum(axis=0)


def velocity(f, rho, force=None):
    """Velocity u (3, nz, ny, nx), with the Guo half-force correction when forced.

    ``force`` is None, (3,) (uniform body force ordered (Fx, Fy, Fz)), or
    (3, nz, ny, nx).
    """
    mom = jnp.einsum("ia,izyx->azyx", _C, f)
    if force is not None:
        mom = mom + _as_field(force) / 2.0
    return mom / rho


def equilibrium(rho, u):
    """Equilibrium populations feq (19, nz, ny, nx).

    feq_i = w_i rho [1 + (c_i.u)/cs2 + (c_i.u)^2/(2 cs2^2) - (u.u)/(2 cs2)]
    """
    cs2 = d3q19.CS2
    cu = jnp.einsum("ia,azyx->izyx", _C, u)
    u2 = (u**2).sum(axis=0)
    return d3q19.W[:, None, None, None] * rho[None] * (1.0 + cu / cs2 + cu**2 / (2.0 * cs2**2) - u2[None] / (2.0 * cs2))


def collide(f, omega, force=None, rates=None):
    """One BGK collision step, optionally with Guo body forcing (force density F).

    f_i <- f_i - omega (f_i - feq_i) + (1 - omega/2) w_i [ (c_i - u).F/cs2
                                            + (c_i.u)(c_i.F)/cs2^2 ]

    Pass ``rates`` (a (19,) vector from ``mrt.build_S`` / ``mrt.uniform_S``) to
    enable the MRT collision path; omit it (or pass None) for standard BGK.
    """
    rho = density(f)
    u = velocity(f, rho, force)  # includes the +F/2 correction when forced
    feq = equilibrium(rho, u)
    if rates is None:
        if force is None:
            return f - omega * (f - feq)
        cs2 = d3q19.CS2
        Fv = _as_field(force)
        cu = jnp.einsum("ia,azyx->izyx", _C, u)
        cF = jnp.einsum("ia,azyx->izyx", _C, Fv)  # broadcasts when Fv is (3,1,1,1)
        uF = (u * Fv).sum(axis=0)
        Si = (1.0 - 0.5 * omega) * d3q19.W[:, None, None, None] * ((cF - uF[None]) / cs2 + cu * cF / cs2**2)
        return f - omega * (f - feq) + Si

    # --- MRT path ---
    f_post = f - mrt.relax(f - feq, rates, d3q19.M, d3q19.Minv)
    if force is None:
        return f_post
    cs2 = d3q19.CS2
    Fv = _as_field(force)
    cu = jnp.einsum("ia,azyx->izyx", _C, u)
    cF = jnp.einsum("ia,azyx->izyx", _C, Fv)
    uF = (u * Fv).sum(axis=0)
    G = d3q19.W[:, None, None, None] * ((cF - uF[None]) / cs2 + cu * cF / cs2**2)
    return f_post + mrt.force_project(G, rates, d3q19.M, d3q19.Minv)


def stream(f):
    """Propagate f_i to the neighbor in direction c_i (periodic rolls)."""
    return jnp.stack([jnp.roll(f[i], shift=s, axis=(0, 1, 2)) for i, s in enumerate(_STREAM_SHIFTS)])


def bounce_back(f, solid):
    """Full-way bounce-back at solid nodes (no-slip): f_i(solid) <- f_opp(i)(solid)."""
    return jnp.where(solid[None], f[d3q19.OPP], f)
