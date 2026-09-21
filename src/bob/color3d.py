"""Color-gradient two-phase D3Q19 LBM.

One step acts on the total f = fR + fB:

  1. collision (MRT by default, BGK via Params.mrt=False; interface-blended omega, Guo forcing)
  2. surface tension from the color gradient — continuum-surface-force body force
     (default, Params.csf=True) or the Reis-Phillips perturbation (csf=False)
  3. Latva-Kokko recoloring — keeps the interface thin
  4. per-color streaming + bounce-back at solids (half-way link bounce-back by
     default, full-way via Params.halfway=False)

Wetting is the geometric BC (``Params.theta`` + precomputed ``wall_normals``).
Solver functions are pure jit/scan-composable JAX; ``wall_normals`` is host-side
NumPy preprocessing, and the measurement helpers live in ``bob.misc`` /
``bob.porous3d`` (host-side NumPy by design).

Conventions:
  fR, fB  (19, nz, ny, nx)  per-color populations
  rho*    (nz, ny, nx)      densities
  u       (3, nz, ny, nx)   velocity (ux, uy, uz)
  F       (3, nz, ny, nx)   color-gradient vector
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from bob import d3q19, hograd, lbm3d, mrt

__all__ = [
    "Params",
    "State",
    "collide_recolor",
    "color_field",
    "densities",
    "step",
    "stream_bounce",
    "surface_force",
    "velocity",
    "wall_normals",
    "wall_restore",
    "wall_seal",
]

_EPS = 1e-12
_C = d3q19.C.astype(jnp.result_type(float))  # (19, 3) velocities as floats (canonical dtype)
_CMAG = jnp.sqrt((_C**2).sum(1))  # (19,) |c_i|; rest direction is 0
W = d3q19.W
CS2 = d3q19.CS2

# Reis-Phillips D3Q19 perturbation coefficients (rest, axial x6, face-diagonal
# x12), Liu, Valocchi & Kang (2012). Like the D2Q9 set they sum to cs2 (mass
# conservation) and have isotropic second moment sum_i B_i c_ia c_ib = cs2 I,
# which is why the D2Q9 (9/4)(sigma/tau) prefactor carries over unchanged.
B = jnp.array([-2 / 9] + [1 / 54] * 6 + [1 / 27] * 12)

# (z, y, x) roll shifts that read the neighbor at x + c_i (negated c_i).
# Static Python ints at import time — nothing concretizes inside a trace.
_NEIGHBOR_SHIFTS = tuple((-int(d3q19.C[i, 2]), -int(d3q19.C[i, 1]), -int(d3q19.C[i, 0])) for i in range(19))


class State(NamedTuple):
    fR: jax.Array  # red  (non-wetting) populations, (19, nz, ny, nx)
    fB: jax.Array  # blue (wetting)     populations, (19, nz, ny, nx)


def _safe_inv(x, cond):
    """Guarded reciprocal: 1/x where ``cond``, exactly 0 elsewhere. The inner
    where keeps the divisor finite on the dead branch, so no NaN ever enters the
    graph (NaNs poison gradients/jit even when masked out afterwards)."""
    return jnp.where(cond, 1.0 / jnp.where(cond, x, 1.0), 0.0)


class Params(NamedTuple):
    omega: float  # BGK relaxation rate (fluid R, and B when omega2 is None)
    sigma: float  # surface-tension parameter (perturbation strength)
    beta: float   # recoloring / segregation strength, in (0, 1]
    omega2: float | None = None      # fluid-B relaxation rate; None => equal viscosity
    gravity: float | None = None     # Boussinesq buoyancy strength; None => no gravity
    mrt: bool = True                 # default backend: MRT collision (set False for plain BGK).
    #                                  MRT+Akai is validated at parity with BGK across the
    #                                  non-porous test suite and demos.
    s_bulk: float | None = None      # MRT bulk rate (defaults to d3q19.DEFAULT_S_BULK)
    s_ghost: float | None = None     # MRT ghost rate (defaults to d3q19.DEFAULT_S_GHOST)
    mrt_chi: float | None = None     # opt-in Leclaire MRT spectrum: non-shear rate = mrt_chi*omega
    #                                  (chi=4/5 stabilizes large viscosity ratios; supersedes
    #                                  s_bulk/s_ghost. None => default fixed spectrum)
    theta: float | None = None       # red contact angle in degrees — geometric wetting BC
    #                                  takes effect when ``step`` also receives the
    #                                  precomputed ``nw`` wall normals
    wetting: str = "akai"            # default wetting BC: "akai" (closed-form rotation, Akai 2018)
    #                                  or "leclaire" (secant, Leclaire 2017). Both rotate the wall
    #                                  color gradient to the prescribed angle keeping |F|; the Akai
    #                                  rotation is exact. Akai is the default as part of the validated
    #                                  MRT + CSF + half-way stack (see csf / wall_grad / halfway below).
    csf: bool = True                 # surface-tension backend. True (default): continuum-surface-
    #                                  force body force F=½σκ∇φ with the curvature wetting
    #                                  enhancements (φ/nc solid extrapolation + curvature; Sedahmed &
    #                                  Coelho, Phys. Fluids 36, 092117 (2024)). False: Reis-Phillips
    #                                  perturbation (the classic color-gradient form). See
    #                                  ``_csf_force``.
    wall_grad: str = "fluid"         # wetting wall-gradient magnitude: "fluid" (default) = native
    #                                  _fluid_gradient (raw magnitude), the physically faithful choice.
    #                                  "bulk" rescales the reoriented wall gradient to the bulk interface
    #                                  |∇rhoN| (_bulk_grad_mag) — a STRONGER angle pin. With a ΔP=0
    #                                  Zou-He pressure inlet it OVER-drives imbibition (Washburn rate
    #                                  1.3-1.7× analytic) and drives a runaway wetting film in free/open
    #                                  geometries, so it is opt-in only (the faithful combination is
    #                                  "fluid" + zou_he_inlet; see demo/washburn).
    recolor_csb_phi: bool = False    # reserved diagnostic flag (not read by the solver): in the CSF path,
    #                                  reconstruct the recoloring color densities from the Eq.34-
    #                                  extrapolated phi (rhoR/rhoB -> rho(1±rhoN_e)/2) instead of the raw
    #                                  split. That only differs at C_SB solid nodes (rhoN_e==rhoN in
    #                                  fluid), whose recolor output is discarded.
    recolor_emag: bool = True        # recoloring directional factor. True (default): the paper's Eq.17
    #                                  |e_i| factor → w_i (e_i·F)/|F| (diagonal links √2× stronger), the
    #                                  MF-LBM-coherent form (a consistent ~+0.05 on the Washburn rate,
    #                                  landing on MF-LBM's 0.80). False: Latva-Kokko / Leclaire form
    #                                  w_i (e_i·F)/(|e_i||F|). Both conserve mass (opposite links cancel).
    #                                  See _csf_force/recolor.
    halfway: bool = True             # wall BC: True (default) = half-way (link) bounce-back — fluid
    #                                  reflects its OWN per-color populations, solids store nothing the
    #                                  fluid reads → no wall color relay (no near-wall 'sparkle' leak),
    #                                  conservative; wall sits at the link midpoint (MF-LBM-coherent).
    #                                  False = classic full-way bounce-back (solids store + relay).
    grad_order: tuple = (2, 4)       # Leclaire (2014) isotropic gradient order (S, I) for the color
    #                                  gradient: (2, 4) = D3Q19 lattice weights; higher orders via bob.hograd


def color_field(rhoR, rhoB):
    """Normalized color field rhoN = (rhoR - rhoB) / (rhoR + rhoB), in [-1, 1]."""
    rho = rhoR + rhoB
    return (rhoR - rhoB) / jnp.where(rho > _EPS, rho, 1.0)


def _reorient(F, nw, theta):
    """Geometric wetting BC (Leclaire et al., PRE 95, 033306 (2017), Sec. II.E):
    rotate F so its angle with the wall normal ``nw`` equals ``theta`` (degrees),
    keeping |F|. Secant root-find of f(v) = v.nw - |v| cos(theta) carried to the
    paper's n=4 (Eq. 29-31) from v0 = nc, v1 = nc - (nc + nw)/2; the result lies
    in span{nc, nw} and is normalized before rescaling by |F|. Dimension-agnostic
    (axis-0 reductions).

    Active only where an interface exists (|F| > eps) AND ``nw`` is set (X_W
    sites) — dry walls exert no standing force, unlike the fictitious-color BC.
    """
    cos_t = jnp.cos(jnp.deg2rad(theta))
    Fmag = jnp.sqrt((F**2).sum(0))
    nwmag2 = (nw**2).sum(0)
    active = (Fmag > _EPS) & (nwmag2 > 0.5)
    inv_F = _safe_inv(Fmag, active)
    nc = F * inv_F[None]

    def f_of(v):
        vmag = jnp.sqrt((v**2).sum(0))
        return (v * nw).sum(0) - vmag * cos_t

    def _sec(va, vb, fa, fb):  # one secant update; on a stalled (≈0) denominator
        d = fb - fa            # the iteration has converged, so keep the latest vb
        ok = jnp.abs(d) > _EPS
        inv = _safe_inv(d, ok)
        return jnp.where(ok[None], (va * fb[None] - vb * fa[None]) * inv[None], vb)

    # Leclaire 2017 Eq. (29)-(31): secant recurrence to n=4 (three updates). A
    # truncation at n=2 leaves the imposed direction only roughly at θ (up to
    # ~12° off for far starts); n=4 converges it tightly, matching the closed-form
    # Akai rotation. Direction only — |F| is untouched, so no change to the
    # surface-tension strength / no risk of a wall spreading film.
    v0 = nc
    v1 = nc - 0.5 * (nc + nw)
    f0, f1 = f_of(v0), f_of(v1)
    v2 = _sec(v0, v1, f0, f1)
    v3 = _sec(v1, v2, f1, f_of(v2))
    v4 = _sec(v2, v3, f_of(v2), f_of(v3))
    v4mag = jnp.sqrt((v4**2).sum(0))
    safe = v4mag > _EPS
    v4 = v4 * _safe_inv(v4mag, safe)[None]
    return jnp.where(active[None], Fmag[None] * v4, F)


def _reorient_akai(F, nw, theta):
    """Closed-form wetting BC (Akai et al., Adv. Water Resour. 116 (2018),
    Sec. 2.2.2; Xu et al. 2017 / OpenFOAM rotation). Same contract as
    ``_reorient``: return a vector with |F| unchanged whose angle with the wall
    normal ``nw`` equals ``theta`` (degrees), lying in span{nc, nw}.

    Equivalent to the paper's n+/- construction with the red-ward interface
    normal: n_s = nw (toward solid), n* = nc = F/|F| (toward red),
    theta' = arccos(n_s.n*).
        n+/- = (cos(+/-theta) - sin(+/-theta)*cos(theta')/sin(theta'))*n_s
               + (sin(+/-theta)/sin(theta'))*n*
    pick the n+/- nearer (Euclidean) to nc, scale by |F|. The +/- branch resolves
    the contact-line side; with n*=nc the paper's 'nearer to n*' is 'nearer to nc'.

    Active only where |F| > eps AND nw is set (C_FB). Where sin(theta')->0
    (interface parallel to wall, nc parallel to +/-nw) the formula is singular and
    there is no defined contact line, so F is left unchanged. Dimension-agnostic
    (axis-0 reductions)."""
    Fmag = jnp.sqrt((F**2).sum(0))
    nwmag2 = (nw**2).sum(0)
    inv_F = _safe_inv(Fmag, Fmag > _EPS)
    nc = F * inv_F[None]

    cos_tp = (nc * nw).sum(0)                         # cos(theta') = n_s.n*
    sin_tp = jnp.sqrt(jnp.clip(1.0 - cos_tp**2, 0.0, 1.0))
    active = (Fmag > _EPS) & (nwmag2 > 0.5) & (sin_tp > _EPS)
    inv_stp = _safe_inv(sin_tp, active)

    def branch(sign):  # sign = +1 or -1 -> +/- theta
        ang = sign * jnp.deg2rad(theta)
        a = jnp.cos(ang) - jnp.sin(ang) * cos_tp * inv_stp
        b = jnp.sin(ang) * inv_stp
        return a[None] * nw + b[None] * nc

    mp, mm = branch(1.0), branch(-1.0)
    dp = ((mp - nc) ** 2).sum(0)                       # pick branch nearer to nc (== n*)
    dm = ((mm - nc) ** 2).sum(0)
    m = jnp.where((dp <= dm)[None], mp, mm)
    return jnp.where(active[None], Fmag[None] * m, F)


def _omega_field(rhoN, params):
    """Per-cell omega from a harmonic blend of kinematic viscosity over the red
    fraction psi=(1+rhoN)/2. Returns the scalar omega when omega2 is None."""
    if params.omega2 is None:
        return params.omega
    psi = jnp.clip((1.0 + rhoN) / 2.0, 0.0, 1.0)
    nu_r = CS2 * (1.0 / params.omega - 0.5)
    nu_b = CS2 * (1.0 / params.omega2 - 0.5)
    nu = 1.0 / (psi / nu_r + (1.0 - psi) / nu_b)      # weighted harmonic mean
    return 1.0 / (nu / CS2 + 0.5)


def densities(state):
    """Per-color densities (rhoR, rhoB), each (nz, ny, nx)."""
    return state.fR.sum(0), state.fB.sum(0)


def velocity(state):
    """Total velocity u = (sum_i (fR_i + fB_i) c_i) / rho, (3, nz, ny, nx)."""
    f = state.fR + state.fB
    return lbm3d.velocity(f, f.sum(0))


def _gather_neighbors(field):
    """Stack of ``field`` evaluated at x + c_i for each direction, (19, nz, ny, nx)."""
    return jnp.stack([jnp.roll(field, shift=s, axis=(0, 1, 2)) for s in _NEIGHBOR_SHIFTS])


def color_gradient(rhoN, order=(2, 4)):
    """Isotropic color gradient F_a = (1/cs2) sum_i w_i rhoN(x+c_i) c_ia, (3, nz, ny, nx).

    ``order`` selects the Leclaire (2014) isotropic stencil ``(S, I)``; default
    ``(2, 4)`` is the D3Q19 lattice-weight gradient (bit-identical original einsum),
    higher orders route through ``bob.hograd``."""
    if order == (2, 4):
        neigh = _gather_neighbors(rhoN)
        return jnp.einsum("i,izyx,ia->azyx", W, neigh, _C) / CS2
    return hograd.gradient(rhoN, order=order)


def _fluid_gradient(rhoN, solid, order=(2, 4)):
    """A priori color gradient from bulk-fluid information only (Leclaire 2017):
    solid-neighbor values are replaced by the center cell's rhoN, so the wall
    contributes nothing to the stencil. Higher ``order`` uses ``bob.hograd`` (wide
    bulk stencil + (2,4) fallback within reach of a solid)."""
    if order != (2, 4):
        return hograd.gradient(rhoN, order=order, solid=solid)
    neigh = _gather_neighbors(rhoN)
    if solid is not None:
        neigh = jnp.where(_gather_neighbors(solid), rhoN[None], neigh)
    return jnp.einsum("i,izyx,ia->azyx", W, neigh, _C) / CS2


def _bulk_grad_mag(rhoN, solid):
    """Interior (undistorted) interface |∇rhoN| — the max over fluid neighbors of the
    fluid-only gradient magnitude; dry-safe. Used by the ``wall_grad="bulk"`` wetting path."""
    mag = jnp.sqrt((_fluid_gradient(rhoN, solid) ** 2).sum(0))
    if solid is None:
        return mag
    fmask = _gather_neighbors((~solid).astype(rhoN.dtype)) > 0.5
    return jnp.max(jnp.where(fmask, _gather_neighbors(mag), 0.0), axis=0)


# ─── CSF curvature wetting model (Sedahmed & Coelho, Phys. Fluids 36, 092117 (2024),
# DOI 10.1063/5.0228835).
# Selected by Params.csf (the default). ────────────────────────────────────────


def _extrapolate_phi_csb(rhoN, solid):
    """Eq. 34 (3D): solid-boundary color φ = lattice-weighted average of its fluid
    neighbors, so the gradient stencil at near-wall fluid cells reads a smooth
    continuation of the fluid field instead of the stored wall value."""
    fluid = (~solid).astype(rhoN.dtype)
    fl_nb = _gather_neighbors(fluid)
    den = (W[:, None, None, None] * fl_nb).sum(0)
    num = (W[:, None, None, None] * fl_nb * _gather_neighbors(rhoN)).sum(0)
    val = jnp.where(den > _EPS, num / jnp.where(den > _EPS, den, 1.0), rhoN)
    return jnp.where(solid & (den > _EPS), val, rhoN)


def _extrapolate_normal_csb(nc, solid):
    """Eq. 35 (3D, zero-interfacial-force): solid-boundary normal = lattice-weighted
    average of its fluid-neighbor normals."""
    fluid = (~solid).astype(nc.dtype)
    fl_nb = _gather_neighbors(fluid)
    den = (W[:, None, None, None] * fl_nb).sum(0)
    csb = solid & (den > _EPS)
    out = [
        jnp.where(
            csb,
            jnp.where(den > _EPS, (W[:, None, None, None] * fl_nb * _gather_neighbors(nc[a])).sum(0)
                      / jnp.where(den > _EPS, den, 1.0), nc[a]),
            nc[a],
        )
        for a in range(nc.shape[0])
    ]
    return jnp.stack(out)


def _curvature(n, order=(2, 4)):
    """Eq. 12 (full 3D): κ = Σ_a (n_a²−1)∂_a n_a + Σ_{a<b}(∂_b n_a + ∂_a n_b) n_a n_b,
    the surface divergence of the unit interface normal n (3, nz, ny, nx). Partials
    use the isotropic ``color_gradient`` stencil (``order``)."""
    nx, ny, nz = n[0], n[1], n[2]
    gx = color_gradient(nx, order=order)               # (∂ₓnx, ∂_y nx, ∂_z nx)
    gy = color_gradient(ny, order=order)               # (∂ₓny, ∂_y ny, ∂_z ny)
    gz = color_gradient(nz, order=order)               # (∂ₓnz, ∂_y nz, ∂_z nz)
    return (
        (nx**2 - 1.0) * gx[0] + (ny**2 - 1.0) * gy[1] + (nz**2 - 1.0) * gz[2]
        + (gx[1] + gy[0]) * nx * ny
        + (gx[2] + gz[0]) * nx * nz
        + (gz[1] + gy[2]) * nz * ny
    )


def _csf_force(rhoN, params, solid, nw):
    """CSF surface tension: extrapolate φ into the wall (Eq. 34), take the color gradient
    C, reorient it at the wall (geometric wetting BC), extrapolate the unit normal (Eq. 35),
    and form κ (Eq. 12). Returns (CSF body force ½σκC, reoriented gradient C)."""
    order = params.grad_order
    rhoN_e = _extrapolate_phi_csb(rhoN, solid) if solid is not None else rhoN
    C = color_gradient(rhoN_e, order=order)
    if nw is not None and params.theta is not None:
        if params.wetting == "akai":
            C = _reorient_akai(C, nw, params.theta)
        elif params.wetting == "leclaire":
            C = _reorient(C, nw, params.theta)
        else:
            raise ValueError(f"unknown Params.wetting={params.wetting!r} (expected 'leclaire' or 'akai')")
    Cmag = jnp.sqrt((C**2).sum(0))
    safe = Cmag > _EPS
    nc = C * _safe_inv(Cmag, safe)[None]
    if solid is not None:
        nc = _extrapolate_normal_csb(nc, solid)
    kappa = _curvature(nc, order=order)
    if params.wall_grad == "bulk" and solid is not None:
        # opt-in bulk-magnitude boost — direction/κ kept.
        unit = C * _safe_inv(Cmag, safe)[None]
        C = jnp.where(safe[None], unit * _bulk_grad_mag(rhoN, solid)[None], C)
    F = 0.5 * params.sigma * (kappa * safe)[None] * C
    return F, C


# 27-point smoothing weights for the wall-normal preprocessing: D3Q27 weights
# indexed by |c|^2 (Leclaire 2017, Eqs. 34-38).
_SMOOTH_W3 = {0: 8.0 / 27.0, 1: 2.0 / 27.0, 2: 1.0 / 54.0, 3: 1.0 / 216.0}


def wall_normals(solid, iters=3):
    """Preprocess a binary solid mask into unit wall normals at X_W (fluid sites
    with at least one solid lattice neighbor), pointing INTO the solid; zero
    elsewhere. Smooth the mask with the 27-point D3Q27-weight kernel ``iters``
    times, take the isotropic stencil gradient, normalize (Leclaire 2017).
    Host-side NumPy preprocessing; returns a (3, nz, ny, nx) jnp array ordered
    (ux, uy, uz) like velocities. This is the geometry pipeline for rock scans:
    compute once per structure and store/reuse."""
    solid = np.asarray(solid, dtype=bool)
    g = solid.astype(float)
    offsets = [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    for _ in range(iters):
        g = sum(
            _SMOOTH_W3[dz * dz + dy * dy + dx * dx] * np.roll(g, shift=(dz, dy, dx), axis=(0, 1, 2))
            for dz, dy, dx in offsets
        )
    C = np.asarray(d3q19.C)
    Wn = np.asarray(d3q19.W)
    grad = np.zeros((3,) + g.shape)
    neigh_solid = np.zeros_like(solid)
    for i in range(19):
        cx, cy, cz = int(C[i, 0]), int(C[i, 1]), int(C[i, 2])
        gi = np.roll(g, shift=(-cz, -cy, -cx), axis=(0, 1, 2))  # g(x + c_i)
        grad[0] += Wn[i] * gi * cx
        grad[1] += Wn[i] * gi * cy
        grad[2] += Wn[i] * gi * cz
        if cx or cy or cz:
            neigh_solid |= np.roll(solid, shift=(-cz, -cy, -cx), axis=(0, 1, 2))
    grad /= float(d3q19.CS2)
    mag = np.sqrt((grad**2).sum(0))
    xw = ~solid & neigh_solid & (mag > 1e-12)
    nw = np.where(xw[None], grad / np.where(mag > 1e-12, mag, 1.0)[None], 0.0)
    return jnp.asarray(nw)


def perturb(f, F, params, omega=None):
    """Reis-Phillips surface-tension perturbation on the total f.

    Δf_i = (9/4) omega_eff sigma |F| ( w_i (c_i.F)^2/|F|^2 - B_i ) — Leclaire et al.
    (2017) Eq. (18), A = (9/4) omega_eff sigma. Conserves mass and vanishes where
    the color gradient does. ``omega`` is the effective (interface-blended)
    relaxation rate from ``_omega_field`` (scalar or (nz, ny, nx) field), defaulting
    to ``params.omega`` (exact at equal viscosity); using omega_eff keeps the
    realized surface tension independent of the viscosity ratio. The Laplace test
    checks the calibration in 3D via dp = 2 sigma / R.
    """
    omega = params.omega if omega is None else omega
    Fmag = jnp.sqrt((F**2).sum(0))
    cF = jnp.einsum("ia,azyx->izyx", _C, F)
    safe = Fmag > _EPS
    inv_F2 = _safe_inv(Fmag**2, safe)
    term = W[:, None, None, None] * cF**2 * inv_F2[None] - B[:, None, None, None]
    pref = (9.0 / 4.0) * omega * params.sigma * Fmag  # A = (9/4) omega_eff sigma
    return f + (pref * safe)[None] * term


def recolor(f, rhoR, rhoB, F, beta, emag=False):
    """Latva-Kokko recoloring. ``emag`` (the solver passes ``Params.recolor_emag``,
    default True) keeps the paper Eq.17 |e_i| factor: directional factor w_i (e_i·F)/|F|
    instead of the Latva-Kokko w_i (e_i·F)/(|e_i||F|). Both conserve color (opposite
    links cancel)."""
    rho = rhoR + rhoB
    inv_rho = _safe_inv(rho, rho > _EPS)
    Fmag = jnp.sqrt((F**2).sum(0))
    cF = jnp.einsum("ia,azyx->izyx", _C, F)
    denom = Fmag[None] if emag else _CMAG[:, None, None, None] * Fmag[None]  # rest -> 0
    direction = jnp.where(denom > _EPS, cF / jnp.where(denom > _EPS, denom, 1.0), 0.0)
    extra = (beta * rhoR * rhoB * inv_rho)[None] * W[:, None, None, None] * direction
    fR = (rhoR * inv_rho)[None] * f + extra
    fB = (rhoB * inv_rho)[None] * f - extra
    return fR, fB


def _velocity(f, rho, force=None):
    """Total velocity with the Guo half-force correction; force None, (3,), or full."""
    mom = jnp.einsum("ia,izyx->azyx", _C, f)
    if force is not None:
        mom = mom + lbm3d._as_field(force) / 2.0
    return mom / rho


def _collide(f, rho, u, omega, force, params):
    """BGK unless params.mrt; omega scalar or (nz,ny,nx) field; force None/(3,)/field."""
    feq = lbm3d.equilibrium(rho, u)
    if not params.mrt:
        om = omega if jnp.ndim(omega) == 0 else omega[None]
        f = f - om * (f - feq)
        if force is None:
            return f
        Fv = lbm3d._as_field(force)
        cu = jnp.einsum("ia,azyx->izyx", _C, u)
        cF = jnp.einsum("ia,azyx->izyx", _C, Fv)
        uF = (u * Fv).sum(axis=0)
        coef = 1.0 - 0.5 * omega
        coef = coef if jnp.ndim(omega) == 0 else coef[None]
        return f + coef * W[:, None, None, None] * ((cF - uF[None]) / CS2 + cu * cF / CS2**2)

    # --- MRT path ---
    S = mrt.build_S(d3q19, omega, s_bulk=params.s_bulk, s_ghost=params.s_ghost, chi=params.mrt_chi)
    f_post = f - mrt.relax(f - feq, S, d3q19.M, d3q19.Minv)
    if force is None:
        return f_post
    Fv = lbm3d._as_field(force)
    cu = jnp.einsum("ia,azyx->izyx", _C, u)
    cF = jnp.einsum("ia,azyx->izyx", _C, Fv)
    uF = (u * Fv).sum(axis=0)
    G = W[:, None, None, None] * ((cF - uF[None]) / CS2 + cu * cF / CS2**2)
    return f_post + mrt.force_project(G, S, d3q19.M, d3q19.Minv)


def _halfway_bounce_stream(f_post, solid):
    """Radius-1 part of the half-way (link) bounce-back: stream + reflect at
    fluid nodes whose i-population came from a solid neighbour. The pointwise
    solid-node seal (keep f_prev) lives in ``wall_seal`` so the staged halo
    path can apply it after trimming ghost planes."""
    fluid = ~solid
    from_solid = lbm3d.stream(jnp.broadcast_to(solid.astype(f_post.dtype), f_post.shape)) > 0.5
    f_s = lbm3d.stream(f_post)
    return jnp.where(from_solid & fluid[None], f_post[d3q19.OPP], f_s)


def _halfway_bounce(f_post, f_prev, solid):
    """Half-way (link) bounce-back per color.
    Stream the post-collision f, then at fluid nodes whose i-population would have
    streamed in from a solid neighbour, use the node's OWN reflected (opposite)
    post-collision population. Solids store nothing the fluid reads (kept at f_prev
    for finiteness) → no wall color relayed into the fluid, each color conserved."""
    return jnp.where(solid[None], f_prev, _halfway_bounce_stream(f_post, solid))


def surface_force(rhoN, params, solid=None, nw=None):
    """Stage 1 of ``step`` — the neighbor-deep color-gradient computation (depth
    ``multigpu.required_phi_halo``); everything downstream until streaming is
    pointwise. Returns ``(F_surf, G)``: ``F_surf`` the CSF body force (``None``
    when ``params.csf`` is False), ``G`` the gradient perturb/recolor use (the
    reoriented C under CSF, the perturbation F otherwise)."""
    if params.csf:
        return _csf_force(rhoN, params, solid, nw)
    if nw is not None and params.theta is not None:  # geometric wetting BC
        if params.wetting == "akai":
            F = _reorient_akai(_fluid_gradient(rhoN, solid, order=params.grad_order), nw, params.theta)
        elif params.wetting == "leclaire":
            F = _reorient(_fluid_gradient(rhoN, solid, order=params.grad_order), nw, params.theta)
        else:
            raise ValueError(f"unknown Params.wetting={params.wetting!r} (expected 'leclaire' or 'akai')")
        if params.wall_grad == "bulk":               # hold the meniscus angle under flow
            Fmag = jnp.sqrt((F**2).sum(0))
            active = Fmag > _EPS
            unit = F / jnp.where(active, Fmag, 1.0)[None]
            F = jnp.where(active[None], unit * _bulk_grad_mag(rhoN, solid)[None], F)
    else:
        F = color_gradient(rhoN, order=params.grad_order)
    return None, F


def collide_recolor(state, rhoR, rhoB, params, F_surf, G, force=None):
    """Stage 2 of ``step`` — pointwise: Boussinesq gravity, blended omega,
    velocity, collision (MRT/BGK + Guo forcing), Reis-Phillips perturbation
    (csf=False), Latva-Kokko recoloring. Returns pre-stream ``(fR, fB)``."""
    f = state.fR + state.fB
    rho = f.sum(0)
    rhoN = color_field(rhoR, rhoB)

    if params.gravity is not None:  # Boussinesq buoyancy: heavy down, light up
        zero = jnp.zeros_like(rhoN)
        buoy = jnp.stack([zero, zero, -params.gravity * rhoN])
        force = buoy if force is None else lbm3d._as_field(force) + buoy

    omega = _omega_field(rhoN, params)

    if params.csf:
        # CSF backend (3D): surface tension is the body force ½σκ∇φ with the
        # curvature wetting enhancements, via Guo forcing — no perturbation.
        total = F_surf if force is None else F_surf + lbm3d._as_field(force)
        u = _velocity(f, rho, total)
        f = _collide(f, rho, u, omega, total, params)
        return recolor(f, rhoR, rhoB, G, params.beta, emag=params.recolor_emag)
    u = _velocity(f, rho, force)
    f = _collide(f, rho, u, omega, force, params)
    f = perturb(f, G, params, omega=omega)
    return recolor(f, rhoR, rhoB, G, params.beta, emag=params.recolor_emag)


def wall_restore(fR, fB, prev, params, solid):
    """Pointwise inert-wall restore (full-way path only): put the pre-step wall
    populations back at solid nodes before streaming — stops recoloring from
    pumping color into the rest population at solid nodes (a one-way mass leak,
    ~21% of a sessile droplet over 4000 steps in 3D); wetting is unaffected (it
    acts on the fluid cells next to the wall). Identity under the half-way BC (its
    seal is ``wall_seal``) or without solids."""
    if solid is None or params.halfway:
        return fR, fB
    return jnp.where(solid[None], prev.fR, fR), jnp.where(solid[None], prev.fB, fB)


def stream_bounce(fR, fB, params, solid=None):
    """Stage 3 of ``step``, radius-1 part: per-color streaming + wall reflection
    (half-way link bounce — fluid reflects its OWN per-color populations, solids
    relay nothing into the fluid — or stream + full-way bounce-back)."""
    if solid is not None and params.halfway:
        return _halfway_bounce_stream(fR, solid), _halfway_bounce_stream(fB, solid)
    fR, fB = lbm3d.stream(fR), lbm3d.stream(fB)
    if solid is not None:
        fR, fB = lbm3d.bounce_back(fR, solid), lbm3d.bounce_back(fB, solid)
    return fR, fB


def wall_seal(fR, fB, prev, params, solid):
    """Pointwise close of the half-way BC: solid nodes keep their pre-step
    populations (stored for finiteness, never read by the fluid). Identity
    otherwise."""
    if solid is None or not params.halfway:
        return State(fR, fB)
    return State(jnp.where(solid[None], prev.fR, fR), jnp.where(solid[None], prev.fB, fB))


def step(state, params, solid=None, force=None, nw=None):
    """One 3D color-gradient step: color gradient / surface force -> collide (MRT+Guo by
    default, BGK via mrt=False; blended omega; the CSF force enters through the Guo
    forcing) -> perturb (csf=False only) -> recolor -> stream -> bounce-back.
    Boussinesq gravity (if set) acts along -z. Wetting: ``params.theta`` +
    precomputed ``nw`` selects the geometric BC (Leclaire 2017 or Akai 2018);
    without them walls are neutral. Composed from the stage functions
    ``surface_force`` (neighbor-deep) / ``collide_recolor`` (pointwise) /
    ``wall_restore`` / ``stream_bounce`` (radius 1) / ``wall_seal`` so
    ``multigpu.staged_halo_step`` can interleave thin halo exchanges.
    """
    rhoR, rhoB = densities(state)
    rhoN = color_field(rhoR, rhoB)
    F_surf, G = surface_force(rhoN, params, solid, nw)
    fR, fB = collide_recolor(state, rhoR, rhoB, params, F_surf, G, force)
    fR, fB = wall_restore(fR, fB, state, params, solid)
    fR, fB = stream_bounce(fR, fB, params, solid)
    return wall_seal(fR, fB, state, params, solid)
