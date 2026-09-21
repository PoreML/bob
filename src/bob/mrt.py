"""Dimension-agnostic MRT (multiple-relaxation-time) collision machinery.

Collision is  f <- f - Minv @ (S * (M @ (f - feq)))  + forcing, where M maps
populations to moments, S is the per-moment relaxation-rate vector, and
M (f - feq) = m - m_eq automatically (so we reuse the existing `equilibrium`,
never deriving moment-space equilibria). Setting every entry of S to `omega`
reproduces BGK exactly: M^{-1} (omega I) M = omega I.

All functions are pure arrays->arrays and compose under jit/scan. The matrices
live in `bob.d3q19`; this module only does the linear algebra.
"""

import jax.numpy as jnp


def moment_matrix(C, polys):
    """Build the (Q, Q) moment matrix M by evaluating each moment polynomial on
    the lattice velocities. `C` is (Q, D) int velocities; `polys` is a list of Q
    callables p(*components) -> scalar. Row q of M is poly q evaluated on every
    velocity. Evaluating on the actual `C` makes M robust to velocity ordering.
    """
    Cn = [tuple(float(v) for v in row) for row in C.tolist()]
    rows = [[float(p(*cv)) for cv in Cn] for p in polys]
    return jnp.array(rows)  # python floats -> canonical float dtype (f64 under x64, f32 otherwise)


def _expand(S, ndim):
    """Reshape a rate vector S of shape (Q,) to broadcast against g of `ndim`
    axes (Q, *spatial). A field S already shaped (Q, *spatial) passes through."""
    if S.ndim == 1:
        return S.reshape(S.shape + (1,) * (ndim - 1))
    return S


def relax(g, S, M, Minv):
    """MRT relaxation of the deviation g = f - feq: returns Minv @ (S * (M @ g)),
    the amount to subtract from f. `g` is (Q, *spatial); S is (Q,) or (Q, *spatial)."""
    sub = "qp,p...->q..."
    m = jnp.einsum(sub, M, g)
    m = _expand(S, g.ndim) * m
    return jnp.einsum(sub, Minv, m)


def force_project(G, S, M, Minv):
    """Project the raw Guo source G into moment space with the (I - S/2) factor:
    returns Minv @ ((1 - S/2) * (M @ G)). With all rates = omega this collapses
    to the scalar (1 - omega/2) G of the BGK Guo term."""
    sub = "qp,p...->q..."
    m = jnp.einsum(sub, M, G)
    m = (1.0 - 0.5 * _expand(S, G.ndim)) * m
    return jnp.einsum(sub, Minv, m)


def build_S(lattice, omega, s_bulk=None, s_ghost=None, chi=None):
    """Assemble the rate vector/field S for a lattice module.

    Conserved slots -> 0 (inert: m - m_eq = 0 there). Shear slots -> `omega`
    (sets viscosity nu = cs2 (1/omega - 1/2)). Bulk slot -> `s_bulk`. Ghost
    slots -> `s_ghost`. Defaults come from the lattice module. `omega` may be a
    scalar or an (*spatial) field; a field returns S shaped (Q, *spatial).

    Note: in the *bulk* the recovered viscosity is independent of the ghost
    rates; bounded (bounce-back) flows additionally show a small wall-slip /
    TRT-magic-parameter dependence on the energy-flux rate — by design we leave
    the ghost rates free (set by `s_ghost`) rather than pinning them to omega.

    `chi` (opt-in) overrides the bulk/ghost spectrum with the Leclaire (2017)
    form: every non-shear moment relaxes at `chi*omega`, tied to the local shear
    rate (chi=4/5 damps under-resolved modes -> stabler at large viscosity ratio;
    chi=1 -> SRT). It supersedes s_bulk/s_ghost and the per-moment rates;
    `chi=None` (default) uses the fixed spectrum described below. Shear stays
    `omega`, so viscosity is preserved.
    """
    Q = lattice.M.shape[0]
    if chi is not None:
        ns = chi * omega  # non-shear (bulk + ghost) rate, tied to the local shear rate
        bk, gh, sh = jnp.array(lattice.BULK_IDX), jnp.array(lattice.GHOST_IDX), jnp.array(lattice.SHEAR_IDX)
        if jnp.ndim(omega) == 0:
            return jnp.zeros(Q).at[bk].set(ns).at[gh].set(ns).at[sh].set(omega)
        base = jnp.zeros((Q,) + omega.shape).at[bk].set(ns[None]).at[gh].set(ns[None])
        return base.at[sh].set(omega[None])
    # Per-moment fixed rates (e.g. d3q19.GHOST_RATES = MF-LBM default spectrum) are used
    # when the caller leaves s_bulk/s_ghost at their defaults; an explicit s_bulk or
    # s_ghost overrides with the lumped two-group model (lattices without
    # GHOST_RATES always use the two-group model).
    granular = s_bulk is None and s_ghost is None and hasattr(lattice, "GHOST_RATES")
    s_bulk = lattice.DEFAULT_S_BULK if s_bulk is None else s_bulk
    s_ghost = lattice.DEFAULT_S_GHOST if s_ghost is None else s_ghost

    if jnp.ndim(omega) == 0:
        S = jnp.zeros(Q)
        if granular:
            for idx, rate in lattice.GHOST_RATES.items():
                S = S.at[idx].set(rate)
        else:
            S = S.at[jnp.array(lattice.BULK_IDX)].set(s_bulk)
            S = S.at[jnp.array(lattice.GHOST_IDX)].set(s_ghost)
        return S.at[jnp.array(lattice.SHEAR_IDX)].set(omega)

    base = jnp.zeros((Q,) + omega.shape)
    if granular:
        for idx, rate in lattice.GHOST_RATES.items():
            base = base.at[idx].set(rate)
    else:
        base = base.at[jnp.array(lattice.BULK_IDX)].set(s_bulk)
        base = base.at[jnp.array(lattice.GHOST_IDX)].set(s_ghost)
    return base.at[jnp.array(lattice.SHEAR_IDX)].set(omega[None])


def uniform_S(lattice, omega):
    """Rate vector with EVERY moment relaxed at `omega` -> reproduces BGK. Used by
    the BGK-equivalence tests and as a simple way to get a pure-BGK MRT path."""
    Q = lattice.M.shape[0]
    if jnp.ndim(omega) == 0:
        return jnp.full(Q, float(omega))
    return jnp.broadcast_to(omega[None], (Q,) + omega.shape)
