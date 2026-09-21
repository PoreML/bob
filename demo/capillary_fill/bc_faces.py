"""Zou-He pressure boundaries for any domain face (x, y or z; low or high side) — the
axis-generalized counterparts of ``bob.bc.zou_he_inlet`` / ``bob.bc.zou_he_outlet``,
which act on the x faces only.

The pore junctions of this demo vent through the y faces as well, so the generalization
lives here with the demo.

The math is the same unified Zou-He reconstruction both solver functions use.
For a face with inward normal n (unit vector into the domain), the five
populations with e_i . n > 0 are unknown after streaming; with the face density
fixed and tangential velocity zero,

    rho u_n = rho_fixed - S0 - 2 S_out          (mass balance)
    f_i     = f_opp(i) + rho u_n * (1/3 axial | 1/6 diagonal) - t_i . N

where S0 sums the face-tangential populations (e_i . n = 0), S_out the outgoing
ones (e_i . n < 0), t_i is e_i's tangential part and N the tangential momentum
correction of the S0 set. The inlet fixes rho per color (rho_in*sa_red red);
the outlet fixes the total rho and splits the reconstruction by the phase
weight chi = (1 +- phi)/2 with phi zero-gradient from the neighbouring plane —
or, with ``sa_red`` given, by that fixed split. ``sa_red`` anchors the port colour
to the far-field fluid it opens onto: any backflow then injects that fluid, and an
arm filled by the other phase cannot turn the port into a reservoir of it (which a
zero-gradient colour does once the phase reaches the port).

On the x faces both functions reproduce ``bob.bc`` to within float32 rounding (1 ULP —
the generic code sums in a different order);
``uv run python demo/capillary_fill/bc_faces.py`` checks it.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from bob import d3q19
from bob.color3d import State

_EPS = 1e-12
_C = np.asarray(d3q19.C)                     # (19, 3) as (cx, cy, cz)
_OPP = np.asarray(d3q19.OPP)
_AX_COL = {"x": 0, "y": 1, "z": 2}           # column of C for each axis
_AX_DIM = {"z": 1, "y": 2, "x": 3}           # array axis of (19, nz, ny, nx) fields


def _face_tables(axis, side):
    """Static index tables for a face: unknown/tangential populations and the
    tangential-correction sign of each unknown against the two tangential axes."""
    col = _AX_COL[axis]
    n_in = 1 if side == 0 else -1            # inward normal component along `axis`
    unknown = [i for i in range(19) if _C[i, col] * n_in > 0]
    tang = [i for i in range(19) if _C[i, col] == 0 and i != 0]
    out = [i for i in range(19) if _C[i, col] * n_in < 0]
    tcols = [c for c in range(3) if c != col]
    return col, n_in, unknown, tang, out, tcols


def _plane(f, axis, side):
    dim = _AX_DIM[axis] - (4 - f.ndim)       # population field (19,nz,ny,nx) or plain (nz,ny,nx)
    return jnp.moveaxis(f, dim, -1)[..., 0 if side == 0 else -1]


def _unknown_vals(c, ru_n, tables):
    """Zou-He values of the unknown populations on a face plane ``c`` (19, a, b)
    for the normal momentum ``ru_n``: {i: value}."""
    _col, _n_in, unknown, tang, _out, tcols = tables
    N = {tc: 0.5 * sum(float(_C[i, tc]) * c[i] for i in tang) for tc in tcols}
    vals = {}
    for i in unknown:
        axial = all(_C[i, tc] == 0 for tc in tcols)
        val = c[int(_OPP[i])] + ru_n * (1.0 / 3.0 if axial else 1.0 / 6.0)
        for tc in tcols:
            if _C[i, tc]:
                val = val - float(_C[i, tc]) * N[tc]
        vals[i] = val
    return vals


def _set_plane(f, axis, side, vals, pore):
    """Write {i: value} into the face plane of ``f`` (open pore cells only)."""
    dim = _AX_DIM[axis]
    pos = 0 if side == 0 else f.shape[dim] - 1
    fm = jnp.moveaxis(f, dim, -1)
    for i, val in vals.items():
        if pore is not None:
            val = jnp.where(pore, val, fm[i, ..., pos])
        fm = fm.at[i, ..., pos].set(val)
    return jnp.moveaxis(fm, -1, dim)


def _recon_face(f, axis, side, ru_n, pore, tables):
    """Rebuild the unknown populations of one color field on one face plane."""
    return _set_plane(f, axis, side, _unknown_vals(_plane(f, axis, side), ru_n, tables), pore)


def _sums(c, tables):
    _, _, _, tang, out, _ = tables
    S0 = c[0] + sum(c[i] for i in tang)
    S_out = sum(c[i] for i in out)
    return S0, S_out


def zou_he_pressure_inlet(state, axis, side, solid=None, rho_in=1.0, sa_red=1.0):
    """Per-color Zou-He pressure inlet on face (axis, side); side 0 = low, -1 = high.
    Fixes the per-color density (rho_in*sa_red red / rest blue), tangential u = 0;
    the wall-normal inflow velocity is the free output. Generalizes
    ``bob.bc.zou_he_inlet`` (== it on axis='x', side=0)."""
    tables = _face_tables(axis, side)
    pore = None if solid is None else _plane(~jnp.asarray(solid), axis, side)

    def one(f, rho_c):
        c = _plane(f, axis, side)
        S0, S_out = _sums(c, tables)
        ru_n = rho_c - S0 - 2.0 * S_out
        return _recon_face(f, axis, side, ru_n, pore, tables)

    return State(one(state.fR, rho_in * sa_red), one(state.fB, rho_in * (1.0 - sa_red)))


def zou_he_pressure_outlet(state, axis, side, solid=None, rho_out=1.0, sa_red=None):
    """Per-color Zou-He pressure outlet on face (axis, side): fixes the total
    density, splits the reconstruction by zero-gradient phi from the neighbour
    plane — or by the fixed colour fraction ``sa_red`` when given (``sa_red=0``:
    the port opens onto the non-wetting far field; backflow is pure NWP).
    Generalizes ``bob.bc.zou_he_outlet`` (== it on axis='x', side=-1, same sa_red)."""
    tables = _face_tables(axis, side)
    pore = None if solid is None else _plane(~jnp.asarray(solid), axis, side)
    dim = _AX_DIM[axis]
    nb = 1 if side == 0 else state.fR.shape[dim] - 2

    cT = _plane(state.fR + state.fB, axis, side)
    S0, S_out = _sums(cT, tables)
    ru_nT = rho_out - S0 - 2.0 * S_out                       # total rho*u_n (negative = outflow)
    if sa_red is not None:
        tot = _unknown_vals(cT, ru_nT, tables)               # rebuild from the totals, fixed colour split
        sa = float(sa_red)
        fR = _set_plane(state.fR, axis, side, {i: sa * v for i, v in tot.items()}, pore)
        fB = _set_plane(state.fB, axis, side, {i: (1.0 - sa) * v for i, v in tot.items()}, pore)
        return State(fR, fB)

    fRm = jnp.moveaxis(state.fR, dim, -1)
    fBm = jnp.moveaxis(state.fB, dim, -1)
    rR2, rB2 = fRm[..., nb].sum(0), fBm[..., nb].sum(0)      # zero-gradient phi
    rho2 = rR2 + rB2
    phi = (rR2 - rB2) / jnp.where(rho2 > _EPS, rho2, 1.0)
    fR = _recon_face(state.fR, axis, side, ru_nT * 0.5 * (1.0 + phi), pore, tables)
    fB = _recon_face(state.fB, axis, side, ru_nT * 0.5 * (1.0 - phi), pore, tables)
    return State(fR, fB)


if __name__ == "__main__":
    # self-test: parity with bob.bc on the x faces, on a random two-phase state
    import jax

    from bob import bc

    key = jax.random.PRNGKey(0)
    kr, kb, ks = jax.random.split(key, 3)
    shape = (19, 6, 7, 8)
    fR = jax.random.uniform(kr, shape, minval=0.01, maxval=0.2)
    fB = jax.random.uniform(kb, shape, minval=0.01, maxval=0.2)
    solid = np.zeros(shape[1:], bool)
    solid[:, ::3, ::2] = True                                # some wall pattern
    s = State(fR, fB)

    a = bc.zou_he_inlet(s, solid=jnp.asarray(solid), rho_in=1.0, sa_red=1.0)
    b = zou_he_pressure_inlet(s, "x", 0, solid=solid, rho_in=1.0, sa_red=1.0)
    din = max(float(jnp.abs(a.fR - b.fR).max()), float(jnp.abs(a.fB - b.fB).max()))

    a = bc.zou_he_outlet(s, solid=jnp.asarray(solid), rho_out=1.0)
    b = zou_he_pressure_outlet(s, "x", -1, solid=solid, rho_out=1.0)
    dout = max(float(jnp.abs(a.fR - b.fR).max()), float(jnp.abs(a.fB - b.fB).max()))

    a = bc.zou_he_outlet(s, solid=jnp.asarray(solid), rho_out=1.0, sa_red=0.0)
    b = zou_he_pressure_outlet(s, "x", -1, solid=solid, rho_out=1.0, sa_red=0.0)
    dfix = max(float(jnp.abs(a.fR - b.fR).max()), float(jnp.abs(a.fB - b.fB).max()))

    print(f"inlet  max|diff| vs bob.bc.zou_he_inlet : {din:.3e}")
    print(f"outlet max|diff| vs bob.bc.zou_he_outlet: {dout:.3e}")
    print(f"outlet(sa_red=0) max|diff| vs bob.bc     : {dfix:.3e}")
    assert din < 5e-7 and dout < 5e-7 and dfix < 5e-7, "face BC does not reproduce bob.bc on the x faces"
    # sanity on a y face: mass-consistent reconstruction leaves finite fields
    c = zou_he_pressure_outlet(s, "y", -1, solid=solid, rho_out=1.0)
    assert bool(jnp.isfinite(c.fR).all() & jnp.isfinite(c.fB).all())
    print("y-face reconstruction finite: OK")
