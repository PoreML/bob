"""Inlet / outlet boundary conditions for the D3Q19 two-phase solver.

Decoupled from ``bob.color3d`` (which owns the physics step); ``porous3d.drain``
composes these around the step, and demos call them directly in custom bodies.
x is the flow axis throughout: inlets act on the x=0 plane (the piston on the
first FLUID slab behind a sealed x=0), outlets on x=-1.

All plane/slab updates follow the GSPMD-safe convention (elementwise
``jnp.where`` on index masks / ``jnp.roll`` — never paired ``.at[].set``
slice-updates at opposite ends of a shardable axis; see ``bob.multigpu``).
Everything is pure ``State -> State`` and jit/scan-composable.
"""

import jax.numpy as jnp
import numpy as np

from bob import d3q19, lbm3d
from bob.color3d import State

__all__ = [
    "eqm_velocity_inlet",
    "inlet_reservoir",
    "outlet_zero_gradient",
    "piston_inlet",
    "zou_he_inlet",
    "zou_he_outlet",
]

_EPS = 1e-12


def inlet_reservoir(state, n_in, u_in=0.0, solid=None):
    """Dirichlet inlet: hold slabs x < n_in as a pure-red (non-wetting) reservoir
    at equilibrium with unit density and velocity (u_in, 0, 0). Masks the
    periodic wrap at x = 0.

    Pass ``solid`` to pin only pore cells. Without it the pin also overwrites WALL
    nodes inside the slabs with pure-red equilibrium — whose diagonal populations
    then stream out of the wall into the next plane's fluid every step, a
    perimeter-scale red source (measured +2.2 red/step through the plane-3 walls
    of an 8x8x14 duct). Any colour-accurate use should pass it.

    Written as an elementwise ``jnp.where`` on an x-index mask (NOT an
    ``.at[:, :, :, :n_in].set``): jax 0.6.2's GSPMD partitioner miscompiles a
    slab slice-update followed by another slice-update at the far end of the
    SAME sharded axis (this BC + ``outlet_zero_gradient`` under
    ``multigpu.shard(axis="x")`` is wrong from step 1); the where/roll forms
    partition exactly on every axis. Values are bit-identical to the slice
    form."""
    nz, ny, nx = state.fR.shape[1], state.fR.shape[2], state.fR.shape[3]
    feq = lbm3d.equilibrium(jnp.ones((nz, ny, 1)), jnp.zeros((3, nz, ny, 1)).at[0].set(u_in))  # constant along x
    slab = jnp.arange(nx) < n_in
    if solid is not None:
        slab = slab & ~jnp.asarray(solid)
    return State(jnp.where(slab, feq, state.fR), jnp.where(slab, 0.0, state.fB))


def zou_he_inlet(state, solid=None, rho_in=1.0, sa_red=1.0):
    """Per-color D3Q19 x=0 Zou-He pressure inlet — the ΔP=0 imbibition driver — faithful to MF-LBM (Boundary_multiphase_inlet.F90).

    Fixes the per-color inlet density (pressure) at ``rho_in*sa_red`` (red) /
    ``rho_in*(1-sa_red)`` (blue) with tangential u_y=u_z=0, reconstructing only the five
    unknown incoming populations {1,7,9,11,13} from the knowns; the wall-normal inflow
    velocity is a FREE output of the capillary suction (vs the u=0 ``inlet_reservoir``
    clamp, which pins it and throttles imbibition). With ``solid`` given, only open pore
    cells on the inlet plane are reconstructed (solid cells keep their values).
    ``sa_red=1`` injects pure wetting (red)."""
    pore0 = None if solid is None else ~jnp.asarray(solid)[:, :, 0]   # (nz, ny) open inlet cells

    def recon(f, rho_c):
        c = f[:, :, :, 0]      # (19, nz, ny) inlet plane
        S0 = c[0] + c[3] + c[4] + c[5] + c[6] + c[15] + c[16] + c[17] + c[18]   # cx=0  (known)
        Sm = c[2] + c[8] + c[10] + c[12] + c[14]                                # cx<0  (known)
        tmp = rho_c - S0 - 2.0 * Sm                                             # = rho_c * u_x
        Ny = 0.5 * ((c[3] + c[15] + c[17]) - (c[4] + c[16] + c[18]))            # y-momentum of cx=0
        Nz = 0.5 * ((c[5] + c[15] + c[18]) - (c[6] + c[16] + c[17]))            # z-momentum of cx=0
        new = {1: c[2] + tmp / 3.0, 7: c[8] + tmp / 6.0 - Ny, 9: c[10] + tmp / 6.0 + Ny,
               11: c[12] + tmp / 6.0 - Nz, 13: c[14] + tmp / 6.0 + Nz}
        for idx, val in new.items():
            plane = val if pore0 is None else jnp.where(pore0, val, c[idx])
            f = f.at[idx, :, :, 0].set(plane)
        return f

    return State(recon(state.fR, rho_in * sa_red), recon(state.fB, rho_in * (1.0 - sa_red)))


def eqm_velocity_inlet(state, u_in, solid=None, sa_red=1.0):
    """MRT-stable rate-controlled velocity inlet (x=0): impose u_x=u_in, density FLOATS.

    A raw Zou-He *velocity* reconstruction (fix u, rebuild only the five
    unknown populations) leaves the higher (ghost) moments uncontrolled — which MRT
    relaxes separately, making the floating-density velocity BC UNSTABLE under the
    default MRT stack regardless of Ca (it is stable only under BGK). This inlet
    instead overwrites the x=0 plane with the FULL equilibrium ``f_i = w_i rho (...)``
    at velocity (u_in, 0, 0) — so every moment is consistent (MRT-stable, exactly why
    the ``inlet_reservoir`` clamp is stable) — but takes the density ``rho`` by
    ZERO-GRADIENT from the neighbour plane x=1 instead of pinning it to 1. So the inlet
    pressure rises on its own against a capillary barrier (rate-controlled drainage that
    actually drives the front; Ca = mu*u_in/sigma), unlike the clamp which pins rho=1 and
    cannot build pressure. ``sa_red=1`` injects pure red (non-wetting); with ``solid``
    only open pore cells on the inlet plane are set."""
    nz, ny = state.fR.shape[1], state.fR.shape[2]
    rho_nb = (state.fR + state.fB).sum(0)[:, :, 1]                       # (nz,ny) zero-gradient density from x=1
    u = jnp.zeros((3, nz, ny, 1)).at[0].set(u_in)
    feq = lbm3d.equilibrium(rho_nb[:, :, None], u)[:, :, :, 0]           # (19,nz,ny) full equilibrium at (rho_nb, u_in)
    fR0, fB0 = sa_red * feq, (1.0 - sa_red) * feq
    if solid is None:
        return State(state.fR.at[:, :, :, 0].set(fR0), state.fB.at[:, :, :, 0].set(fB0))
    pore0 = ~jnp.asarray(solid)[:, :, 0]
    fR = state.fR.at[:, :, :, 0].set(jnp.where(pore0, fR0, state.fR[:, :, :, 0]))
    fB = state.fB.at[:, :, :, 0].set(jnp.where(pore0, fB0, state.fB[:, :, :, 0]))
    return State(fR, fB)


def piston_inlet(state, u_in, solid=None, sa_red=1.0, plane=1):
    """Bounce-back velocity inlet (Ladd moving piston) — MF-LBM's default inlet
    (``Boundary_multiphase_inlet.F90``, ``inlet_bounce_back_velocity_BC_*``).

    REQUIRES the plane behind ``plane`` (normally x=0) to be SOLID in the mask
    passed to ``step``: the half-way bounce-back then reflects each colour's
    counter-stream off that face exactly (blue bounces back as blue — never
    destroyed, never repainted), the periodic x-wrap is cut by the solid plane,
    and this BC only adds the Ladd moving-wall momentum kick at the first fluid
    slab ``x=plane``:

        f_i  +=  6 * w_i * u_in * sa_red        (red,  cx>0 slots {1,7,9,11,13})
        g_i  +=  6 * w_i * u_in * (1 - sa_red)  (blue)

    Sum(6*w_i) over the five slots = 1, so the injected mass is exactly
    ``rho0 * u_in * sa_red`` per pore node per step — the swept volume of a
    piston face advancing at ``u_in``. The colour ledger is therefore metered by
    the *velocity* (no Ca-independent repaint leak, no colour ratchet: the
    reflection preserves colour identity per population and the anchor is the
    fixed injection rate). At
    ``sa_red=1`` zero blue is ever written — blue sees a plain impermeable
    piston and its pressure builds, which IS the entry-pressure physics.

    ``u_in`` sets Ca directly (Ca = mu*u_in/sigma); the inlet pressure is the
    free output. Velocity is imposed per NODE (MF-LBM's uniform ``w_in``) — keep
    open buffer slabs between the piston and the rock, and monitor inlet density
    if the medium can block completely. Pair with ``zou_he_outlet``; apply after
    streaming (``porous3d.drain(inlet="piston")`` does, and seals x=0 itself).
    ``plane`` must be a static Python int."""
    plane = int(plane)
    pore = None if solid is None else ~jnp.asarray(solid)[:, :, plane]
    fR, fB = state.fR, state.fB
    for i in (1, 7, 9, 11, 13):
        wi = float(np.asarray(d3q19.W)[i])   # via numpy: stays concrete inside jit/scan traces
        kr = 6.0 * wi * u_in * sa_red
        kb = 6.0 * wi * u_in * (1.0 - sa_red)
        if pore is None:
            fR = fR.at[i, :, :, plane].add(kr)
            fB = fB.at[i, :, :, plane].add(kb)
        else:
            fR = fR.at[i, :, :, plane].add(jnp.where(pore, kr, 0.0))
            fB = fB.at[i, :, :, plane].add(jnp.where(pore, kb, 0.0))
    return State(fR, fB)


def zou_he_outlet(state, solid=None, rho_out=1.0, sa_red=None):
    """Per-color D3Q19 x=-1 Zou-He pressure OUTLET — after Sedahmed & Coelho (Phys. Fluids
    36, 092117 (2024), App. B, Eqs B.9-B.18) + zero-gradient phi (Eq. 39).

    Fixes the outlet density ``rho_out``; the *total* tangential (normal-to-outlet)
    velocity is computed from both fluids (Eq. B.9) and split per color by the phase
    weight ``chi_r=(1+phi)/2``, ``chi_b=(1-phi)/2`` (Eqs B.12-B.13), with ``phi`` taken
    zero-gradient from the penultimate slab (Eq. 39). Only the five unknown incoming
    (-x) populations {2,8,10,12,14} are reconstructed per color (Eqs B.14-B.18); the
    outgoing flux leaves freely. With ``solid`` given, only open pore cells on the
    outlet plane are reconstructed."""
    pore0 = None if solid is None else ~jnp.asarray(solid)[:, :, -1]   # (nz, ny)
    cT = (state.fR + state.fB)[:, :, :, -1]                            # (19, nz, ny) total at outlet
    S0 = cT[0] + cT[3] + cT[4] + cT[5] + cT[6] + cT[15] + cT[16] + cT[17] + cT[18]   # cx=0  (known)
    Sp = cT[1] + cT[7] + cT[9] + cT[11] + cT[13]                                     # cx>0  (known, leaving)
    tmp = -rho_out + S0 + 2.0 * Sp                                     # = rho * u_x  (Eq. B.9)
    if sa_red is not None:
        # FIXED colour split of the TOTAL reconstruction. The default zero-gradient phi is
        # self-referential: whatever colour reaches the outlet, the reconstruction re-writes
        # — the diffusive red tail seeds it and ratchets a spurious red pool at the outlet
        # (the outlet counterpart of the inlet repaint leak that ``piston_inlet`` avoids).
        # ``sa_red=0`` anchors the outlet to the defending (blue)
        # reservoir: the five incoming slots are rebuilt from the TOTALS and painted
        # sa_red red / (1-sa_red) blue, so with 0 no red is ever re-created and arriving
        # red genuinely exits. Use for drainage with a leak-free inlet (e.g. the piston).
        NyT = 0.5 * ((cT[3] + cT[15] + cT[17]) - (cT[4] + cT[16] + cT[18]))
        NzT = 0.5 * ((cT[5] + cT[15] + cT[18]) - (cT[6] + cT[16] + cT[17]))
        newT = {2: cT[1] - tmp / 3.0, 8: cT[7] - tmp / 6.0 + NyT, 10: cT[9] - tmp / 6.0 - NyT,
                12: cT[11] - tmp / 6.0 + NzT, 14: cT[13] - tmp / 6.0 - NzT}
        fR, fB = state.fR, state.fB
        for idx, tot in newT.items():
            fr, fb = float(sa_red) * tot, (1.0 - float(sa_red)) * tot
            if pore0 is not None:
                fr = jnp.where(pore0, fr, fR[idx, :, :, -1])
                fb = jnp.where(pore0, fb, fB[idx, :, :, -1])
            fR = fR.at[idx, :, :, -1].set(fr)
            fB = fB.at[idx, :, :, -1].set(fb)
        return State(fR, fB)
    rR2, rB2 = state.fR[:, :, :, -2].sum(0), state.fB[:, :, :, -2].sum(0)  # zero-gradient phi (Eq. 39)
    rho2 = rR2 + rB2
    phi = (rR2 - rB2) / jnp.where(rho2 > _EPS, rho2, 1.0)              # (nz, ny)

    def recon(f, chi):
        c = f[:, :, :, -1]
        Ny = 0.5 * ((c[3] + c[15] + c[17]) - (c[4] + c[16] + c[18]))   # per-color transverse y
        Nz = 0.5 * ((c[5] + c[15] + c[18]) - (c[6] + c[16] + c[17]))   # per-color transverse z
        t = tmp * chi
        new = {2: c[1] - t / 3.0, 8: c[7] - t / 6.0 + Ny, 10: c[9] - t / 6.0 - Ny,
               12: c[11] - t / 6.0 + Nz, 14: c[13] - t / 6.0 - Nz}
        for idx, val in new.items():
            plane = val if pore0 is None else jnp.where(pore0, val, c[idx])
            f = f.at[idx, :, :, -1].set(plane)
        return f

    return State(recon(state.fR, 0.5 * (1.0 + phi)), recon(state.fB, 0.5 * (1.0 - phi)))


def outlet_zero_gradient(state):
    """Open outflow: copy slab x = -2 into x = -1 rescaled to unit total density —
    the pressure anchor (without it constant-rate injection just compresses the
    domain). Masks the periodic wrap at x = -1.

    Written as ``jnp.roll`` + ``jnp.where`` on the last x-plane (NOT an
    ``.at[:, :, :, -1].set``) — see ``inlet_reservoir`` for the GSPMD
    slice-update pitfall this dodges; values are bit-identical."""
    nx = state.fR.shape[3]
    rho_prev = jnp.roll((state.fR + state.fB).sum(0), 1, axis=-1)  # at x=-1: rho(x=-2)
    scale = 1.0 / jnp.where(rho_prev > _EPS, rho_prev, 1.0)
    last = jnp.arange(nx) == nx - 1
    fR = jnp.where(last, jnp.roll(state.fR, 1, axis=-1) * scale, state.fR)
    fB = jnp.where(last, jnp.roll(state.fB, 1, axis=-1) * scale, state.fB)
    return State(fR, fB)
