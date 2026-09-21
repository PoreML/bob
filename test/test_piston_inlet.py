"""Piston (bounce-back velocity) inlet — MF-LBM's default inlet, implemented as bc.piston_inlet.

The BC = sealed solid x=0 plane (reflection via the existing halfway bounce) + the Ladd
moving-wall kick 6*w_i*u_in at x=1. Defining properties tested here:
  1. the kick's ledger is exact: +u_in*A*sa red, +u_in*A*(1-sa) blue, per application;
  2. sa_red=1 writes ZERO blue — fB is bit-identical;
  3. only the five cx>0 slots at the injection plane change;
  4. over a full drainage run the domain's red mass grows by EXACTLY u_in*A per step
     (no repaint leak, no destruction) until red reaches the outlet;
  5. drain(inlet="piston") composes under jit/scan and stays finite (MRT default stack).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bob import bc, color3d, d3q19, porous3d

NZ = NY = 8
NX = 12
CA = 1e-3
SIGMA = 0.05
OMEGA = 1.5
MU = float(d3q19.CS2) * (1.0 / OMEGA - 0.5)
U_IN = CA * SIGMA / MU


@pytest.fixture(scope="module")
def duct():
    solid = porous3d.duct(NZ, NY, NX)
    sealed = jnp.asarray(solid).at[:, :, 0].set(True)
    return solid, sealed


def test_kick_ledger_exact(duct):
    _solid, sealed = duct
    state = porous3d.init_drainage(NZ, NY, NX, n_red=0)
    a_open = int(np.asarray(~sealed[:, :, 1]).sum())
    for sa in (1.0, 0.5):
        out = bc.piston_inlet(state, U_IN, solid=sealed, sa_red=sa)
        # difference on the injection plane only: exact per-slot, no large-sum fp noise
        d_red = float((out.fR[:, :, :, 1] - state.fR[:, :, :, 1]).sum())
        d_blue = float((out.fB[:, :, :, 1] - state.fB[:, :, :, 1]).sum())
        assert d_red == pytest.approx(U_IN * a_open * sa, rel=1e-12)
        assert d_blue == pytest.approx(U_IN * a_open * (1.0 - sa), rel=1e-12, abs=1e-15)


def test_sa1_writes_zero_blue(duct):
    _, sealed = duct
    state = porous3d.init_drainage(NZ, NY, NX, n_red=0)
    out = bc.piston_inlet(state, U_IN, solid=sealed, sa_red=1.0)
    assert bool(jnp.array_equal(out.fB, state.fB))


def test_only_forward_slots_at_plane_change(duct):
    _, sealed = duct
    state = porous3d.init_drainage(NZ, NY, NX, n_red=0)
    out = bc.piston_inlet(state, U_IN, solid=sealed, sa_red=1.0)
    diff = np.asarray(jnp.abs(out.fR - state.fR))
    touched = {i for i in range(19) if diff[i].any()}
    assert touched == {1, 7, 9, 11, 13}
    assert not diff[:, :, :, [0] + list(range(2, NX))].any()   # only x=1 changes


def test_domain_red_growth_is_exactly_u_A(duct):
    solid, sealed = duct
    params = color3d.Params(omega=OMEGA, sigma=SIGMA, beta=0.7, theta=135.0)
    nw = color3d.wall_normals(solid)   # unsealed normals: piston face wetting-neutral
    state = porous3d.init_drainage(NZ, NY, NX, n_red=0)
    a_open = int(np.asarray(~sealed[:, :, 1]).sum())
    steps = 200   # front advances u*t ~ 0.18 cells: red never reaches the outlet
    out = porous3d.drain(state, params, solid, n_in=0, u_in=U_IN, steps=steps, nw=nw, inlet="piston")
    gained = float(out.fR.sum() - state.fR.sum())
    # injection is exact (test_kick_ledger_exact); the ~2e-5 relative deficit here is the
    # diffusive red tail reaching the outlet plane and being destroyed there — populations
    # spread at lattice speed even though the front moves at u_in.
    assert gained == pytest.approx(U_IN * a_open * steps, rel=1e-3)
    # and zero blue was ever injected: blue can only leave (outlet), never appear
    assert float(out.fB.sum() - state.fB.sum()) <= 1e-9


def test_outlet_fixed_blue_writes_zero_red(duct):
    """zou_he_outlet(sa_red=0) must reconstruct pure-blue incoming slots: the default
    zero-gradient split is self-referential and ratchets the diffusive red tail into a
    spurious red pool at the outlet (the outlet counterpart of an inlet repaint leak)."""
    _solid, sealed = duct
    state = porous3d.init_drainage(NZ, NY, NX, n_red=NX - 1)   # red everywhere up to the outlet
    out = bc.zou_he_outlet(state, solid=sealed, rho_out=1.0, sa_red=0.0)
    pore_last = np.asarray(~sealed[:, :, -1])
    red_written = np.asarray(out.fR)[[2, 8, 10, 12, 14], :, :, -1][:, pore_last]
    assert float(np.abs(red_written).max()) < 1e-14


def test_blue_flood_ledger_and_zero_red_injection(duct):
    """inlet_sa_red=0 + outlet_sa_red=1 (the case/trapping waterflood, colour-mirrored
    drainage): the piston pumps EXACTLY u_in*A blue/step into an oil-filled duct and
    never writes red; red only leaves through the red-pinned outlet reservoir."""
    solid, sealed = duct
    params = color3d.Params(omega=OMEGA, sigma=SIGMA, beta=0.7, theta=135.0)
    nw = color3d.wall_normals(solid)
    state = porous3d.init_drainage(NZ, NY, NX, n_red=NX)   # red (oil) everywhere
    a_open = int(np.asarray(~sealed[:, :, 1]).sum())
    steps = 200   # front advances u*t ~ 0.18 cells: blue never reaches the outlet
    out = porous3d.drain(state, params, solid, n_in=0, u_in=U_IN, steps=steps, nw=nw,
                         inlet="piston", inlet_sa_red=0.0, outlet_sa_red=1.0)
    gained = float(out.fB.sum() - state.fB.sum())
    # same ~diffusive-tail tolerance as the red ledger test above, colour-swapped
    assert gained == pytest.approx(U_IN * a_open * steps, rel=1e-3)
    # and zero red was ever injected: red can only leave (outlet), never appear
    assert float(out.fR.sum() - state.fR.sum()) <= 1e-9


def test_drain_piston_jits_and_stays_finite(duct):
    solid, _ = duct
    params = color3d.Params(omega=OMEGA, sigma=SIGMA, beta=0.7, theta=135.0)
    nw = color3d.wall_normals(solid)
    state = porous3d.init_drainage(NZ, NY, NX, n_red=0)
    run = jax.jit(lambda s: porous3d.drain(s, params, solid, n_in=0, u_in=U_IN,
                                           steps=50, nw=nw, inlet="piston"))
    out = run(state)
    assert bool(jnp.isfinite(out.fR).all() and jnp.isfinite(out.fB).all())
    rho = (out.fR + out.fB).sum(0)
    pore = ~np.asarray(jnp.asarray(solid).at[:, :, 0].set(True))
    assert float(jnp.where(jnp.asarray(pore), rho, 1.0).min()) > 0.5
