"""Washburn's law — spontaneous capillary imbibition in a 3D tube (sqrt-t scaling and rate)."""

import jax
import jax.numpy as jnp
import numpy as np

from bob import color3d, d3q19, porous3d
from bob.utils import washburn


def test_cylinder_is_uniform_round_tube():
    R = 6
    solid = np.asarray(porous3d.cylinder(21, 21, 40, R))
    assert solid.shape == (21, 21, 40)
    assert (solid[:, :, 0] == solid[:, :, -1]).all()  # uniform along x
    assert not solid[10, 10, 0]  # centre is pore
    assert solid[0, 0, 0]  # corner is solid
    area = (~solid[:, :, 0]).sum()  # cross-section pore count
    assert abs(area - np.pi * R**2) / (np.pi * R**2) < 0.15  # ~ pi R^2 (staircased)


def test_imbibe_equals_drain_zero_velocity_3d():
    solid = porous3d.cylinder(13, 13, 30, R=4)
    nw = color3d.wall_normals(np.asarray(solid))
    p = color3d.Params(omega=1.0, sigma=0.02, beta=0.7, theta=40.0, omega2=1.82)
    s0 = porous3d.init_drainage(13, 13, 30, n_red=5)
    a = porous3d.imbibe(s0, p, solid, n_in=4, steps=20, nw=nw)
    b = porous3d.drain(s0, p, solid, n_in=4, u_in=0.0, steps=20, nw=nw)
    np.testing.assert_allclose(np.asarray(a.fR), np.asarray(b.fR), atol=1e-12)


def test_filled_length_counts_penetration_3d():
    solid = np.asarray(porous3d.cylinder(13, 13, 20, R=4))
    rhoR = np.zeros((13, 13, 20))
    rhoB = np.zeros((13, 13, 20))
    pore = ~solid
    rhoR[..., :10] = pore[..., :10]                    # red fills pore in slabs 0..9
    rhoB[..., 10:] = pore[..., 10:]
    W = np.asarray(d3q19.W)[:, None, None, None]
    st = color3d.State(jnp.asarray(W * rhoR), jnp.asarray(W * rhoB))
    cross = int(pore[:, :, 4].sum())
    expected = int(pore[:, :, 4:10].sum()) / cross      # red occupies slabs 4..9
    assert abs(porous3d.filled_length(st, solid, x0=4) - expected) < 1e-9


# Sedahmed & Coelho, Phys. Fluids 36, 092117 (2024), DOI 10.1063/5.0228835, §IV.A (binary fluid
# flow in a horizontal capillary tube) — strict settings: gamma = 1/45, viscosity ratio
# M = nu_nw/nu_w = 1/5 (tau_w=1.0 -> nu_w=1/6, tau_nw=0.6 -> nu_nw=1/30), beta=0.95.
# The wetting (more viscous, invading) fluid imbibes the non-wetting one, dP=0.
#
# Boundary condition: the faithful dP=0 setup is a per-color Zou-He PRESSURE inlet
# (inlet="zouhe") with the default solver (csf + akai + Eq.34/35 + recolor_emag +
# wall_grad="fluid"). The inflow velocity is a free output of the capillary suction
# (vs the u=0 clamp, which throttles imbibition ~2x). On a BARE walled tube the
# single-layer Zou-He is supply-limited (its inflow is set by the resistive near-wall
# chamber, so A_meas/A_an ~ 0.4-0.6); a frictionless/wide reservoir recovers the full
# analytic rate (demo/washburn). The R^2 (sqrt-t scaling) is the physics check; the
# A bound below brackets the supply-limited bare-tube rate.
_WASH_GAMMA = 1.0 / 45
_WASH_OM2 = 1.0 / 0.6                 # nu_nw = cs2 (1/om2 - 1/2) = 1/30  ->  M = 1/5


def test_washburn_3d_tube():
    nz, ny, nx, R, n_in = 11, 11, 90, 4, 5             # paper's faithful 3D cylindrical tube
    solid = porous3d.cylinder(nz, ny, nx, R=R)
    nw = color3d.wall_normals(np.asarray(solid))
    p = color3d.Params(omega=1.0, sigma=_WASH_GAMMA, beta=0.95, theta=40.0,
                     omega2=_WASH_OM2)                   # default solver: csf+akai+Eq.34/35+emag+fluid
    runner = jax.jit(lambda s: porous3d.imbibe(s, p, solid, n_in, 1500, nw=nw, inlet="zouhe"))
    state = porous3d.init_drainage(nz, ny, nx, n_red=n_in)
    ts, Ls, step = [], [], 0
    for _ in range(24):
        state = runner(state)
        step += 1500
        ts.append(step)
        Ls.append(porous3d.filled_length(state, solid, x0=n_in))
    muw, munw = float(d3q19.CS2) * 0.5, float(d3q19.CS2) * (1.0 / _WASH_OM2 - 0.5)
    y = washburn.rectify(Ls, muw, munw, nx - n_in)
    A_meas, _, r2 = washburn.fit_slope(ts, y, 0.25, 0.95)
    assert r2 > 0.97, f"two-fluid Washburn not linear: R^2={r2:.3f}"
    A_an = washburn.capillary_A(3, R, _WASH_GAMMA, 40.0)
    # supply-limited bare-tube Zou-He (~0.4 of analytic; observed 0.39, R^2~1.000)
    assert 0.28 < A_meas / A_an < 1.2, f"A_meas/A_analytic={A_meas / A_an:.2f} off"


def test_capillary_A_matches_formula():
    a2 = washburn.capillary_A(2, 20, 0.02, 60.0)
    assert abs(a2 - 20 * 0.02 * 0.5 / 6.0) < 1e-12
    a3 = washburn.capillary_A(3, 12, 0.02, 0.0)
    assert abs(a3 - 12 * 0.02 * 1.0 / 4.0) < 1e-12


def test_two_fluid_reduces_to_sqrt_t_when_displaced_inviscid():
    # mu_O -> 0 limit: L^2 = (2A/mu_I) t  (the one-fluid sqrt law)
    A, mu_I = 0.01, 0.1
    ts = np.linspace(1.0, 1000.0, 50)
    L = washburn.two_fluid_Lt(ts, A, mu_I, 1e-9, 100.0, L0=0.0)
    assert np.allclose(L, np.sqrt(2.0 * A / mu_I * ts), rtol=1e-3)


def test_rectified_two_fluid_is_linear_in_t():
    A, mu_I, mu_O, L_tube = 0.02, 0.1, 0.01, 120.0
    ts = np.linspace(1.0, 2000.0, 60)
    L = washburn.two_fluid_Lt(ts, A, mu_I, mu_O, L_tube, L0=0.0)
    y = washburn.rectify(L, mu_I, mu_O, L_tube)
    slope, _intercept, r2 = washburn.fit_slope(ts, y, 0.0, 1.0)
    assert abs(slope - A) < 1e-6
    assert r2 > 0.9999
