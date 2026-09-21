"""Validation tests for the MRT (multiple-relaxation-time) collision operator.

Tier 0  D3Q19 moment-matrix / constant sanity (invertibility, conserved moments,
        rate-group partition)
Tier 1  collision invariants, single- and two-phase: MRT with uniform rates
        reproduces BGK exactly, and mass/momentum are conserved

Run:  uv run pytest test/test_mrt.py -v
"""

import jax
import jax.numpy as jnp
import numpy as np

# ---- Tier 0: D3Q19 matrix sanity ----------------------------------------- #
from bob import color3d, d3q19, lbm3d, misc, mrt


def test_d3q19_M_invertible_roundtrip():
    assert d3q19.M.shape == (19, 19)
    np.testing.assert_allclose(np.asarray(d3q19.Minv @ d3q19.M), np.eye(19), atol=1e-9)


def test_d3q19_M_recovers_conserved_moments():
    rho = jnp.array([[[1.0, 1.2]]])                      # (1,1,2)
    u = 0.03 * jnp.ones((3, 1, 1, 2))
    feq = lbm3d.equilibrium(rho, u)
    m = jnp.einsum("qp,pzyx->qzyx", d3q19.M, feq)
    np.testing.assert_allclose(np.asarray(m[d3q19.RHO_IDX]), np.asarray(rho), atol=1e-12)
    np.testing.assert_allclose(np.asarray(m[d3q19.JX_IDX]), np.asarray(rho * u[0]), atol=1e-12)
    np.testing.assert_allclose(np.asarray(m[d3q19.JY_IDX]), np.asarray(rho * u[1]), atol=1e-12)
    np.testing.assert_allclose(np.asarray(m[d3q19.JZ_IDX]), np.asarray(rho * u[2]), atol=1e-12)


def test_d3q19_rate_groups_partition_directions():
    idx = (d3q19.CONSERVED_IDX + d3q19.SHEAR_IDX + d3q19.BULK_IDX + d3q19.GHOST_IDX)
    assert sorted(idx) == list(range(19))
    assert len(d3q19.SHEAR_IDX) == 5      # 5 deviatoric stress moments set viscosity


# ---- Tier 1: lbm3d (3D single-phase) ------------------------------------- #
def _rand_f3(nz=4, ny=4, nx=4, seed=0):
    rho = 1.0 + 0.1 * jax.random.uniform(jax.random.PRNGKey(seed), (nz, ny, nx))
    u = 0.04 * jax.random.normal(jax.random.PRNGKey(seed + 1), (3, nz, ny, nx))
    return lbm3d.equilibrium(rho, u)


def test_lbm3d_mrt_equals_bgk_unforced():
    f = _rand_f3()
    omega = 1.25
    S = mrt.uniform_S(d3q19, omega)
    np.testing.assert_allclose(np.asarray(lbm3d.collide(f, omega, rates=S)),
                               np.asarray(lbm3d.collide(f, omega)), atol=1e-12)


def test_lbm3d_mrt_equals_bgk_forced():
    f = _rand_f3(seed=5)
    omega = 0.9
    force = jnp.array([1e-4, -2e-5, 3e-5])
    S = mrt.uniform_S(d3q19, omega)
    np.testing.assert_allclose(np.asarray(lbm3d.collide(f, omega, force, rates=S)),
                               np.asarray(lbm3d.collide(f, omega, force)), atol=1e-12)


def test_lbm3d_mrt_conserves_mass():
    f = _rand_f3(seed=2)
    S = mrt.build_S(d3q19, 1.4)
    out = lbm3d.collide(f, 1.4, rates=S)
    np.testing.assert_allclose(np.asarray(out.sum(0)), np.asarray(f.sum(0)), atol=1e-10)


# ---- Tier 1: color3d (3D two-phase) -------------------------------------- #
def test_color3d_step_mrt_equals_bgk_when_uniform():
    st = misc.init_sphere(24, 24, 24, R=7)
    p_bgk = color3d.Params(omega=1.2, sigma=0.02, beta=0.7, mrt=False)
    p_mrt = p_bgk._replace(mrt=True, s_bulk=1.2, s_ghost=1.2)
    a = color3d.step(st, p_bgk)
    b = color3d.step(st, p_mrt)
    np.testing.assert_allclose(np.asarray(b.fR), np.asarray(a.fR), atol=1e-11)
    np.testing.assert_allclose(np.asarray(b.fB), np.asarray(a.fB), atol=1e-11)
