"""Params.mrt_chi: opt-in Leclaire-style MRT spectrum.

Contract:
  - mrt_chi=None (default) -> the default fixed spectrum (per-moment d3q19.GHOST_RATES, or the
    lumped two-group rates on lattices without them).
  - mrt_chi=chi -> every non-shear (bulk + ghost) moment relaxes at chi*omega, tied to the
    local shear rate (Leclaire 2017, chi=4/5). Shear stays omega (viscosity unchanged),
    conserved stay 0.
  - mrt_chi=1.0 -> SRT/BGK-equivalent (all non-shear = omega); this is NOT the granular default.
"""

import jax.numpy as jnp
import numpy as np

from bob import color3d, d3q19, lbm3d, mrt


def test_build_S_chi_none_is_granular_default_3d():
    S = np.asarray(mrt.build_S(d3q19, 1.8, chi=None))
    assert S[4] == 1.8 and S[8] == 1.8  # shear -> omega
    assert S[0] == 0.0 and S[3] == 0.0  # conserved -> 0
    assert S[9] == 1.19  # bulk (granular MF-LBM default)
    assert S[13] == 1.98 and S[10] == 1.2 and S[16] == 1.4  # ghost (granular spectrum)
    assert np.allclose(S, np.asarray(mrt.build_S(d3q19, 1.8)))  # chi=None == no-chi default


def test_build_S_chi_value_ties_nonshear_to_omega_3d():
    S = np.asarray(mrt.build_S(d3q19, 1.8, chi=0.8))
    assert S[4] == 1.8 and S[8] == 1.8  # shear unchanged (viscosity preserved)
    assert S[0] == 0.0  # conserved
    nonshear = [S[i] for i in (9, 10, 11, 12, 13, 14, 15, 16, 17, 18)]
    assert all(abs(v - 1.44) < 1e-12 for v in nonshear)  # 0.8 * 1.8


def test_build_S_chi_one_is_srt_not_default_3d():
    S1 = np.asarray(mrt.build_S(d3q19, 1.8, chi=1.0))
    assert all(abs(S1[i] - 1.8) < 1e-12 for i in (9, 13, 16))  # all non-shear = omega (SRT)
    assert not np.allclose(S1, np.asarray(mrt.build_S(d3q19, 1.8, chi=None)))  # != granular default


def test_build_S_chi_field_omega_3d():
    omega = jnp.full((2, 3, 4), 1.5)
    S = mrt.build_S(d3q19, omega, chi=0.8)
    assert S.shape == (19, 2, 3, 4)
    assert jnp.allclose(S[4], 1.5)  # shear
    assert jnp.allclose(S[13], 1.2)  # ghost = 0.8 * 1.5
    assert jnp.allclose(S[0], 0.0)  # conserved


def test_params_mrt_chi_default_none():
    assert color3d.Params(omega=1.0, sigma=0.02, beta=0.7).mrt_chi is None


def _noneq_red_step(p):
    """One color3d step from a NON-equilibrium pure-red state. Pure red -> no color gradient,
    so perturbation/recolor are inert and the MRT collision is the only thing acting on the
    non-equilibrium part — isolating the effect of the spectrum (chi)."""
    nz, ny, nx = 3, 4, 5
    base = lbm3d.equilibrium(jnp.ones((nz, ny, nx)), jnp.zeros((3, nz, ny, nx)))
    pert = 0.01 * jnp.cos(jnp.arange(19, dtype=jnp.float64))[:, None, None, None]  # excites bulk+ghost+shear
    return color3d.step(color3d.State(base + pert, jnp.zeros_like(base)), p)


def test_step_mrt_chi_none_matches_default_3d():
    base = _noneq_red_step(color3d.Params(omega=1.0, sigma=0.02, beta=0.7))
    none = _noneq_red_step(color3d.Params(omega=1.0, sigma=0.02, beta=0.7, mrt_chi=None))
    assert jnp.allclose(base.fR, none.fR)  # mrt_chi=None is identical to the default spectrum


def test_step_mrt_chi_value_changes_result_3d():
    base = _noneq_red_step(color3d.Params(omega=1.0, sigma=0.02, beta=0.7))
    chi = _noneq_red_step(color3d.Params(omega=1.0, sigma=0.02, beta=0.7, mrt_chi=0.8))
    assert not jnp.allclose(base.fR, chi.fR)  # chi changes the non-shear relaxation
