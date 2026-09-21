"""Core validation of the 3D color-gradient solver (D3Q19) against classical benchmarks.

Ordered from lattice constants up to two-phase physics:

  1. D3Q19 lattice sanity (incl. 4th-moment isotropy the perturbation needs)
  2. single-phase core: moments, collision/stream conservation, bounce-back
  3. square duct vs the analytic series (the genuinely-3D Poiseuille)
  4. an elongated red box rounds into a ball
  5. Laplace law dp = 2*sigma/R across radii + bounded spurious currents
  6. the contact-angle metric on a synthetic hemisphere (a 90-degree cap)

Run:  uv run pytest test/test_solver_core.py -v
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bob import color3d, d3q19, lbm3d, misc

# Baseline two-phase parameters shared by the tests below.
PARAMS = color3d.Params(omega=1.0, sigma=0.02, beta=0.7)


# --- 1. D3Q19 lattice sanity ------------------------------------------------ #
def test_shapes():
    assert d3q19.C.shape == (19, 3)
    assert d3q19.W.shape == (19,)
    assert d3q19.OPP.shape == (19,)


def test_weights_sum_to_one():
    assert abs(float(d3q19.W.sum()) - 1.0) < 1e-12


def test_velocity_set_is_rest_plus_axial_plus_face_diagonals():
    mag2 = np.sort(np.asarray((d3q19.C**2).sum(axis=1)))
    np.testing.assert_array_equal(mag2, np.array([0] + [1] * 6 + [2] * 12))


def test_opposite_directions():
    # C[OPP[i]] == -C[i]: every direction has a true reverse.
    np.testing.assert_array_equal(np.asarray(d3q19.C[d3q19.OPP]), -np.asarray(d3q19.C))


def test_lattice_isotropy():
    c, w = d3q19.C.astype(jnp.float64), d3q19.W
    # First moment vanishes; second moment is isotropic: sum_i w_i c_i c_i = cs2 I.
    np.testing.assert_allclose(np.asarray((w[:, None] * c).sum(0)), np.zeros(3), atol=1e-12)
    second = jnp.einsum("i,ia,ib->ab", w, c, c)
    np.testing.assert_allclose(np.asarray(second), d3q19.CS2 * np.eye(3), atol=1e-12)


def test_fourth_moment_isotropy():
    # sum_i w_i c_ia c_ib c_ic c_id = cs2^2 (d_ab d_cd + d_ac d_bd + d_ad d_bc).
    # This is why the D2Q9 (9/4)(sigma/tau) perturbation prefactor carries over to D3Q19.
    c, w = np.asarray(d3q19.C, dtype=float), np.asarray(d3q19.W)
    fourth = np.einsum("i,ia,ib,ic,id->abcd", w, c, c, c, c)
    d = np.eye(3)
    expect = d3q19.CS2**2 * (np.einsum("ab,cd->abcd", d, d) + np.einsum("ac,bd->abcd", d, d) + np.einsum("ad,bc->abcd", d, d))
    np.testing.assert_allclose(fourth, expect, atol=1e-12)


# --- 2. single-phase core --------------------------------------------------- #
def _random_state3d(key, nz=4, ny=5, nx=6):
    """A mild density bump (rho ~ 1) with small velocities (|u| < 0.05)."""
    krho, ku = jax.random.split(key)
    rho = 1.0 + 0.1 * jax.random.uniform(krho, (nz, ny, nx))
    u = 0.05 * (jax.random.uniform(ku, (3, nz, ny, nx)) - 0.5)
    return rho, u


def test_equilibrium_zero_velocity():
    # At rest feq_i collapses to w_i * rho.
    feq = lbm3d.equilibrium(jnp.ones((3, 4, 5)), jnp.zeros((3, 3, 4, 5)))
    for i in range(19):
        np.testing.assert_allclose(np.asarray(feq[i]), float(d3q19.W[i]), atol=1e-12)


def test_equilibrium_recovers_moments():
    rho, u = _random_state3d(jax.random.PRNGKey(0))
    feq = lbm3d.equilibrium(rho, u)
    np.testing.assert_allclose(np.asarray(lbm3d.density(feq)), np.asarray(rho), atol=1e-10)
    np.testing.assert_allclose(np.asarray(lbm3d.velocity(feq, rho)), np.asarray(u), atol=1e-10)


def test_collision_conserves_mass_and_momentum():
    rho, u = _random_state3d(jax.random.PRNGKey(1))
    f = jnp.abs(lbm3d.equilibrium(rho, u) + 1e-3 * jax.random.normal(jax.random.PRNGKey(2), (19, *rho.shape)))
    rho0 = lbm3d.density(f)
    mom0 = lbm3d.velocity(f, rho0) * rho0
    fpost = lbm3d.collide(f, omega=1.0)  # force-free BGK is collision-invariant
    rho1 = lbm3d.density(fpost)
    mom1 = lbm3d.velocity(fpost, rho1) * rho1
    np.testing.assert_allclose(np.asarray(rho1), np.asarray(rho0), atol=1e-10)
    np.testing.assert_allclose(np.asarray(mom1), np.asarray(mom0), atol=1e-10)


def test_streaming_conserves_mass_and_shifts_packets():
    f = jax.random.uniform(jax.random.PRNGKey(3), (19, 4, 5, 6))
    np.testing.assert_allclose(float(lbm3d.stream(f).sum()), float(f.sum()), atol=1e-10)
    # A lone packet in direction i moves exactly one cell along c_i.
    for i in (1, 3, 5, 7, 11, 15):  # one axial per axis + one diagonal per plane
        cx, cy, cz = (int(v) for v in d3q19.C[i])
        g = jnp.zeros((19, 4, 5, 6)).at[i, 2, 2, 2].set(1.0)
        out = lbm3d.stream(g)
        assert float(out[i, 2 + cz, 2 + cy, 2 + cx]) == pytest.approx(1.0)
        assert float(out[i, 2, 2, 2]) == pytest.approx(0.0)


def test_bounce_back_reverses_at_solid():
    f = jax.random.uniform(jax.random.PRNGKey(4), (19, 3, 4, 4))
    solid = jnp.zeros((3, 4, 4), dtype=bool).at[1, 1, 1].set(True)
    out = lbm3d.bounce_back(f, solid)
    # Solid node: direction i now holds the old opposite-direction value.
    np.testing.assert_allclose(np.asarray(out[:, 1, 1, 1]), np.asarray(f[d3q19.OPP, 1, 1, 1]), atol=1e-12)
    # Fluid node: untouched.
    np.testing.assert_allclose(np.asarray(out[:, 0, 0, 0]), np.asarray(f[:, 0, 0, 0]), atol=1e-12)


# --- 3. square duct vs analytic series -------------------------------------- #
def _duct_analytic(y, z, a, G, mu, n_terms=99):
    """Series solution for body-force-driven laminar flow along x in a square
    duct |y| <= a, |z| <= a (e.g. White, Viscous Fluid Flow):

        u(y,z) = (16 G a^2 / mu pi^3) sum_{n odd} (-1)^((n-1)/2)
                 [1 - cosh(n pi z / 2a)/cosh(n pi / 2)] cos(n pi y / 2a) / n^3
    """
    u = np.zeros(np.broadcast(y, z).shape)
    for n in range(1, n_terms + 1, 2):
        k = n * np.pi / (2.0 * a)
        u += (-1) ** ((n - 1) // 2) * (1.0 - np.cosh(k * z) / np.cosh(n * np.pi / 2.0)) * np.cos(k * y) / n**3
    return 16.0 * G * a**2 / (mu * np.pi**3) * u


def _run_duct(n=22, nx=4, omega=1.0, gx=1e-5, nsteps=12_000):
    """Duct flow: solid border on the (z, y) cross-section, periodic along x,
    uniform body force gx. Returns the steady velocity field."""
    solid = jnp.zeros((n, n, nx), dtype=bool)
    solid = solid.at[0, :, :].set(True).at[-1, :, :].set(True)
    solid = solid.at[:, 0, :].set(True).at[:, -1, :].set(True)
    f = lbm3d.equilibrium(jnp.ones((n, n, nx)), jnp.zeros((3, n, n, nx)))
    force = jnp.array([gx, 0.0, 0.0])

    def body(f, _):
        f = lbm3d.collide(f, omega, force)
        f = lbm3d.stream(f)
        f = lbm3d.bounce_back(f, solid)
        return f, None

    f, _ = jax.lax.scan(body, f, None, length=nsteps)
    return lbm3d.velocity(f, lbm3d.density(f), force)


def test_square_duct_profile_matches_series():
    n, omega, gx = 22, 1.0, 1e-5
    u = _run_duct(n=n, omega=omega, gx=gx)
    ux = np.asarray(u[0, :, :, 2])  # one (z, y) cross-section; flow is x-invariant

    # Full-way bounce-back with Guo body forcing: the effective no-slip plane
    # is one full half-cell outside the last fluid node.  For n total nodes
    # (solid walls at 0 and n-1), the effective half-width that minimises the
    # L2 distance to the series solution is a = (n-1)/2 — not (n-2)/2.  The
    # (n-2)/2 convention applies to pressure-driven flow; with body forcing the
    # extra F/2 Guo correction shifts the apparent wall outward by half a node.
    nu = d3q19.CS2 * (1.0 / omega - 0.5)
    a = (n - 1) / 2.0
    coord = np.arange(n) - (n - 1) / 2.0
    zz, yy = np.meshgrid(coord, coord, indexing="ij")
    u_an = _duct_analytic(yy, zz, a, gx, nu)  # mu = rho*nu with rho ~ 1

    fluid = np.s_[1 : n - 1, 1 : n - 1]
    err = np.sqrt(np.mean((ux[fluid] - u_an[fluid]) ** 2)) / np.max(u_an[fluid])
    assert err < 0.05, f"duct profile L2 error {err:.3%}"
    # Peak lands in one of the four centre nodes (even fluid count -> 4-fold symmetry).
    iz, iy = np.unravel_index(np.asarray(ux).argmax(), ux.shape)
    assert iz in (n // 2 - 1, n // 2) and iy in (n // 2 - 1, n // 2)


def test_square_duct_flow_scales_linearly_with_force():
    u1 = _run_duct(n=18, gx=1e-5, nsteps=6_000)
    u2 = _run_duct(n=18, gx=2e-5, nsteps=6_000)
    ratio = float(jnp.max(u2[0]) / jnp.max(u1[0]))
    assert abs(ratio - 2.0) < 0.02, f"flow not linear in force: ratio {ratio:.3f}"


# --- 4. two-phase: a red box rounds into a ball ------------------------------ #
def test_perturbation_coefficients_conditions():
    # Mass conservation (sum B_i = cs2) and isotropic second moment
    # (sum_i B_i c_ia c_ib = cs2 I) — the two identities the D3Q19
    # Reis-Phillips set must share with the D2Q9 set for the same prefactor.
    assert abs(float(color3d.B.sum()) - d3q19.CS2) < 1e-12
    c = np.asarray(d3q19.C, dtype=float)
    second = np.einsum("i,ia,ib->ab", np.asarray(color3d.B), c, c)
    np.testing.assert_allclose(second, d3q19.CS2 * np.eye(3), atol=1e-12)


def test_init_sphere_unit_density_and_round():
    st = misc.init_sphere(32, 32, 32, R=9)
    rhoR, rhoB = color3d.densities(st)
    np.testing.assert_allclose(np.asarray(rhoR + rhoB), 1.0, atol=1e-12)
    assert misc.sphericity(np.asarray(rhoR)) < 1.05
    vol = float((rhoR > rhoB).sum())
    assert abs(vol - 4 / 3 * np.pi * 9**3) < 0.15 * (4 / 3 * np.pi * 9**3)


def test_step_conserves_mass_per_color():
    st = misc.init_sphere(16, 16, 16, R=4)
    mR0, mB0 = (float(m.sum()) for m in color3d.densities(st))
    out, _ = jax.lax.scan(lambda s, _: (color3d.step(s, PARAMS), None), st, None, length=50)
    mR1, mB1 = (float(m.sum()) for m in color3d.densities(out))
    assert abs(mR1 - mR0) / mR0 < 1e-10
    assert abs(mB1 - mB0) / mB0 < 1e-10


def test_box_rounds_into_ball():
    st = misc.init_box(32, 32, 32, d=8, h=10, w=16)  # elongated red box
    s0 = misc.sphericity(np.asarray(color3d.densities(st)[0]))  # ~1.47
    out, _ = jax.lax.scan(lambda s, _: (color3d.step(s, PARAMS), None), st, None, length=2500)
    rhoR, rhoB = color3d.densities(out)
    s1 = misc.sphericity(np.asarray(rhoR))
    assert s1 < 1.15, f"not round: sphericity {s0:.2f} -> {s1:.2f}"
    assert s1 < s0 - 0.3
    # Colors stay segregated: pure phases survive on both sides of the interface.
    rhoN = np.asarray(color3d.color_field(rhoR, rhoB))
    assert rhoN.max() > 0.99 and rhoN.min() < -0.99


# --- 5. Laplace law dp = 2*sigma/R ------------------------------------------ #
def _relaxed_sphere(R, L, steps=2500):
    st = misc.init_sphere(L, L, L, R=R)
    out, _ = jax.lax.scan(lambda s, _: (color3d.step(s, PARAMS), None), st, None, length=steps)
    return out


def test_laplace_law_2sigma_over_R():
    sigma = PARAMS.sigma
    radii = [8, 10, 12]
    dPs = []
    for R in radii:
        out = _relaxed_sphere(R, L=3 * R + 8)
        dPs.append(misc.pressure_jump(out, R))
    inv_R = 1.0 / np.array(radii, dtype=float)
    dPs = np.array(dPs)
    slope, intercept = np.polyfit(inv_R, dPs, 1)
    resid = dPs - (slope * inv_R + intercept)
    r2 = 1.0 - np.sum(resid**2) / np.sum((dPs - dPs.mean()) ** 2)
    assert r2 > 0.99, f"Laplace law not linear: R^2={r2:.4f}"
    sigma_eff = slope / 2.0  # 3D sphere has two curvatures: dp = 2*sigma/R
    assert 0.7 * sigma < sigma_eff < 1.4 * sigma, f"sigma_eff={sigma_eff:.4e} vs input {sigma:.4e}"


def test_surface_tension_independent_of_viscosity_ratio_3d():
    """Leclaire 2017 Eq. (18) A = (9/4) omega_eff sigma
    keeps the measured surface tension independent of the viscosity ratio."""
    sigma, R, L = 0.02, 9, 35

    def sigma_meas(omega2):
        st = misc.init_sphere(L, L, L, R=R)
        p = color3d.Params(omega=1.0, sigma=sigma, beta=0.7, omega2=omega2)
        out, _ = jax.lax.scan(lambda s, _: (color3d.step(s, p), None), st, None, length=2500)
        return misc.pressure_jump(out, R) * R / 2.0  # 3D sphere: dp = 2 sigma / R

    s_equal = sigma_meas(1.0)  # omega_eff == omega_red
    s_visc = sigma_meas(0.4)  # blue 5x more viscous -> omega_eff < omega_red
    rel = abs(s_visc - s_equal) / s_equal
    assert rel < 0.10, f"3D surface tension drifted {rel:.1%} with a 5x viscosity ratio"


def test_static_sphere_spurious_currents_bounded():
    out = _relaxed_sphere(R=8, L=32, steps=2500)
    assert float(jnp.abs(color3d.velocity(out)).max()) < 1e-2


# --- 6. 3D contact angle ----------------------------------------------------- #
def test_contact_angle_of_synthetic_hemisphere():
    # A half-ball centered on the wall plane is a 90-degree cap.
    nz, n, R = 24, 48, 12
    zz, yy, xx = np.indices((nz, n, n))
    red = ((xx - (n - 1) / 2.0) ** 2 + (yy - (n - 1) / 2.0) ** 2 + (zz - 1.0) ** 2 <= R**2) & (zz >= 1)
    rhoR = jnp.asarray(red.astype(float))
    st = color3d.State(
        d3q19.W[:, None, None, None] * rhoR[None],
        d3q19.W[:, None, None, None] * (1 - rhoR)[None],
    )
    assert abs(misc.contact_angle(st, wall_plane=0) - 90.0) < 10.0
    # Sessile-droplet wetting itself is validated by the geometric-BC tests in
    # test_wetting_bc.py (3D imposed theta == measured angle).
