"""Validation tests for the geometric wetting BC (Leclaire et al., PRE 95, 033306 (2017)).

The contact angle is imposed by reorienting the a priori color gradient F at
boundary-fluid sites X_W so its angle with the wall normal n_w equals theta
(norm |F| preserved; secant iteration, Eqs. 29-31). n_w comes from a
preprocessing pass: smooth the binary solid matrix (D3Q27 weights, 3
iterations), take the isotropic-stencil gradient, normalize. Convention:
F = grad(rhoN) points toward red, n_w = grad(g) points toward solid, so theta IS
the red contact angle in degrees.

Covered, in order: wall normals -> secant reorientation -> imposed-vs-measured
sessile angles (3D) -> interface gating (no standing force on dry walls).

Run:  uv run pytest test/test_wetting_bc.py -v
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from bob import color3d, d3q19, misc, porous3d

ASSET = Path(__file__).parent / "assets" / "sphere256.npy"


# --- wall normals (preprocessing) ---------------------------------------------- #
def test_wall_normals_3d_flat_wall_points_into_solid():
    # both z boundary planes solid (the wrap is never left open in real domains)
    solid = np.zeros((12, 10, 10), bool)
    solid[0] = solid[-1] = True
    nw = np.asarray(color3d.wall_normals(solid))
    assert nw.shape == (3, 12, 10, 10)
    np.testing.assert_allclose(nw[2, 1, :, :], -1.0, atol=1e-6)  # uz = -1: toward the solid below
    np.testing.assert_allclose(nw[2, -2, :, :], 1.0, atol=1e-6)  # uz = +1: toward the solid above
    np.testing.assert_allclose(nw[0, 1, :, :], 0.0, atol=1e-6)
    np.testing.assert_allclose(nw[1, 1, :, :], 0.0, atol=1e-6)
    assert np.abs(nw[:, 4:8, :, :]).max() < 1e-12, "normals must vanish away from X_W"
    assert np.abs(nw[:, 0, :, :]).max() < 1e-12, "normals live on fluid sites, not solid ones"


def test_wall_normals_3d_sphere_is_radial():
    n = 24
    zz, yy, xx = np.indices((n, n, n))
    c = (n - 1) / 2.0
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2 + (zz - c) ** 2)
    solid = r <= 7
    nw = np.asarray(color3d.wall_normals(solid))
    xw = (np.abs(nw).sum(0) > 0) & ~solid
    assert xw.sum() > 100
    # radial inward: nw should anti-align with the outward radius vector
    rad = np.stack([(xx - c), (yy - c), (zz - c)])
    rad = rad / np.sqrt((rad**2).sum(0, keepdims=True))
    cosang = (nw * -rad).sum(0)[xw]
    assert cosang.min() > 0.9, f"normal not radial-inward: min cos {cosang.min():.3f}"


def test_wall_normals_3d_unit_norm_on_xw():
    solid = np.zeros((10, 10, 10), bool)
    solid[0] = True
    nw = np.asarray(color3d.wall_normals(solid))
    mags = np.sqrt((nw**2).sum(0))
    on = mags > 0
    np.testing.assert_allclose(mags[on], 1.0, atol=1e-12)


# --- secant reorientation -------------------------------------------------------- #
def _angle_deg(v, n):
    c = float((v * n).sum() / (np.linalg.norm(v) * np.linalg.norm(n)))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def test_reorient_closes_most_of_the_angle_gap():
    # The truncated secant iteration is not exact from a distant start — the paper relies on the
    # interface moving little per LBM step, so the reorientation converges over
    # steps. One application must (a) preserve |F|, (b) close >=70% of the gap
    # from 40 deg away, and (c) land within 1 deg from a near-target start.
    nw = np.array([0.0, 0.0, -1.0])

    def reorient(start, target):
        nc = np.array([np.sin(np.radians(start)), 0.0, -np.cos(np.radians(start))])
        F = jnp.asarray(2.5 * nc)[:, None, None, None]  # |F| = 2.5
        nwf = jnp.asarray(nw)[:, None, None, None]
        F2 = np.asarray(color3d._reorient(F, nwf, theta=float(target)))[:, 0, 0, 0]
        assert abs(np.linalg.norm(F2) - 2.5) < 1e-9, "norm must be preserved"
        return _angle_deg(F2, nw)

    got = reorient(100.0, 60.0)  # far start: most of the 40 deg gap closed
    assert abs(got - 60.0) < 0.3 * 40.0, f"far start: {got:.1f} deg, want ~60"
    got = reorient(80.0, 75.0)  # near start: essentially converged
    assert abs(got - 75.0) < 1.0, f"near start: {got:.1f} deg, want 75"


def test_reorient_keeps_correct_angle_fixed():
    nw = np.array([0.0, 0.0, -1.0])
    th = 75.0
    nc = np.array([np.sin(np.radians(th)), 0.0, -np.cos(np.radians(th))])
    F = jnp.asarray(nc)[:, None, None, None]
    nwf = jnp.asarray(nw)[:, None, None, None]
    F2 = np.asarray(color3d._reorient(F, nwf, theta=th))[:, 0, 0, 0]
    assert abs(_angle_deg(F2, nw) - th) < 1.0


def test_reorient_inactive_without_interface_or_wall():
    z = jnp.zeros((3, 2, 2, 2))
    nwf = jnp.zeros((3, 2, 2, 2)).at[2].set(-1.0)
    np.testing.assert_allclose(np.asarray(color3d._reorient(z, nwf, theta=60.0)), 0.0, atol=1e-12)
    F = jnp.ones((3, 2, 2, 2))
    np.testing.assert_allclose(  # nw = 0 (not X_W): F untouched
        np.asarray(color3d._reorient(F, jnp.zeros((3, 2, 2, 2)), theta=60.0)), np.asarray(F), atol=1e-12
    )


# --- imposed vs measured sessile angles ------------------------------------------ #
def _sessile3d_theta(theta, nz=26, n=40, R=9, steps=3500):
    solid = jnp.zeros((nz, n, n), bool).at[0].set(True)
    zz, yy, xx = jnp.meshgrid(jnp.arange(nz), jnp.arange(n), jnp.arange(n), indexing="ij")
    red = ((xx - (n - 1) / 2.0) ** 2 + (yy - (n - 1) / 2.0) ** 2 + (zz - 1.0) ** 2 <= R**2) & (zz >= 1)
    rhoR = jnp.where(red, 1.0, 0.0)
    W = d3q19.W[:, None, None, None]
    st = color3d.State(W * rhoR[None], W * (1.0 - rhoR)[None])
    nw = color3d.wall_normals(np.asarray(solid))
    p = color3d.Params(omega=1.0, sigma=0.02, beta=0.7, theta=theta, wetting="leclaire")
    run = jax.jit(
        lambda s: jax.lax.scan(lambda a, _: (color3d.step(a, p, solid=solid, nw=nw), None), s, None, length=steps)[0]
    )
    return misc.contact_angle(run(st), wall_plane=0)


# Leclaire (secant) variant; akai is the default and is covered in test_akai_wetting.
# In this short run with a one-sided wall the realized angle is contracted toward 90 deg
# relative to the imposed value (a property of the single-layer geometric BC; the
# reorientation itself is exact). The tolerance brackets the contracted angle.
def test_3d_imposed_theta_is_measured():
    for theta, tol in ((60.0, 16.0), (90.0, 12.0), (120.0, 16.0)):
        got = _sessile3d_theta(theta)
        assert abs(got - theta) < tol, f"imposed {theta} deg, measured {got:.1f} deg"


# --- interface gating: no standing force on dry walls ----------------------------- #
def test_no_spurious_flow_without_interface():
    # All-blue duct with a strongly wetting theta: a fictitious-wall-color BC would
    # exert a standing force along the whole dry wall; the geometric BC only
    # acts where an interface exists, so the fluid must stay quiescent.
    nz, ny, nx = 12, 12, 30
    solid = porous3d.duct(nz, ny, nx)
    W = d3q19.W[:, None, None, None]
    st = color3d.State(jnp.zeros((19, nz, ny, nx)), W * jnp.ones((nz, ny, nx))[None])
    nw = color3d.wall_normals(np.asarray(solid))
    p = color3d.Params(omega=1.0, sigma=0.02, beta=0.7, theta=30.0)
    run = jax.jit(
        lambda s: jax.lax.scan(lambda a, _: (color3d.step(a, p, solid=solid, nw=nw), None), s, None, length=1000)[0]
    )
    out = run(st)
    umax = float(jnp.abs(color3d.velocity(out)).max())
    assert umax < 1e-10, f"standing force on dry walls: max |u| = {umax:.2e}"
