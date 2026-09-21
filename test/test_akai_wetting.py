"""Validation tests for the closed-form wetting BC (Akai et al., Adv. Water Resour. 116 (2018)).

The reorientation hits the imposed angle exactly in one application, preserves |F|, picks the
contact-line side consistently, is inert where no interface/wall exists, and is the default
``Params.wetting``; a 3D sessile droplet realizes the imposed angle; an unknown option fails loudly.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bob import color3d, d3q19, misc


def _angle_deg(v, n):
    c = float((v * n).sum() / (np.linalg.norm(v) * np.linalg.norm(n)))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def _akai_one(start_deg, target_deg, nw=(0.0, -1.0)):
    # nc at `start_deg` from the wall normal nw (2D), |F| = 2.5; reorient to target.
    nw = np.asarray(nw)
    perp = np.array([1.0, 0.0])  # wall-tangent axis
    nc = np.cos(np.radians(start_deg)) * nw + np.sin(np.radians(start_deg)) * perp
    F = jnp.asarray(2.5 * nc)[:, None, None]
    nwf = jnp.asarray(nw)[:, None, None]
    out = np.asarray(color3d._reorient_akai(F, nwf, theta=float(target_deg)))[:, 0, 0]
    return out, nw


def test_reorient_akai_hits_angle_exactly():
    # closed form is exact in one shot (unlike the truncated secant iteration), for any start.
    for start, target in ((100.0, 60.0), (40.0, 120.0), (95.0, 90.0), (10.0, 75.0)):
        out, nw = _akai_one(start, target)
        assert abs(np.linalg.norm(out) - 2.5) < 1e-9, "norm preserved"
        assert abs(_angle_deg(out, nw) - target) < 1e-6, f"{start}->{target}: got {_angle_deg(out, nw):.3f}"


def test_reorient_akai_picks_nearer_branch():
    # starting near the wall-tangent (start~90), target 60 must rotate toward nc's
    # side (stay on the same tangential half-plane), not flip across the normal.
    out, _nw = _akai_one(85.0, 60.0)
    assert out[0] > 0, "must keep nc's tangential sign (nearer branch)"


def test_reorient_akai_degenerate_parallel_interface_unchanged():
    # nc parallel to nw (interface parallel to wall, sin(theta')->0): leave F unchanged, no NaN.
    nw = jnp.asarray([0.0, -1.0])[:, None, None]
    F = jnp.asarray([0.0, -2.5])[:, None, None]  # nc == nw exactly
    out = np.asarray(color3d._reorient_akai(F, nw, theta=60.0))
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out[:, 0, 0], [0.0, -2.5], atol=1e-9)


def test_reorient_akai_inactive_without_interface_or_wall():
    z = jnp.zeros((2, 2, 2))
    nwf = jnp.zeros((2, 2, 2)).at[1].set(-1.0)
    np.testing.assert_allclose(np.asarray(color3d._reorient_akai(z, nwf, 60.0)), 0.0, atol=1e-12)
    F = jnp.ones((2, 2, 2))
    np.testing.assert_allclose(  # nw = 0 (not X_W): F untouched
        np.asarray(color3d._reorient_akai(F, jnp.zeros((2, 2, 2)), 60.0)), np.asarray(F), atol=1e-12
    )


def test_params_wetting_default_is_akai():
    # default is akai (closed-form rotation) as part of the default solver stack
    # csf + akai + Eq.34/35 + recolor_emag + wall_grad="fluid"; leclaire stays
    # selectable via wetting="leclaire".
    assert color3d.Params(omega=1.0, sigma=0.02, beta=0.7).wetting == "akai"


def _sessile3d_angle(theta, wetting, nz=26, n=40, R=9, steps=3500):
    solid = jnp.zeros((nz, n, n), bool).at[0].set(True)
    zz, yy, xx = jnp.meshgrid(jnp.arange(nz), jnp.arange(n), jnp.arange(n), indexing="ij")
    red = ((xx - (n - 1) / 2.0) ** 2 + (yy - (n - 1) / 2.0) ** 2 + (zz - 1.0) ** 2 <= R**2) & (zz >= 1)
    rhoR = jnp.where(red, 1.0, 0.0)
    W = d3q19.W[:, None, None, None]
    st = color3d.State(W * rhoR[None], W * (1.0 - rhoR)[None])
    nw = color3d.wall_normals(np.asarray(solid))
    p = color3d.Params(omega=1.0, sigma=0.02, beta=0.7, theta=theta, wetting=wetting)
    run = jax.jit(
        lambda s: jax.lax.scan(lambda a, _: (color3d.step(a, p, solid=solid, nw=nw), None), s, None, length=steps)[0]
    )
    return misc.contact_angle(run(st), wall_plane=0)


def test_akai_3d_imposed_theta_is_measured():
    for theta, tol in ((60.0, 16.0), (90.0, 12.0), (120.0, 16.0)):
        got = _sessile3d_angle(theta, "akai")
        assert abs(got - theta) < tol, f"akai 3D imposed {theta}, measured {got:.1f}"


def test_unknown_wetting_raises():
    # a typo'd wetting must fail loudly, not silently run the wrong BC
    solid = jnp.zeros((4, 4, 4), bool).at[0].set(True)
    nw = color3d.wall_normals(np.asarray(solid))
    st = misc.init_sphere(4, 4, 4, 1)
    p = color3d.Params(omega=1.0, sigma=0.02, beta=0.7, theta=90.0, wetting="bogus")
    with pytest.raises(ValueError, match="unknown Params.wetting"):
        color3d.step(st, p, solid=solid, nw=nw)
