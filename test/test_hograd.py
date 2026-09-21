"""Validation tests for the high-order isotropic gradient operator (bob.hograd).

1. the (2,4) order reproduces the existing color3d gradient exactly;
2. the tabulated weights satisfy Leclaire's moment conditions (normalization +
   spatial order S + isotropy order I) and match the paper's stated values;
3. the operator actually achieves its advertised spatial order of accuracy on a
   smooth field, higher orders converging faster;
4. the near-solid boundary fallback keeps the wide stencil off solid nodes.
"""

from fractions import Fraction as Fr

import jax.numpy as jnp
import numpy as np
import pytest

from bob import color3d, hograd

# ── 1. (2,4) == the lattice-weight gradient already in the solver ──────────────

def test_order24_matches_color3d_gradient_3d():
    rng = np.random.default_rng(1)
    rhoN = jnp.asarray(rng.standard_normal((10, 12, 14)))
    ref = color3d.color_gradient(rhoN)
    got = hograd.gradient(rhoN, order=(2, 4))
    assert jnp.allclose(got, ref, atol=1e-12)


def test_order24_fluid_gradient_matches_color3d():
    rng = np.random.default_rng(2)
    rhoN = jnp.asarray(rng.standard_normal((10, 12, 14)))
    solid = jnp.asarray(rng.random((10, 12, 14)) < 0.2)
    ref = color3d._fluid_gradient(rhoN, solid)
    got = hograd.gradient(rhoN, order=(2, 4), solid=solid)
    assert jnp.allclose(got, ref, atol=1e-12)


# ── 2. weights satisfy the exact moment conditions ─────────────────────────────

def _dfact(n):
    r = 1
    while n > 1:
        r *= n
        n -= 2
    return r


def _iso(e):
    r = 1
    for ek in e:
        r *= _dfact(ek - 1)
    return r


def _even_exp_reps(ndim, degree):
    from itertools import product
    reps = set()
    for e in product(range(0, degree + 1, 2), repeat=ndim):
        if sum(e) == degree:
            reps.add(tuple(sorted(e, reverse=True)))
    return sorted(reps, reverse=True)


def _moment(weights, e):
    """Exact moment M(e) = sum_v w(|v|) * prod v_k^{e_k} over the actual stencil
    offsets (variants of every representative), independent of any R/D box."""
    tot = Fr(0)
    for rep, w in weights.items():
        if w == 0:
            continue
        for v in hograd._variants(rep):
            term = 1
            for vk, ek in zip(v, e):
                term *= vk ** ek
            tot += Fr(w) * term
    return tot


@pytest.mark.parametrize("ndim", [2, 3])
def test_weights_satisfy_moment_conditions(ndim):
    table = hograd._WEIGHTS_2D if ndim == 2 else hograd._WEIGHTS_3D
    for (S, Iord), weights in table.items():
        # normalization  M(2,0,..) = 1
        e0 = tuple([2] + [0] * (ndim - 1))
        assert _moment(weights, e0) == 1, f"{ndim}D {(S, Iord)} norm"
        # spatial order: all-even moments of degree 4..S vanish
        for d in range(4, S + 1, 2):
            for e in _even_exp_reps(ndim, d):
                assert _moment(weights, e) == 0, f"{ndim}D {(S, Iord)} spatial M{e}"
        # isotropy: degree S+2..I moments proportional to the isotropic tensor
        for d in range(S + 2, Iord + 1, 2):
            ratios = {_moment(weights, e) / Fr(_iso(e)) for e in _even_exp_reps(ndim, d)}
            assert len(ratios) == 1, f"{ndim}D {(S, Iord)} isotropy deg {d}: {ratios}"


def test_weights_match_paper_prose():
    # values stated in Leclaire 2014 prose (Sec. 2.2) — the HTML table is corrupt.
    assert hograd._WEIGHTS_2D[(4, 6)] == {(1, 0): Fr(13, 30), (1, 1): Fr(2, 15),
                                          (2, 0): Fr(-1, 120), (2, 1): Fr(-1, 60)}
    assert hograd._WEIGHTS_3D[(4, 6)] == {(1, 0, 0): Fr(1, 15), (1, 1, 0): Fr(1, 5),
                                          (1, 1, 1): Fr(-1, 30), (2, 0, 0): Fr(1, 40),
                                          (2, 1, 0): Fr(-1, 60)}


# ── 3. numerical spatial order of accuracy on a smooth periodic field ──────────

def _order_2d(order, sizes=(32, 48, 64, 96)):
    errs, hs = [], []
    p, q = 3.0, 2.0  # integer wavenumbers -> exact grid periodicity
    for N in sizes:
        h = 2.0 * np.pi / N
        x = np.arange(N) * h
        X, Y = np.meshgrid(x, x, indexing="xy")  # axes (y, x)
        F = np.sin(p * X) * np.cos(q * Y)
        dFdx = p * np.cos(p * X) * np.cos(q * Y)
        g = np.asarray(hograd.gradient(jnp.asarray(F), order=order))[0] / h
        errs.append(np.max(np.abs(g - dFdx)))
        hs.append(h)
    slope = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    return slope


@pytest.mark.parametrize("order,expected", [((2, 4), 2), ((4, 6), 4), ((6, 8), 6)])
def test_spatial_order_of_accuracy_2d(order, expected):
    slope = _order_2d(order)
    assert abs(slope - expected) < 0.6, f"order {order}: measured slope {slope:.2f}, expected ~{expected}"


def test_higher_order_is_more_accurate_2d():
    # at fixed resolution, error must drop monotonically as the order rises
    N = 64
    h = 2.0 * np.pi / N
    x = np.arange(N) * h
    X, Y = np.meshgrid(x, x, indexing="xy")
    F = np.sin(3 * X) * np.cos(2 * Y)
    dFdx = 3 * np.cos(3 * X) * np.cos(2 * Y)
    prev = np.inf
    for order in [(2, 4), (4, 6), (6, 8), (8, 10)]:
        g = np.asarray(hograd.gradient(jnp.asarray(F), order=order))[0] / h
        err = np.max(np.abs(g - dFdx))
        assert err < prev, f"order {order} err {err:.2e} not < {prev:.2e}"
        prev = err


def test_spatial_order_of_accuracy_3d():
    p, q, r = 2.0, 1.0, 1.0
    for order, expected in [((2, 4), 2), ((4, 6), 4)]:
        errs, hs = [], []
        for N in (24, 32, 48):
            h = 2.0 * np.pi / N
            x = np.arange(N) * h
            Z, Y, X = np.meshgrid(x, x, x, indexing="ij")  # axes (z, y, x)
            F = np.sin(p * X) * np.cos(q * Y) * np.cos(r * Z)
            dFdx = p * np.cos(p * X) * np.cos(q * Y) * np.cos(r * Z)
            g = np.asarray(hograd.gradient(jnp.asarray(F), order=order))[0] / h
            errs.append(np.max(np.abs(g - dFdx)))
            hs.append(h)
        slope = np.polyfit(np.log(hs), np.log(errs), 1)[0]
        assert abs(slope - expected) < 0.7, f"3D order {order}: slope {slope:.2f}"


# ── 4. boundary fallback keeps the wide stencil off solids ─────────────────────

def test_boundary_fallback_is_finite_and_bulk_matches():
    rng = np.random.default_rng(3)
    rhoN = jnp.asarray(rng.standard_normal((40, 40)))
    solid = np.zeros((40, 40), dtype=bool)
    solid[:5, :] = True  # a wall slab
    solid = jnp.asarray(solid)
    g = hograd.gradient(rhoN, order=(6, 8), solid=solid)
    assert jnp.all(jnp.isfinite(g))
    bulk = hograd.gradient(rhoN, order=(6, 8))
    # Rows truly clear of the wall AND its periodic image (wall rows 0-4, R=3, N=40):
    # contaminated rows are 0..7 and 37..39, so 12..34 is genuine bulk and must match.
    assert jnp.allclose(g[:, 12:34, :], bulk[:, 12:34, :], atol=1e-12)
    # and the wall-adjacent fluid row genuinely used the fallback (differs from bulk)
    assert not jnp.allclose(g[:, 5, :], bulk[:, 5, :], atol=1e-9)
