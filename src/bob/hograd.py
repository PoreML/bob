"""High-order isotropic discrete gradient operators (Leclaire et al. 2014).

Leclaire, El-Hachem, Trepanier, Reggio, *High Order Spatial Generalization of 2D
and 3D Isotropic Discrete Gradient Operators with Fast Evaluation on GPUs*,
J. Sci. Comput. 59 (2014).

A gradient of a scalar field on a regular lattice is approximated as a weighted
sum over neighbours::

    d F / d x_a  ~=  sum_v  w(|v|) * v_a * F(x + v)          (lattice spacing h = 1)

with radially symmetric weights ``w`` (depending only on the sorted magnitudes of
the offset ``v``) chosen so the discretization is spatially accurate to order
``S`` and *isotropic* to order ``I >= S+2``. The **isotropy** order controls the
directional error of the gradient — which for the colour-gradient LBM is what
drives the parasitic (spurious) currents at a static interface.

Two families are provided. All weights are exact rationals derived directly from
the moment conditions (normalization, spatial order ``S``, isotropy order ``I``)
and checked in ``test/test_hograd.py``:

* ``(S, S+2)`` — raises the **spatial** order too (wider stencil, radius S/2).
  Best for smooth fields. On a *thin* LBM colour interface the wide stencil rings
  and can *increase* spurious currents.
* ``(2, I)`` — the Sbragaglia (2007) high-**isotropy** family at fixed 2nd spatial
  order: ``(2,4) (2,6) (2,8) (2,10)``. This is the family that reduces the
  interface spurious currents, and ``(2,8)`` reproduces the classic Sbragaglia E8
  weights exactly.

**Alignment with bob.** Order **(2,4)** is exactly the D3Q19 lattice-weight
gradient already used in ``bob.color3d`` (``W_i/cs2``): 2D
``w(1,0)=1/3, w(1,1)=1/12``; 3D ``w(1,0,0)=1/6, w(1,1,0)=1/12``. So this module is
a drop-in generalization — ``gradient(field, order=(2,4))`` reproduces
``color3d.color_gradient`` (``test/test_hograd.py``); higher orders only
widen the stencil.

Conventions match the rest of bob: a 2D field is ``(ny, nx)`` with spatial axes
``(y, x)``; a 3D field is ``(nz, ny, nx)`` with axes ``(z, y, x)``. The returned
gradient has a leading component axis ordered ``(d/dx, d/dy[, d/dz])`` like the
velocity arrays. Everything downstream of the (static, import-time) stencil is
pure JAX, so ``gradient`` composes under ``jax.jit`` / ``jax.lax.scan``.
"""

from fractions import Fraction as _Fr
from itertools import permutations, product

import jax.numpy as jnp
import numpy as np

# ─── Verified stencil weights, keyed by (spatial order S, isotropic order I) ───
# Each entry maps a representative ``(|v| sorted descending)`` to its weight.
# Zero-weight shells are simply omitted. Derived exactly (moment conditions);
# see test/test_hograd.py::test_weights_satisfy_moment_conditions.

_WEIGHTS_2D = {
    # (S, S+2) family — higher spatial order
    (2, 4): {(1, 0): _Fr(1, 3), (1, 1): _Fr(1, 12)},
    (4, 6): {(1, 0): _Fr(13, 30), (1, 1): _Fr(2, 15), (2, 0): _Fr(-1, 120), (2, 1): _Fr(-1, 60)},
    (6, 8): {
        (1, 0): _Fr(31, 70), (1, 1): _Fr(27, 140), (2, 1): _Fr(-3, 70),
        (2, 2): _Fr(3, 560), (3, 0): _Fr(-1, 630), (3, 1): _Fr(1, 280),
    },
    (8, 10): {
        (1, 0): _Fr(27, 70), (1, 1): _Fr(88, 315), (2, 0): _Fr(1, 30), (2, 1): _Fr(-53, 630),
        (2, 2): _Fr(2, 105), (3, 0): _Fr(-1, 105), (3, 1): _Fr(4, 315), (3, 2): _Fr(-1, 630),
        (4, 0): _Fr(1, 1440), (4, 1): _Fr(-1, 1260),
    },
    # S = 2 high-isotropy family (Sbragaglia) — recommended for spurious currents
    (2, 6): {(1, 0): _Fr(4, 15), (1, 1): _Fr(1, 10), (2, 0): _Fr(1, 120)},
    (2, 8): {
        (1, 0): _Fr(4, 21), (1, 1): _Fr(4, 45), (2, 0): _Fr(1, 60),
        (2, 1): _Fr(2, 315), (2, 2): _Fr(1, 5040),
    },
    (2, 10): {
        (1, 0): _Fr(262, 1785), (1, 1): _Fr(93, 1190), (2, 0): _Fr(7, 340), (2, 1): _Fr(6, 595),
        (2, 2): _Fr(9, 9520), (3, 0): _Fr(2, 5355), (3, 1): _Fr(1, 7140),
    },
}

_WEIGHTS_3D = {
    (2, 4): {(1, 0, 0): _Fr(1, 6), (1, 1, 0): _Fr(1, 12)},
    (4, 6): {
        (1, 0, 0): _Fr(1, 15), (1, 1, 0): _Fr(1, 5), (1, 1, 1): _Fr(-1, 30),
        (2, 0, 0): _Fr(1, 40), (2, 1, 0): _Fr(-1, 60),
    },
    (2, 6): {
        (1, 0, 0): _Fr(2, 15), (1, 1, 0): _Fr(1, 15), (1, 1, 1): _Fr(1, 60), (2, 0, 0): _Fr(1, 120),
    },
    (2, 8): {
        (1, 0, 0): _Fr(4, 45), (1, 1, 0): _Fr(1, 21), (1, 1, 1): _Fr(2, 105), (2, 0, 0): _Fr(5, 504),
        (2, 1, 0): _Fr(1, 315), (2, 1, 1): _Fr(1, 630), (2, 2, 0): _Fr(1, 5040),
    },
}

AVAILABLE_ORDERS = {2: sorted(_WEIGHTS_2D), 3: sorted(_WEIGHTS_3D)}


def _weight_table(ndim):
    return _WEIGHTS_2D if ndim == 2 else _WEIGHTS_3D


def _variants(rep):
    """All distinct integer offsets v (in (x, y[, z]) order) whose sorted
    magnitudes equal the representative ``rep`` — every axis permutation and sign
    flip of a non-zero component."""
    out = set()
    for perm in set(permutations(rep)):
        nz = [k for k, val in enumerate(perm) if val != 0]
        for signs in product((1, -1), repeat=len(nz)):
            v = list(perm)
            for k, s in zip(nz, signs):
                v[k] *= s
            out.add(tuple(v))
    return sorted(out)


def _build_stencil(order, ndim):
    """(roll_shifts, coeffs): ``roll_shifts`` is a tuple of static-int shift
    tuples (array-axis order) and ``coeffs`` is the ``(n_offset, ndim)`` array
    whose column ``a`` is ``w(|v|) * v_a``. Offsets are enumerated straight from
    the (non-zero) weight representatives, so any (S, I) family works and the roll
    count tracks the true stencil sparsity."""
    if order not in _weight_table(ndim):
        raise ValueError(f"no {ndim}D weights for order {order}; available: {AVAILABLE_ORDERS[ndim]}")
    weights = _weight_table(ndim)[order]
    shifts, coeffs = [], []
    for rep, w in weights.items():
        if w == 0:
            continue
        wf = float(w)
        for v in _variants(rep):
            # roll shift that brings F(x + v) onto the current cell: negate v and
            # reverse to array-axis order (y, x) in 2D, (z, y, x) in 3D.
            shifts.append(tuple(-c for c in reversed(v)))
            coeffs.append([wf * c for c in v])
    # NumPy (host) constant — NOT jnp: a jnp array built during one jit trace and
    # cached here would trip JAX's cross-trace leak check when reused in another
    # trace (e.g. the CSF vs perturbation steps). jnp treats the np array as a
    # fresh constant per trace.
    return tuple(shifts), np.asarray(coeffs, dtype=np.float64)  # (n_offset, ndim)


# Cache the (static shifts, coeffs) per (order, ndim) — built once, reused inside jit.
_STENCILS = {}


def _stencil(order, ndim):
    key = (order, ndim)
    if key not in _STENCILS:
        _STENCILS[key] = _build_stencil(order, ndim)
    return _STENCILS[key]


def _apply(field, order, ndim):
    """Bulk high-order gradient of ``field`` (no boundary handling)."""
    shifts, coeffs = _stencil(order, ndim)
    axes = tuple(range(ndim))
    neigh = jnp.stack([jnp.roll(field, shift=s, axis=axes) for s in shifts])  # (n, ...)
    return jnp.tensordot(coeffs, neigh, axes=([0], [0]))  # (ndim, ...)


def radius(order, ndim):
    """Chebyshev radius (max |offset component|) of the ``(S, I)`` stencil — the
    per-application ghost-plane depth ``gradient`` needs (1 for the default (2,4),
    2 for (2,6)/(2,8)); used by ``bob.multigpu.required_halo``."""
    shifts, _ = _stencil(order, ndim)
    return max(max(abs(c) for c in s) for s in shifts)


def gradient(field, order=(2, 4), solid=None):
    """High-order isotropic gradient of a scalar ``field``.

    ``field`` is ``(ny, nx)`` (2D) or ``(nz, ny, nx)`` (3D); ``ndim`` is inferred.
    Returns ``(ndim, *field.shape)`` ordered ``(d/dx, d/dy[, d/dz])``.

    ``order`` is ``(S, I)`` — one of ``AVAILABLE_ORDERS[ndim]``. ``(2, 4)`` is the
    lattice-weight gradient already in ``bob.color3d``.

    ``solid`` (bool mask, ``True = solid``, same shape as ``field``): where given,
    the wide high-order stencil would read solid nodes near a wall, so any cell
    whose stencil touches a solid — and the solid cells themselves — fall back to
    the a-priori low-order **(2,4)** fluid gradient (solid neighbour replaced by
    the centre value, so walls contribute no fake interface; Leclaire 2017). This
    generalizes the paper's "standard 1D gradient at X_W" to the reach of the wide
    stencil. Bulk cells keep the full high-order accuracy."""
    ndim = field.ndim
    hi = _apply(field, order, ndim)
    if order == (2, 4):
        return hi if solid is None else _fluid_gradient(field, solid, ndim)
    if solid is None:
        return hi
    lo = _fluid_gradient(field, solid, ndim)
    shifts, _ = _stencil(order, ndim)
    axes = tuple(range(ndim))
    solid_f = solid.astype(field.dtype)
    touches = solid | (sum(jnp.roll(solid_f, shift=s, axis=axes) for s in shifts) > 0.5)
    return jnp.where(touches[None], lo, hi)


def _fluid_gradient(field, solid, ndim):
    """A-priori low-order (2,4) gradient using bulk-fluid info only: solid-neighbour
    values are replaced by the centre cell (Leclaire 2017). Matches
    ``color3d._fluid_gradient``."""
    shifts, coeffs = _stencil((2, 4), ndim)
    axes = tuple(range(ndim))
    if solid is None:
        neigh = jnp.stack([jnp.roll(field, shift=s, axis=axes) for s in shifts])
    else:
        neigh = jnp.stack(
            [jnp.where(jnp.roll(solid, shift=s, axis=axes), field, jnp.roll(field, shift=s, axis=axes)) for s in shifts]
        )
    return jnp.tensordot(coeffs, neigh, axes=([0], [0]))
