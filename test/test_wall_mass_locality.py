"""Mass locality at solid walls: solid nodes must be inert.

The collide/perturb/recolor operators run on the whole grid; bounce-back is
meant to keep solid nodes inert. In a color-gradient model that is not automatic:
if wall nodes are allowed to recolor, the never-streamed rest population turns
each wall into a slow one-way sink, and a sessile droplet's red mass leaks into
the wall — up to 21% of the droplet over 4000 steps in 3D. The invariants are
independent of the wetting BC; the tests run under the geometric wetting BC
(theta + wall normals).

The solver therefore keeps solid nodes truly inert: they only store and bounce
back, never collide/perturb/recolor. Two invariants pin it:
  1. wall populations are frozen across a step (the decisive, dimension- and
     magnitude-independent check) — it fails if collision is allowed to relax
     the wall's rest population;
  2. the macroscopic symptom — red trapped in walls stays bounded instead of
     climbing without bound.

Run:  uv run pytest test/test_wall_mass_locality.py -v
"""

import jax
import jax.numpy as jnp
import numpy as np

from bob import color3d, d3q19


def _sessile3d(nz, n, R, theta, beta=0.7, sigma=0.02, omega=1.0):
    solid = jnp.zeros((nz, n, n), bool).at[0].set(True)
    zz, yy, xx = jnp.meshgrid(jnp.arange(nz), jnp.arange(n), jnp.arange(n), indexing="ij")
    red = ((xx - (n - 1) / 2.0) ** 2 + (yy - (n - 1) / 2.0) ** 2 + (zz - 1.0) ** 2 <= R**2) & (zz >= 1)
    rhoR = jnp.where(red, 1.0, 0.0)
    W = d3q19.W[:, None, None, None]
    st = color3d.State(W * rhoR[None], W * (1.0 - rhoR)[None])
    nw = color3d.wall_normals(np.asarray(solid))
    p = color3d.Params(omega=omega, sigma=sigma, beta=beta, theta=theta)
    return st, solid, nw, p


# --- invariant 1: the rest population at walls is inert ----------------------- #
# (The 18 moving links at a wall legitimately change each step — that is the
# bounce-back reflection of incoming fluid. Only the never-streamed rest
# population i=0 must be frozen; a wall that collides would relax it.)
def test_3d_wall_rest_population_frozen_across_a_step():
    st, solid, nw, p = _sessile3d(nz=20, n=32, R=7, theta=110.0)
    out = color3d.step(st, p, solid=solid, nw=nw)
    snp = np.asarray(solid)
    dR = np.abs(np.asarray(out.fR[0])[snp] - np.asarray(st.fR[0])[snp]).max()
    dB = np.abs(np.asarray(out.fB[0])[snp] - np.asarray(st.fB[0])[snp]).max()
    assert dR < 1e-12 and dB < 1e-12, f"wall rest population not inert: max change fR0={dR:.2e}, fB0={dB:.2e}"


# --- invariant 2: no leak into walls (mass conserved + bounded trapping) ------- #
def test_3d_sessile_red_does_not_leak_into_walls():
    st, solid, nw, p = _sessile3d(nz=20, n=32, R=7, theta=110.0)
    run = jax.jit(
        lambda s: jax.lax.scan(lambda a, _: (color3d.step(a, p, solid=solid, nw=nw), None), s, None, length=500)[0]
    )
    snp = np.asarray(solid)
    r0 = np.asarray(color3d.densities(st)[0])
    total0 = r0.sum()
    fracs = [r0[snp].sum() / total0]
    for _ in range(8):  # 4000 steps
        st = run(st)
        r = np.asarray(color3d.densities(st)[0])
        assert abs(r.sum() - total0) < 1e-6 * total0, "red mass not conserved"
        fracs.append(r[snp].sum() / total0)
    # With recoloring walls this climbs monotonically through ~0.21 and keeps rising;
    # the inert wall plateaus near ~0.03. Cap well between the two.
    assert fracs[-1] < 0.06, f"red trapped in walls too high: {fracs[-1]:.3f} (series {np.round(fracs, 3)})"


def test_inert_wall_blue_domain_stays_blue():
    # No red anywhere: wall nodes must not manufacture color from nothing.
    nz, n = 16, 24
    solid = jnp.zeros((nz, n, n), bool).at[0].set(True)
    W = d3q19.W[:, None, None, None]
    st = color3d.State(jnp.zeros((19, nz, n, n)), W * jnp.ones((nz, n, n))[None])
    nw = color3d.wall_normals(np.asarray(solid))
    p = color3d.Params(omega=1.0, sigma=0.02, beta=0.7, theta=110.0)
    run = jax.jit(
        lambda s: jax.lax.scan(lambda a, _: (color3d.step(a, p, solid=solid, nw=nw), None), s, None, length=500)[0]
    )
    rhoR = np.asarray(color3d.densities(run(st))[0])
    assert abs(rhoR).max() < 1e-9, f"red nucleated from nothing: max |rhoR| = {abs(rhoR).max():.2e}"
