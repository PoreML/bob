"""Porous-media drainage: 3D pore geometry, the drainage driver, diagnostics.

Loaders/croppers return (nz, ny, nx) bool masks with the dataset convention
``True = solid, False = pore``; diagnostics are host-side NumPy; the JAX entry
points are ``drain`` / ``imbibe``.

Shared domain layout: x is the flow axis; z = 0 / nz-1 and y = 0 / ny-1 are
solid no-slip walls (wetting, ``theta > 90`` for drainage); x in [0, n_in)
is the inlet reservoir of pure red at velocity (u_in, 0, 0); x = nx-1 is the
zero-gradient unit-density outlet. The inlet/outlet slabs mask the periodic
streaming wrap.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from bob import bc, color3d, d3q19


# --- structure loading -------------------------------------------------------- #
def load_structure(path):
    """Load a binary geometry volume: 3D bool, ``True = solid``."""
    vol = np.load(path)
    if vol.dtype != np.bool_ or vol.ndim != 3:
        raise ValueError(f"expected a 3D bool volume, got {vol.dtype} array of shape {vol.shape}")
    return vol


# The canonical 64^3 crop of the geometry-dataset sample sphere_0000
# (test/assets/sphere256.npy) used by the porous-drainage tests: porosity 0.303,
# pore space x-spanning. Kept as a slice of the committed full volume instead of
# a second .npy so the repo stays single-source.
SPHERE64_SLICE = (slice(45, 109), slice(131, 195), slice(122, 186))


def sphere64(asset):
    """The canonical 64^3 porous-drainage crop, cut from the full 256^3 volume at ``asset``."""
    return load_structure(asset)[SPHERE64_SLICE].copy()


def _flood_fill(open_mask, seeds):
    """Cells of ``open_mask`` reachable from ``seeds`` by 6-connected steps (no wrap)."""
    reach = open_mask & seeds
    while True:
        grown = reach.copy()
        grown[1:, :, :] |= reach[:-1, :, :]
        grown[:-1, :, :] |= reach[1:, :, :]
        grown[:, 1:, :] |= reach[:, :-1, :]
        grown[:, :-1, :] |= reach[:, 1:, :]
        grown[:, :, 1:] |= reach[:, :, :-1]
        grown[:, :, :-1] |= reach[:, :, 1:]
        grown &= open_mask
        if (grown == reach).all():
            return reach
        reach = grown


def crop_spanning(vol, size=64, porosity=(0.28, 0.34), seed=42, tries=60):
    """Deterministic search for a ``size``^3 crop whose porosity falls in the
    ``porosity`` window and whose pore space spans x (6-connected). Same args
    always give the same crop; raises RuntimeError if no offset qualifies.

    The canonical test crop (``sphere64`` / ``SPHERE64_SLICE``) is NOT regenerated
    through this search — it is the fixed slice sphere_0000[45:109, 131:195, 122:186].
    """
    vol = np.asarray(vol)
    if size > min(vol.shape):
        raise ValueError(f"crop size {size} exceeds the smallest volume dimension {min(vol.shape)}")
    lo, hi = porosity
    rng = np.random.default_rng(seed)
    for _ in range(tries):
        z, y, x = (int(rng.integers(0, d - size + 1)) for d in vol.shape)
        crop = vol[z : z + size, y : y + size, x : x + size]
        phi = 1.0 - crop.mean()
        if not (lo <= phi <= hi):
            continue
        seeds = np.zeros(crop.shape, bool)
        seeds[:, :, 0] = True
        if _flood_fill(~crop, seeds)[:, :, -1].any():
            return crop.copy()
    raise RuntimeError(f"no spanning crop with porosity in [{lo}, {hi}] found in {tries} tries")


# --- geometry assembly -------------------------------------------------------- #
def duct(nz, ny, nx):
    """Open square duct along x: solid z/y boundary planes, open interior."""
    mask = np.zeros((nz, ny, nx), bool)
    mask[0, :, :] = mask[-1, :, :] = True
    mask[:, 0, :] = mask[:, -1, :] = True
    return jnp.asarray(mask)


def cylinder(nz, ny, nx, R):
    """Straight circular tube along x: pore where (y-yc)^2 + (z-zc)^2 < R^2,
    solid outside, for every x; centered in the y-z cross-section. Staircased
    walls are the lattice price for a round tube; ``wall_normals`` handles them.
    The first ``n_in`` x-slabs serve as the inlet reservoir (held by the driver),
    so no separate end chamber is modeled."""
    zz, yy = np.indices((nz, ny))
    zc, yc = (nz - 1) / 2.0, (ny - 1) / 2.0
    disk_solid = ((yy - yc) ** 2 + (zz - zc) ** 2) >= R**2           # (nz, ny)
    mask = np.broadcast_to(disk_solid[:, :, None], (nz, ny, nx))
    return jnp.asarray(np.ascontiguousarray(mask))


def with_buffers(crop, buffer=10):
    """Drainage domain around a crop: ``buffer`` fully open slabs on each x side
    (so the interface starts clear of the inlet reset zone), then solid
    z/y boundary planes over the whole domain (no-slip walls, mask the wrap)."""
    crop = np.asarray(crop)
    nz, ny, nx = crop.shape
    solid = np.zeros((nz, ny, nx + 2 * buffer), bool)
    solid[:, :, buffer : buffer + nx] = crop
    solid[0, :, :] = solid[-1, :, :] = True
    solid[:, 0, :] = solid[:, -1, :] = True
    return jnp.asarray(solid)


# --- initial state + drainage driver ------------------------------------------- #
def init_drainage(nz, ny, nx, n_red):
    """Red slug of unit density in slabs x < n_red, blue elsewhere (also inside
    solids — diagnostics mask those out and bounce-back keeps them inert)."""
    xx = jnp.broadcast_to(jnp.arange(nx), (nz, ny, nx))
    rhoR = jnp.where(xx < n_red, 1.0, 0.0)
    W = d3q19.W[:, None, None, None]
    return color3d.State(W * rhoR[None], W * (1.0 - rhoR)[None])


def drain(state, params, solid, n_in, u_in, steps, nw=None, inlet="clamp", inlet_sa_red=1.0, outlet_sa_red=None, step_fn=None):
    """Run ``steps`` forced-drainage steps: color step -> inlet -> zero-gradient
    outlet. Pure, jit-able (wrap in jax.jit). Pass ``nw`` (from
    ``color3d.wall_normals``) with ``params.theta`` set to use the geometric wetting BC.

    ``inlet`` selects the inlet BC: ``"clamp"`` (default) is the velocity reservoir
    ``inlet_reservoir(n_in, u_in)`` (over-specifies rho=1 AND u, so inlet pressure cannot
    build); ``"zouhe"`` is the ΔP=0 per-color Zou-He *pressure* inlet ``zou_he_inlet``
    (inflow velocity free; ``n_in``/``u_in`` ignored); ``"velocity"`` is the rate-controlled
    ``eqm_velocity_inlet(u_in)`` — imposes u_x=u_in (sets the capillary number) with the
    inlet density/pressure FLOATING, so it builds against capillary barriers and actually
    drives the front (MRT-stable, single plane at x=0, ``n_in`` ignored; paired with the
    ``zou_he_outlet`` pressure outlet; the equilibrium form because a raw Zou-He velocity
    reconstruction is MRT-unstable — see ``eqm_velocity_inlet``).
    ``"piston"`` is MF-LBM's default bounce-back velocity inlet ``piston_inlet(u_in)``
    (Ladd moving wall): drain SEALS the x=0 plane as solid — the half-way bounce then
    reflects each colour's counter-stream off the piston face (blue bounces back as blue,
    never destroyed; the periodic wrap is cut) — and adds the swept-volume kick
    ``6*w_i*u_in`` at x=1, injecting exactly ``u_in*A`` red/step and zero blue
    (``sa_red=1``). Inlet pressure floats (builds against entry barriers);
    Ca = mu*u_in/sigma directly. ``nw`` from the unsealed mask leaves the piston face
    wetting-neutral (MF-LBM pins ghost phi red instead); ``n_in`` ignored.
    ``inlet_sa_red`` (zouhe/velocity/piston): colour fraction of the injected fluid —
    1.0 (default) pure red = non-wetting drainage, 0.0 pure blue = the wetting-phase
    flood (case/trapping); the piston ledger stays exact per colour (u_in*A*sa red +
    u_in*A*(1-sa) blue per step).
    ``outlet_sa_red`` (all pressure-outlet modes): None (default) keeps the zero-gradient
    colour split; a float pins the outlet colour (0.0 = defending blue — required for
    leak-free-inlet drainage; 1.0 = red reservoir for the wetting flood, else the
    self-referential outlet split ratchets the diffusive invader tail into a fake
    pool at the outlet; see zou_he_outlet).
    ``step_fn`` (optional, ``State -> State``): replaces the ``color3d.step`` call — the
    hook for ``multigpu.staged_halo_step`` / ``halo_step`` (multi-device halo exchange). Everything it
    needs (params, the STEP mask — sealed for the piston — and nw) must be baked in; the
    BCs here still use this function's own ``solid``/``step_solid`` and are unchanged."""
    modes = ("clamp", "zouhe", "velocity", "piston")
    if inlet not in modes:  # a typo would otherwise silently fall through to the clamp
        raise ValueError(f"unknown inlet mode {inlet!r}; expected one of {modes}")
    step_solid = solid
    if inlet == "piston":  # seal the piston plane; the halfway bounce IS the reflection half of the BC
        step_solid = jnp.asarray(solid).at[:, :, 0].set(True)

    def body(s, _):
        s = color3d.step(s, params, solid=step_solid, nw=nw) if step_fn is None else step_fn(s)
        if inlet == "zouhe":
            # full paper BC: per-color Zou-He pressure inlet + outlet (dP=0)
            s = bc.zou_he_inlet(s, solid=solid, rho_in=1.0, sa_red=inlet_sa_red)
            s = bc.zou_he_outlet(s, solid=solid, rho_out=1.0, sa_red=outlet_sa_red)
        elif inlet == "velocity":
            # rate-controlled: MRT-stable equilibrium velocity inlet (u_in fixed, pressure floats) + pressure outlet
            s = bc.eqm_velocity_inlet(s, u_in, solid=solid, sa_red=inlet_sa_red)
            s = bc.zou_he_outlet(s, solid=solid, rho_out=1.0, sa_red=outlet_sa_red)
        elif inlet == "piston":
            # Ladd moving-wall kick at the first fluid slab (x=1); reflection already done
            # by the halfway bounce off the sealed x=0 plane inside step()
            s = bc.piston_inlet(s, u_in, solid=step_solid, sa_red=inlet_sa_red)
            s = bc.zou_he_outlet(s, solid=step_solid, rho_out=1.0, sa_red=outlet_sa_red)
        else:
            s = bc.inlet_reservoir(s, n_in, u_in)
            s = bc.outlet_zero_gradient(s)
        return s, None

    return jax.lax.scan(body, state, None, length=steps)[0]


def imbibe(state, params, solid, n_in, steps, nw=None, inlet="clamp"):
    """Spontaneous imbibition: ``drain`` with u_in = 0 (capillary-driven, zero
    applied pressure). Wetting red (``params.theta < 90``) imbibing into a less-viscous
    blue (``params.omega2``). Pass ``inlet="zouhe"`` for the faithful ΔP=0 pressure inlet
    (inflow floats with the capillary demand; ``"clamp"`` pins it at u=0 and
    under-imbibes — see demo/washburn)."""
    return drain(state, params, solid, n_in, u_in=0.0, steps=steps, nw=nw, inlet=inlet)


# --- diagnostics (host-side NumPy) ---------------------------------------------- #
def _red_pore(state, solid):
    """Bool (nz, ny, nx): pore cells where red is the majority color."""
    rhoR, rhoB = color3d.densities(state)
    return np.asarray(rhoR > rhoB) & ~np.asarray(solid)


def saturation(state, solid):
    """Red mass fraction of the pore space: sum(rhoR) / sum(rhoR + rhoB) over pores.

    Counts the *whole* open domain — x-buffers and the clamped inlet reservoir
    included — so it understates the saturation of the crop interior alone."""
    rhoR, rhoB = color3d.densities(state)
    pore = ~np.asarray(solid)
    r, b = np.asarray(rhoR)[pore].sum(), np.asarray(rhoB)[pore].sum()
    return float(r / (r + b))


def breakthrough(state, solid, x):
    """True if any red-majority pore cell sits at slab >= x."""
    return bool(_red_pore(state, solid)[:, :, x:].any())


def filled_length(state, solid, x0):
    """Red-majority pore cells at slabs x >= x0 divided by the tube cross-section
    pore count at slab x0 — the imbibed length L(t). NumPy."""
    red = _red_pore(state, solid)                          # (nz, ny, nx) bool
    cross = int((~np.asarray(solid)[:, :, x0]).sum())
    return float(red[:, :, x0:].sum() / max(cross, 1))


def invasion_front(state, solid):
    """Largest x index holding a red-majority pore cell; -1 if there is no red.

    This is the leading-cell position, not a threshold on the cross-section-averaged
    red fraction: fingered 3D invasion has no clean averaged edge, so the leading
    cell is the well-defined front."""
    cols = _red_pore(state, solid).any(axis=(0, 1))
    return int(np.where(cols)[0].max()) if cols.any() else -1


def connected_to_inlet(state, solid):
    """Fraction of red pore cells 6-connected to the inlet slab (x = 0).
    No spurious red nucleation => ~1. Returns 1.0 when there is no red at all."""
    red = _red_pore(state, solid)
    if not red.any():
        return 1.0
    seeds = np.zeros(red.shape, bool)
    seeds[:, :, 0] = True
    return float(_flood_fill(red, seeds).sum() / red.sum())


def pressure_drop(state, solid, x_split):
    """Mean pressure cs2*rho over upstream pore cells (x < x_split) minus the
    downstream mean — the entry-pressure observable. ``x_split`` must be
    interior (0 < x_split < nx); an edge split leaves one side empty -> NaN."""
    rho = np.asarray((state.fR + state.fB).sum(0))
    pore = ~np.asarray(solid)
    up = rho[:, :, :x_split][pore[:, :, :x_split]].mean()
    down = rho[:, :, x_split:][pore[:, :, x_split:]].mean()
    return float(d3q19.CS2 * (up - down))
