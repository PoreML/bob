"""Validation tests for porous-media drainage on 3D pore geometry (bob.porous3d).

Red (non-wetting) injected at constant velocity displaces wetting blue. The
geometry is a 64^3 crop of a real synthetic sphere pack (cut from
test/assets/sphere256.npy — sphere_0000 @ zyx(45,131,122), porosity 0.303).
Covered, in order: asset/loader -> flood fill -> spanning-crop search ->
inlet/outlet BCs -> geometry assembly + initial state -> diagnostics on
synthetic states -> duct drainage (front speed and volume balance).

Run:  uv run pytest test/test_porous_drainage.py -v
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bob import bc, color3d, d3q19, misc, porous3d

ASSET = Path(__file__).parent / "assets" / "sphere256.npy"

# Capillary number: Ca = mu*u_in/sigma with mu = cs2*(1/omega - 1/2) = 1/6.
# The invasion tests inject at U_IN_INVADE (Ca ~ 0.07): at Ca ~ 0.01 capillary
# invasion pins at the first throat within a test-sized step budget. Walls are
# inert (no color exchange with the fluid), so there is no wall-film creep
# inflating the front and it advances at the delivered mean speed ~0.2*u_in.
U_IN_INVADE = 0.02
# theta=140: strongly non-wetting red (geometric wetting BC).
PARAMS3 = color3d.Params(omega=1.0, sigma=0.05, beta=0.7, theta=140.0)


# --- asset + loader ----------------------------------------------------------- #
def test_asset_loads_and_is_porous():
    vol = porous3d.sphere64(ASSET)
    assert vol.shape == (64, 64, 64) and vol.dtype == bool
    phi = 1.0 - vol.mean()
    assert 0.28 <= phi <= 0.34, f"porosity {phi:.3f} outside the ~0.3 window"


def test_asset_pore_space_spans_x():
    vol = porous3d.sphere64(ASSET)
    seeds = np.zeros(vol.shape, bool)
    seeds[:, :, 0] = True
    reach = porous3d._flood_fill(~vol, seeds)
    assert reach[:, :, -1].any(), "no inlet->outlet pore path"


def test_load_structure_rejects_non_bool(tmp_path):
    bad = tmp_path / "bad.npy"
    np.save(bad, np.zeros((4, 4, 4), dtype=float))
    with pytest.raises(ValueError, match="bool"):
        porous3d.load_structure(bad)


def test_load_structure_rejects_wrong_ndim(tmp_path):
    bad = tmp_path / "flat.npy"
    np.save(bad, np.zeros((4, 4), dtype=bool))
    with pytest.raises(ValueError, match="3D"):
        porous3d.load_structure(bad)


# --- flood fill (6-connectivity) ---------------------------------------------- #
def test_flood_fill_finds_straight_tube():
    solid = np.ones((5, 5, 8), bool)
    solid[2, 2, :] = False  # 1-cell tube along x
    seeds = np.zeros(solid.shape, bool)
    seeds[:, :, 0] = True
    reach = porous3d._flood_fill(~solid, seeds)
    assert reach[2, 2, -1]


def test_flood_fill_blocked_tube_does_not_leak():
    solid = np.ones((5, 5, 8), bool)
    solid[2, 2, :] = False
    solid[2, 2, 4] = True  # plug
    seeds = np.zeros(solid.shape, bool)
    seeds[:, :, 0] = True
    reach = porous3d._flood_fill(~solid, seeds)
    assert reach[2, 2, 3] and not reach[2, 2, 5:].any()


def test_flood_fill_no_diagonal_leak():
    # Two pore cells touching only at a corner must not connect (6-connectivity).
    solid = np.ones((3, 3, 3), bool)
    solid[0, 0, 0] = False
    solid[1, 1, 1] = False
    seeds = np.zeros(solid.shape, bool)
    seeds[0, 0, 0] = True
    reach = porous3d._flood_fill(~solid, seeds)
    assert reach[0, 0, 0] and not reach[1, 1, 1]


# --- crop_spanning -------------------------------------------------------------- #
def _tube_volume(n=32, tube_z=8, tube_y=8):
    # Mostly solid with one open 3x3 tube along x. A 16^3 crop that contains the
    # whole tube cross-section has porosity 9/(16*16) ~ 0.035 and spans x.
    vol = np.ones((n, n, n), bool)
    vol[tube_z : tube_z + 3, tube_y : tube_y + 3, :] = False
    return vol


def test_crop_spanning_deterministic_and_spans():
    vol = _tube_volume()
    # window admits only full-tube crops (0.035); partial-tube crops (~0.023) fail
    a = porous3d.crop_spanning(vol, size=16, porosity=(0.03, 0.04), seed=7)
    b = porous3d.crop_spanning(vol, size=16, porosity=(0.03, 0.04), seed=7)
    assert (a == b).all() and a.shape == (16, 16, 16)
    seeds = np.zeros(a.shape, bool)
    seeds[:, :, 0] = True
    assert porous3d._flood_fill(~a, seeds)[:, :, -1].any()


def test_crop_spanning_raises_when_nothing_qualifies():
    vol = np.ones((32, 32, 32), bool)  # all solid: porosity 0 everywhere
    with pytest.raises(RuntimeError, match="no spanning crop"):
        porous3d.crop_spanning(vol, size=16, porosity=(0.28, 0.34), seed=0, tries=10)


def test_crop_spanning_rejects_oversize():
    vol = np.zeros((16, 16, 16), bool)
    with pytest.raises(ValueError, match="crop size"):
        porous3d.crop_spanning(vol, size=32)


def test_crop_spanning_requires_x_spanning_pores():
    # Porosity passes the window but plugs at x=8 and x=24 block every possible
    # 16-cell x-window, so no crop spans: the search must reject on connectivity.
    vol = _tube_volume()
    vol[8:11, 8:11, 8] = True
    vol[8:11, 8:11, 24] = True
    with pytest.raises(RuntimeError, match="no spanning crop"):
        porous3d.crop_spanning(vol, size=16, porosity=(0.025, 0.04), seed=7, tries=40)


# --- 3D inlet/outlet BCs -------------------------------------------------------- #
def test_inlet_reservoir3d_sets_pure_red_at_velocity():
    st = misc.init_sphere(nz=6, ny=7, nx=12, R=2)
    st = bc.inlet_reservoir(st, n_in=3, u_in=0.01)
    rhoR, rhoB = color3d.densities(st)
    np.testing.assert_allclose(np.asarray(rhoR[:, :, :3]), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(rhoB[:, :, :3]), 0.0, atol=1e-12)
    u = color3d.velocity(st)
    np.testing.assert_allclose(np.asarray(u[0][:, :, :3]), 0.01, atol=1e-12)
    np.testing.assert_allclose(np.asarray(u[1][:, :, :3]), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(u[2][:, :, :3]), 0.0, atol=1e-12)


def test_inlet_reservoir3d_default_is_rest_weights():
    st = misc.init_sphere(nz=5, ny=5, nx=8, R=1)
    st = bc.inlet_reservoir(st, n_in=2)
    expect = np.broadcast_to(np.asarray(d3q19.W)[:, None, None, None], (19, 5, 5, 2))
    np.testing.assert_allclose(np.asarray(st.fR[:, :, :, :2]), expect, atol=1e-12)
    np.testing.assert_allclose(np.asarray(st.fB[:, :, :, :2]), 0.0, atol=1e-12)


def test_outlet_zero_gradient3d_anchors_unit_density():
    st = misc.init_sphere(nz=5, ny=5, nx=8, R=1)
    # make the second-to-last slab 20% over-dense to verify the rescale
    st = color3d.State(st.fR.at[:, :, :, -2].multiply(1.2), st.fB.at[:, :, :, -2].multiply(1.2))
    out = bc.outlet_zero_gradient(st)
    rho_last = np.asarray((out.fR + out.fB).sum(0))[:, :, -1]
    np.testing.assert_allclose(rho_last, 1.0, atol=1e-12)


# --- geometry assembly + driver -------------------------------------------------- #
def test_duct_walls():
    solid = np.asarray(porous3d.duct(6, 7, 12))
    assert solid[0].all() and solid[-1].all()
    assert solid[:, 0, :].all() and solid[:, -1, :].all()
    assert not solid[1:-1, 1:-1, :].any()


def test_with_buffers_layout():
    crop = porous3d.sphere64(ASSET)
    solid = np.asarray(porous3d.with_buffers(crop, buffer=10))
    assert solid.shape == (64, 64, 84)
    assert solid[0].all() and solid[-1].all() and solid[:, 0, :].all() and solid[:, -1, :].all()
    assert not solid[1:-1, 1:-1, :10].any() and not solid[1:-1, 1:-1, -10:].any()
    assert (solid[1:-1, 1:-1, 10:74] == crop[1:-1, 1:-1, :]).all()


def test_init_drainage3d_is_red_slug():
    st = porous3d.init_drainage(5, 6, 10, n_red=4)
    rhoR, rhoB = color3d.densities(st)
    np.testing.assert_allclose(np.asarray(rhoR[:, :, :4]), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(rhoR[:, :, 4:]), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(rhoR + rhoB), 1.0, atol=1e-12)


# --- diagnostics on synthetic states --------------------------------------------- #
def _state_from_red(red):
    rhoR = jnp.asarray(np.asarray(red, dtype=float))
    W = d3q19.W[:, None, None, None]
    return color3d.State(W * rhoR[None], W * (1.0 - rhoR)[None])


def test_saturation_is_red_fraction_of_pore_space():
    solid = porous3d.duct(6, 6, 10)
    red = np.zeros((6, 6, 10), bool)
    red[:, :, :5] = True
    assert abs(porous3d.saturation(_state_from_red(red), solid) - 0.5) < 1e-12


def test_breakthrough_and_inlet_connectivity():
    solid = porous3d.duct(6, 6, 10)
    red = np.zeros((6, 6, 10), bool)
    red[1:-1, 1:-1, :3] = True  # 48 inlet-connected red pore cells
    red[2, 2, 7] = True  # 1 disconnected red blob
    st = _state_from_red(red)
    assert porous3d.breakthrough(st, solid, x=7)
    assert not porous3d.breakthrough(st, solid, x=8)
    assert abs(porous3d.connected_to_inlet(st, solid) - 48 / 49) < 1e-12


def test_invasion_front_is_max_red_x():
    solid = porous3d.duct(6, 6, 10)
    red = np.zeros((6, 6, 10), bool)
    red[1:-1, 1:-1, :4] = True
    assert porous3d.invasion_front(_state_from_red(red), solid) == 3


def test_empty_red_diagnostics_have_documented_sentinels():
    solid = porous3d.duct(6, 6, 10)
    st = _state_from_red(np.zeros((6, 6, 10), bool))  # all blue
    assert porous3d.invasion_front(st, solid) == -1
    assert porous3d.connected_to_inlet(st, solid) == 1.0
    assert porous3d.saturation(st, solid) == 0.0


def test_with_buffers_zero_buffer_keeps_shape():
    crop = porous3d.sphere64(ASSET)
    solid = np.asarray(porous3d.with_buffers(crop, buffer=0))
    assert solid.shape == (64, 64, 64)
    assert solid[0].all() and solid[-1].all() and solid[:, 0, :].all() and solid[:, -1, :].all()


def test_pressure_drop_measures_density_step():
    solid = porous3d.duct(4, 4, 8)
    fR = d3q19.W[:, None, None, None] * jnp.ones((19, 4, 4, 8))
    fR = fR.at[:, :, :, :4].multiply(1.2)
    st = color3d.State(fR, jnp.zeros_like(fR))
    assert abs(porous3d.pressure_drop(st, solid, x_split=4) - d3q19.CS2 * 0.2) < 1e-12


# --- duct drainage sanity --------------------------------------------------------- #
def test_duct_front_advances_at_u_in():
    # No-slip square duct: the rho=1 velocity inlet relaxes to a duct profile
    # whose mean flux is well below nominal (weak compliance + 4-wall no-slip),
    # so the front advances at the true delivered mean speed ~0.2*u_in. Assert a
    # generous band around that and that swept volume tracks injected red volume.
    nz, ny, nx, n_in = 16, 16, 120, 4
    solid = porous3d.duct(nz, ny, nx)
    nw = color3d.wall_normals(np.asarray(solid))
    st = porous3d.init_drainage(nz, ny, nx, n_red=10)  # clear of the inlet reset zone
    runner = jax.jit(lambda s: porous3d.drain(s, PARAMS3, solid, n_in, U_IN_INVADE, 1500, nw=nw))
    st = runner(st)  # warm-up: interface forms, transients decay
    x1, red1 = porous3d.invasion_front(st, solid), float(np.asarray(color3d.densities(st)[0]).sum())
    for _ in range(3):
        st = runner(st)
    x2, red2 = porous3d.invasion_front(st, solid), float(np.asarray(color3d.densities(st)[0]).sum())
    speed = (x2 - x1) / 4500.0
    assert 0.1 * U_IN_INVADE < speed < 0.5 * U_IN_INVADE, f"front speed {speed:.2e}, expected ~0.2*{U_IN_INVADE}"
    swept = (x2 - x1) * (ny - 2) * (nz - 2)
    gained = red2 - red1
    assert abs(swept - gained) < 0.30 * gained, f"volume balance off: swept {swept:.0f} vs gained {gained:.0f}"
