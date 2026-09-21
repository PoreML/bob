"""demo/scal unit tests: porous-plate/domain builders and the pressure Ladder.

Pure NumPy/Python — no GPU, no solver steps. scal_demo is imported from the
demo folder (demos are not a package)."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo" / "scal"))
import scal_demo


def test_plate_holes_and_open_fraction():
    # plate design: 5x5 through-pores on an 8-pitch, 20 thick, offset 3
    # (a 2x2-pore / 3-thick plate is breached — see the build_plate docstring)
    plate = scal_demo.build_plate(128, 128)
    assert plate.shape == (128, 128, 20)
    assert plate.dtype == bool
    open_frac = (~plate).mean()
    assert abs(open_frac - 25 / 64) < 1e-12  # 5x5 holes on an 8-pitch
    # every hole is through-going: open columns identical across the thickness
    assert (plate == plate[:, :, :1]).all()
    # offset 3 clears the solid boundary walls on both low edges
    assert plate[:3, :, 0].all() and plate[:, :3, 0].all()
    # hole lattice periodic in y/z with period 8 (128 % 8 == 0)
    assert (plate[:, :, 0] == np.roll(plate[:, :, 0], 8, axis=0)).all()
    assert (plate[:, :, 0] == np.roll(plate[:, :, 0], 8, axis=1)).all()


def test_domain_layout_and_percolation():
    rock = np.load(Path(__file__).parent / "assets" / "bentheimer128.npy")
    solid, reg = scal_demo.build_domain(rock, buf_in=10, buf_out=6, thick=3, hole=2, pitch=4)
    nz, ny, nx = solid.shape
    assert (nz, ny, nx) == (128, 128, 10 + 128 + 3 + 6)
    assert reg["inbuf"] == (0, 10) and reg["rock"] == (10, 138)
    assert reg["plate"] == (138, 141) and reg["outbuf"] == (141, nx)
    # solid no-slip walls seal the y/z faces over the WHOLE domain (core-holder
    # convention, same as porous3d.with_buffers)
    assert solid[0, :, :].all() and solid[-1, :, :].all()
    assert solid[:, 0, :].all() and solid[:, -1, :].all()
    # buffers fully open inside the walls, rock slice preserved inside the walls,
    # plate in direct contact with the rock face
    core = (slice(1, -1), slice(1, -1))
    assert not solid[core[0], core[1], : reg["rock"][0]].any()
    assert not solid[core[0], core[1], reg["outbuf"][0] :].any()
    assert (solid[core[0], core[1], reg["rock"][0] : reg["rock"][1]] == rock[1:-1, 1:-1, :]).all()
    # whole domain still percolates inlet->outlet through rock AND plate holes
    lab, _ = ndimage.label(~solid)
    span = (set(np.unique(lab[:, :, 0])) & set(np.unique(lab[:, :, -1]))) - {0}
    assert len(span) >= 1


def test_theta_field_regions():
    rock = np.zeros((8, 8, 5), bool)
    solid, reg = scal_demo.build_domain(rock, buf_in=3, buf_out=3, thick=3, hole=2, pitch=4)
    th = scal_demo.theta_field_for(solid.shape, reg)
    assert th.shape == solid.shape
    assert (th[:, :, reg["rock"][0] : reg["rock"][1]] == 135.0).all()
    assert (th[:, :, reg["plate"][0] :] == 150.0).all()  # plate + outlet buffer water-wet


def test_ladder_drainage_reverses_on_stall():
    lad = scal_demo.Ladder(pc0=0.01, dpc=0.002)
    pcs = [lad.pc]
    for sw in (0.95, 0.80, 0.60, 0.50, 0.499, 0.4985):  # last two: |dSw| < 0.005 twice
        lad.advance(sw)
        pcs.append(lad.pc)
    assert lad.leg == "imb" and not lad.done
    # 5 up-steps then one down-step off the max
    assert np.allclose(pcs[:6], [0.01, 0.012, 0.014, 0.016, 0.018, 0.02])
    assert abs(lad.pc_max - 0.02) < 1e-12 and abs(pcs[-1] - 0.018) < 1e-12


def test_ladder_pc_cap_forces_reversal():
    lad = scal_demo.Ladder(pc0=0.01, dpc=0.002, pc_cap=0.011)
    lad.advance(0.9)  # would go to 0.012 > cap -> reverse instead
    assert lad.leg == "imb" and abs(lad.pc - 0.008) < 1e-12


def test_ladder_imbibition_done_on_stall_and_floor():
    lad = scal_demo.Ladder(pc0=0.01, dpc=0.002)
    for sw in (0.9, 0.6, 0.5995, 0.599):  # drain: stall twice -> flips to imb
        lad.advance(sw)
    assert lad.leg == "imb"
    lad.advance(0.7)  # big change: stall resets, keep lowering
    lad.advance(0.85)
    lad.advance(0.851)  # stall 1
    lad.advance(0.8512)  # stall 2 -> done
    assert lad.done
    # floor: never stalls, walks down to -floor_factor*pc_max and stops
    lad2 = scal_demo.Ladder(pc0=0.004, dpc=0.002, floor_factor=1.0)
    lad2.advance(0.9)
    lad2.advance(0.6)
    lad2.advance(0.5995)
    lad2.advance(0.599)
    assert lad2.leg == "imb"
    sw, n = 0.3, 0
    while not lad2.done and n < 100:
        sw += 0.05  # always a significant change
        lad2.advance(sw)
        n += 1
    assert lad2.done and lad2.pc >= -lad2.pc_max - 1e-12


def test_ladder_flat_start_does_not_stall():
    # below the entry pressure NOTHING moves -- that flatness must not count as a
    # stall (else drainage reverses before ever draining / imbibition quits at Swi)
    lad = scal_demo.Ladder(pc0=0.005, dpc=0.002, pc_cap=0.02)
    for _ in range(6):
        lad.advance(0.999)  # dead flat below entry
    assert lad.leg == "drain"  # still climbing (until the cap forces reversal)
    lad2 = scal_demo.Ladder(pc0=0.01, dpc=0.002)
    for sw in (0.9, 0.6, 0.5995, 0.599):  # drain leg: activates then stalls -> flip
        lad2.advance(sw)
    assert lad2.leg == "imb"
    for _ in range(4):
        lad2.advance(0.599)  # imb flat at Swi: pre-activation, must NOT finish
    assert not lad2.done


def test_ladder_roundtrip():
    lad = scal_demo.Ladder(pc0=0.01, dpc=0.002, pc_cap=0.05)
    lad.advance(0.9)
    lad.advance(0.7)
    clone = scal_demo.Ladder.from_dict(json.loads(json.dumps(lad.to_dict())))
    for sw in (0.65, 0.649, 0.6485):
        lad.advance(sw)
        clone.advance(sw)
    assert (lad.leg, lad.pc, lad.done) == (clone.leg, clone.pc, clone.done)
