"""Calibrate the SCAL pressure ladder from the rock's throat sizes (CPU, ~1 min).

The invasion-percolation entry pressure is set by the critical throat radius
r_c: the largest r such that the pore cells with distance-to-wall >= r still
percolate along x. Young-Laplace with the water contact angle theta_w = 180 -
theta_rock gives pc_entry = 2*sigma*cos(theta_w)/r_c. The ladder starts at
half the entry pressure, steps by entry/8, and is capped at 2.5x entry, a cap
set by the rock alone.

Relation to the outlet plate (plate_holdoff.py, output/holdoff.json): under a
4-5 cell diffuse interface the sharp-interface estimate 2*sigma*cos/r ~ 0.1 of a
2-voxel hole is not reached -- such a plate breaches at a measured pc ~ 0.015,
below the rock cap (0.028), and a 1-voxel hole plate is hydraulically closed
(zero water flux). scal_demo.py therefore uses 5-voxel plate pores (entry
~0.035, above the cap) together with a per-block scrubber in the outlet buffer
(--scrub, default on) that converts any produced red to blue, mass-conserving
and logged. If holdoff.json is present, the measured 2-voxel breach pressure is
stored in the ladder file as ``plate_breach_measured`` for reference.

Writes plate_check/output/ladder.json (regenerable; run_scal.slurm re-runs this
if the file is missing). ``--structure``/``--out`` calibrate other rocks with
the same rule, e.g. output/ladder256.json for test/assets/bentheimer256.npy
(the 256^3 sample is independent of the 128^3 one -- its throat is tighter, so
the 128 numbers do not transfer)."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

SIGMA, THETA_ROCK = 0.05, 135.0
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent / "output"


def critical_radius(pore, tol=0.05):
    """Largest r where {dist-to-solid >= r} still spans x (6-connectivity), by bisection."""
    dt = ndimage.distance_transform_edt(pore)

    def spans(r):
        lab, _ = ndimage.label(dt >= r)
        return bool((set(np.unique(lab[:, :, 0])) & set(np.unique(lab[:, :, -1]))) - {0})

    lo, hi = 1.0, float(dt.max())
    assert spans(lo), "pore space does not percolate at r=1 -- wrong geometry?"
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if spans(mid) else (lo, mid)
    return lo, dt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--structure", type=Path, default=ROOT / "test" / "assets" / "bentheimer128.npy")
    ap.add_argument("--out", type=Path, default=OUT / "ladder.json")
    args = ap.parse_args()

    pore = ~np.load(args.structure)
    rc, dt = critical_radius(pore)
    cosw = float(np.cos(np.radians(180.0 - THETA_ROCK)))
    pc_entry = 2.0 * SIGMA * cosw / rc
    holdoff = OUT / "holdoff.json"
    plate_breach = None
    if holdoff.exists():
        plate_breach = json.loads(holdoff.read_text()).get("2", {}).get("breach_pc")
    ladder = {
        "r_crit_vox": round(rc, 3),
        "pc_entry": round(pc_entry, 5),
        "pc0": round(0.5 * pc_entry, 5),
        "dpc": round(pc_entry / 8.0, 5),
        "pc_cap": round(2.5 * pc_entry, 5),  # set by the rock alone, independent of the plate
        "plate_breach_measured": plate_breach,
        "sigma": SIGMA,
        "theta_rock": THETA_ROCK,
    }
    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps(ladder, indent=2) + "\n")
    q = np.percentile(dt[pore], [50, 90, 99])
    print(json.dumps(ladder, indent=2))
    print(f"pore dist-to-wall p50/p90/p99 = {q[0]:.1f}/{q[1]:.1f}/{q[2]:.1f} vox")
    if plate_breach:
        print(f"2-voxel-pore plate breach measured at {plate_breach} (ladder cap {ladder['pc_cap']})")
    assert 2.0 <= rc <= 12.0, f"r_crit {rc:.2f} vox outside the expected Bentheimer range"


if __name__ == "__main__":
    main()
