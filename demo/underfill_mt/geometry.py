"""MT6225A TFBGA underfill-gap geometry (paper case 3: Wang et al., Microelectron.
Eng. 2016, PII S0167931715300575).

Voxelizes `mt6225a_package_geometry.json` into a bob (nz, ny, nx) bool mask
(True = solid) of the capillary underfill gap:

  * z = 0        substrate plane (solid)
  * z = nz-1     die plane (solid)
  * y = 0/ny-1   sealed chip side edges (walls on both sides, as the geometric
                 wetting BC requires: a one-sided wall under a periodic axis
                 biases the contact angle)
  * balls        spheres of diameter b = 0.30 mm truncated by both plates
                 (standoff 0.21 mm < b, center at mid-gap), one per populated site
  * x            flow axis: I-type dispensation edge at low x (+ inlet buffer),
                 vent edge at high x (+ outlet buffer)

The 25 depopulated sites (center + corners of the 17x17 grid) are simply absent —
they form the wide no-bump channels whose slower/faster filling the paper compares.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GEOMETRY_JSON = HERE / "mt6225a_package_geometry.json"
DX_MM = 0.03  # lattice pitch: gap 0.21 mm -> 7 fluid layers, ball dia 0.30 mm -> 10 cells
DX_MM_HIRES = 0.015  # 2x preset: 14 gap layers, ball dia 20 cells — resolves trapped micro voids


def load_package(path=GEOMETRY_JSON):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_gap_mask(pkg, dx=DX_MM, buffer_in=10, buffer_out=10, bump_shape="sphere", margin_mm=0.0):
    """Voxelize the underfill gap. Returns (solid, info): solid is (nz, ny, nx) bool
    True=solid; info holds the mm cell-center coordinate arrays, the chip x/y slices
    (excluding buffers and margins) and headline numbers for reports.

    ``bump_shape``: "sphere" (default; ball-diameter sphere truncated by both plates —
    leaves thin wedge crevices where the sphere grazes each plate) or "cylinder"
    (ball-diameter cylinder spanning the full gap — no near-tangent wedges).

    ``margin_mm``: the paper's flow margin (Wang 2016 Fig. 4 — flow region = main
    filling part + margin part) as bump-free side strips of that width along both
    y edges, at gap height (flat-margin model). Enables the paper's edge detour
    flow around the chip side-edges; the x-direction margins are the existing
    inlet/outlet buffers. 0.0 (default) = sealed side walls at the chip edge.
    The paper publishes no margin width, so it is a free parameter here."""
    D, E = pkg["body"]["D"], pkg["body"]["E"]
    gap = pkg["standoff_A1"]
    r = pkg["ball_diameter_b"] / 2.0
    nz_fluid = round(gap / dx)
    nx_chip, ny_chip = round(D / dx), round(E / dx)
    margin_c = round(margin_mm / dx)
    nx = buffer_in + nx_chip + buffer_out
    ny = ny_chip + 2 + 2 * margin_c
    nz = nz_fluid + 2

    solid = np.zeros((nz, ny, nx), bool)
    solid[0], solid[-1] = True, True
    solid[:, 0, :] = solid[:, -1, :] = True

    # cell-center physical coordinates (mm): chip spans x,y in [-D/2, D/2], z from substrate
    xs = (np.arange(nx) - buffer_in + 0.5) * dx - D / 2.0
    ys = (np.arange(ny) - 1 - margin_c + 0.5) * dx - E / 2.0
    zs = (np.arange(nz) - 1 + 0.5) * dx
    zc = gap / 2.0  # truncated-sphere ball center height

    if bump_shape not in ("sphere", "cylinder"):
        raise ValueError(f"unknown bump_shape {bump_shape!r}; expected 'sphere' or 'cylinder'")
    for ball in pkg["balls"]:
        bx, by = ball["x"], ball["y"]
        ix = np.nonzero(np.abs(xs - bx) <= r)[0]
        iy = np.nonzero(np.abs(ys - by) <= r)[0]
        iz = np.nonzero(np.abs(zs - zc) <= r)[0] if bump_shape == "sphere" else np.arange(1, nz - 1)
        if not (ix.size and iy.size and iz.size):
            continue
        dy2 = (ys[iy] - by)[None, :, None] ** 2
        dx2 = (xs[ix] - bx)[None, None, :] ** 2
        dz2 = (zs[iz] - zc)[:, None, None] ** 2 if bump_shape == "sphere" else np.zeros((iz.size, 1, 1))
        solid[np.ix_(iz, iy, ix)] |= dz2 + dy2 + dx2 < r * r

    chip_x = slice(buffer_in, buffer_in + nx_chip)
    chip_y = slice(1 + margin_c, 1 + margin_c + ny_chip)
    interior = ~solid[1:-1, chip_y, chip_x]
    info = {
        "dx": dx,
        "xs": xs,
        "ys": ys,
        "zs": zs,
        "chip_x": chip_x,
        "chip_y": chip_y,
        "margin_mm": margin_mm,
        "margin_cells": margin_c,
        "nz_fluid": nz_fluid,
        "gap_porosity": float(interior.mean()),
        "n_balls": len(pkg["balls"]),
        "bump_shape": bump_shape,
        "shape": (nz, ny, nx),
    }
    return solid, info
