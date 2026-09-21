"""Mathematical description of the Geometry-3 pore junction of Zacharoudiou, Chapman,
Boek & Crawshaw, J. Fluid Mech. 824 (2017), doi:10.1017/jfm.2017.363.

Pure rectangles on the reference's native 3 um lattice — no raster mask, so walls are
exactly straight (the only steps are the design's own staircase corners).

The integer node dimensions below were measured on the lattice dump in figure 13 of the
reference (3.0 um/px) by run-length analysis and snapped to the symmetric design. In the
axis-aligned view: T3 (60 um, inlet) left, T1 (27 um) right, T2 (47 um) bottom, T4
(100 um) top. The pore body is a stepped octagon, up-down symmetric about the T3/T1
centerline, all mouths centered on the body:

        |--T4=35--|
      [ tier  83 x 15 ]
    [  core 109 x 53   ]   <- T3 (19) enters left, T1 (9) exits right, centered
      [ tier  83 x 15 ]
        |--T2=16--|

Every fluid opening ends on a domain face (x=0 reservoir/inlet, x=-1 T1
outlet, y=0 T4 outlet, y=-1 T2 outlet) so the four ports sit at ambient
pressure via Zou-He planes. Etch depth 55 um -> 18 nodes in z.
"""

from __future__ import annotations

import numpy as np

DX_UM = 3.0                  # native lattice pitch of the reference
DEPTH = 18                   # 55 um etch depth in nodes

# stepped-octagon pore body, in body-local coordinates (y down, x right)
CORE_W, CORE_H = 109, 53     # central rectangle
TIER_W, TIER_H = 83, 15      # staircase tier above and below the core
BODY_H = CORE_H + 2 * TIER_H              # 83 rows total
TIER_X = (CORE_W - TIER_W) // 2           # 13: tier inset from the core's left edge

# mouths (width, offset of the run start in body-local coords), all centered
T4_W, T4_X = 35, (CORE_W - 35) // 2       # top slot, along x
T2_W, T2_X = 16, (CORE_W - 16) // 2       # bottom slot, along x
T3_W, T3_Y = 19, (BODY_H - 19) // 2       # left slot, along y (the inlet throat)
T1_W, T1_Y = 9, (BODY_H - 9) // 2         # right slot, along y

ARM = 200                    # straight channel length beyond the body (600 um, as in the reference)
RES_X = 30                   # reservoir pad depth ahead of T3
PAD = 25                     # pad half-extension beyond the T3 mouth


def plan():
    """(ny, nx) bool plan, True = solid; plus the body origin (by, bx)."""
    by, bx = ARM, RES_X + ARM                     # body top-left corner
    ny = by + BODY_H + ARM
    nx = bx + CORE_W + ARM
    p = np.ones((ny, nx), bool)

    def carve(y0, y1, x0, x1):
        p[y0:y1, x0:x1] = False

    carve(by + TIER_H, by + TIER_H + CORE_H, bx, bx + CORE_W)                  # core
    carve(by, by + TIER_H, bx + TIER_X, bx + TIER_X + TIER_W)                  # top tier
    carve(by + TIER_H + CORE_H, by + BODY_H, bx + TIER_X, bx + TIER_X + TIER_W)  # bottom tier
    carve(by + T3_Y, by + T3_Y + T3_W, RES_X, bx)                              # T3 arm
    carve(by + T3_Y - PAD, by + T3_Y + T3_W + PAD, 0, RES_X)                   # reservoir pad
    carve(by + T1_Y, by + T1_Y + T1_W, bx + CORE_W, nx)                        # T1 arm
    carve(0, by, bx + T4_X, bx + T4_X + T4_W)                                  # T4 arm
    carve(by + BODY_H, ny, bx + T2_X, bx + T2_X + T2_W)                        # T2 arm
    return p, (by, bx)


def build_domain():
    """3D solid mask (True = solid), metric regions, and view metadata."""
    p, (by, bx) = plan()
    ny, nx = p.shape
    nz = DEPTH + 2
    solid = np.ones((nz, ny, nx), bool)
    solid[1:-1] = p[None]

    probe = 60                                   # arm probe length for filling metrics
    regions = {
        "T3": (slice(by + T3_Y, by + T3_Y + T3_W), slice(bx - probe, bx)),
        "junction": (slice(by, by + BODY_H), slice(bx, bx + CORE_W)),
        "T1": (slice(by + T1_Y, by + T1_Y + T1_W), slice(bx + CORE_W, bx + CORE_W + probe)),
        "T2": (slice(by + BODY_H, by + BODY_H + probe), slice(bx + T2_X, bx + T2_X + T2_W)),
        "T4": (slice(by - probe, by), slice(bx + T4_X, bx + T4_X + T4_W)),
    }
    view = {"junction_origin": [by, bx], "junction_shape": [BODY_H, CORE_W], "dx_um": DX_UM}
    return solid, regions, view


def render(path, dpi=140):
    """Paper-palette plan render with the metric probe boxes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    solid, regions, _ = build_domain()
    mid = solid[solid.shape[0] // 2]
    img = np.where(mid[:, :, None], np.array([28, 172, 66], np.uint8), np.array([32, 48, 160], np.uint8))
    fig, ax = plt.subplots(figsize=(10, 8.5))
    ax.imshow(img)
    for name, (ys, xs) in regions.items():
        ax.add_patch(plt.Rectangle((xs.start, ys.start), xs.stop - xs.start, ys.stop - ys.start,
                                   fill=False, edgecolor="white", lw=1.2, ls="--"))
        ax.text(xs.start + 2, ys.start - 4, name, color="white", fontsize=10)
    nz, ny, nx = solid.shape
    ax.set_title(f"capillary_fill domain — {nz}x{ny}x{nx} @ {DX_UM:.0f} um/vox "
                 f"(arms {ARM * DX_UM:.0f} um; T3 inlet left, T1 right, T2 bottom, T4 top)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


if __name__ == "__main__":
    from pathlib import Path

    out = Path("demo/capillary_fill/output")
    out.mkdir(parents=True, exist_ok=True)
    solid, regions, view = build_domain()
    print("domain", solid.shape, "pore cells", int((~solid).sum()))
    render(out / "domain.png")
    print("wrote", out / "domain.png")
