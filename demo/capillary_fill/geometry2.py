"""Mathematical description of the Geometry-2 pore junction (square pore, four unequal
throats) of Zacharoudiou, Chapman, Boek & Crawshaw, J. Fluid Mech. 824 (2017),
doi:10.1017/jfm.2017.363 — the micro-model of figures 3 and 4 of the reference.

Pure rectangles on the reference's native 3 um lattice, like ``geometry.py`` for
Geometry 3. Dimensions from the reference (table 1 / figure 3 caption): etch depth 45 um,
throat widths 33, 51, 65, 107 um, contact angle 26 deg. The pore body is a plain 238 um
square (the Geometry-1 pore width; the LB frames of figure 4(b) measure 238 +- 5 um and
show no corner notches). The throat placement follows figure 4 (experiment and LB frames
agree, after rotating 45 deg back to axis-aligned): the NWP is injected through the 51 um
throat, the 107 um throat is straight opposite, the 33 um throat is on one side and the
65 um throat on the other. Young-Laplace order of the downstream entry pressures (table 1
of the reference): T4 107 um (1357 Pa) < T3 65 um (1620 Pa) < T2 51 um (1794 Pa). In the
dynamic figure-4 experiment the syringe feeds the 51 um throat and the NWP leaves through
T4 first.

Layout here (bob's flow axis is x, piston at x=0):

                  |-T1=11-|                 (33 um, y=0 face)
        inlet ==> [  body 79 x 79  ] ==> outlet T4=36 (107 um, x=-1 face)
        T_in=17 (51 um)  |-T3=22-|          (65 um, y=-1 face)

All four arms are straight, ARM nodes long, and end on a domain face: the inlet
arm on x=0 (sealed piston plane), T4 on x=-1, T1 on y=0, T3 on y=-1. Etch depth
45 um -> 15 nodes in z between the two bounding wall planes.
"""

from __future__ import annotations

import numpy as np

DX_UM = 3.0                  # native lattice pitch of the reference
DEPTH = 15                   # 45 um etch depth in nodes
BODY = 79                    # 238 um square pore body

# throat widths in nodes (um / 3, rounded) and the run offset that centres each mouth on the body
W_IN, W_IN_Y = 17, (BODY - 17) // 2      # 51 um — NWP injection throat (x=0 side)
W_T4, W_T4_Y = 36, (BODY - 36) // 2      # 107 um — straight opposite (x=-1 side)
W_T1, W_T1_X = 11, (BODY - 11) // 2      # 33 um — side throat on the y=0 face
W_T3, W_T3_X = 22, (BODY - 22) // 2      # 65 um — side throat on the y=-1 face
WIDTHS_UM = {"inlet": 51, "T1": 33, "T3": 65, "T4": 107}

ARM = 150                    # straight arm length (450 um; the reference LB frames show 440-560 um arms)


def plan(arm=ARM):
    """(ny, nx) bool plan, True = solid; plus the body origin (by, bx)."""
    by, bx = arm, arm
    ny = arm + BODY + arm
    nx = arm + BODY + arm
    p = np.ones((ny, nx), bool)

    def carve(y0, y1, x0, x1):
        p[y0:y1, x0:x1] = False

    carve(by, by + BODY, bx, bx + BODY)                                  # square pore body
    carve(by + W_IN_Y, by + W_IN_Y + W_IN, 0, bx)                        # inlet arm (51 um), from x=0
    carve(by + W_T4_Y, by + W_T4_Y + W_T4, bx + BODY, nx)                # T4 arm (107 um), to x=-1
    carve(0, by, bx + W_T1_X, bx + W_T1_X + W_T1)                        # T1 arm (33 um), to y=0
    carve(by + BODY, ny, bx + W_T3_X, bx + W_T3_X + W_T3)                # T3 arm (65 um), to y=-1
    return p, (by, bx)


def build_domain(arm=ARM):
    """3D solid mask (True = solid), metric regions, and view metadata.

    Regions: ``inlet`` = the whole inlet arm, ``body`` = the pore body, ``T1``/``T3``/``T4``
    = the whole downstream arms (red fraction over a whole arm = how far the NWP has
    invaded it, which is what figure 4's stages are matched on)."""
    p, (by, bx) = plan(arm)
    ny, nx = p.shape
    nz = DEPTH + 2
    solid = np.ones((nz, ny, nx), bool)
    solid[1:-1] = p[None]
    regions = {
        "inlet": (slice(by + W_IN_Y, by + W_IN_Y + W_IN), slice(0, bx)),
        "body": (slice(by, by + BODY), slice(bx, bx + BODY)),
        "T1": (slice(0, by), slice(bx + W_T1_X, bx + W_T1_X + W_T1)),
        "T3": (slice(by + BODY, ny), slice(bx + W_T3_X, bx + W_T3_X + W_T3)),
        "T4": (slice(by + W_T4_Y, by + W_T4_Y + W_T4), slice(bx + BODY, nx)),
    }
    view = {"body_origin": [by, bx], "body_shape": [BODY, BODY], "arm": arm, "dx_um": DX_UM}
    return solid, regions, view


def render(path, arm=ARM, dpi=140):
    """Paper-palette plan render with the metric region boxes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    solid, regions, _ = build_domain(arm)
    mid = solid[solid.shape[0] // 2]
    img = np.where(mid[:, :, None], np.array([28, 172, 66], np.uint8), np.array([252, 252, 252], np.uint8))
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(img)
    for name, (ys, xs) in regions.items():
        ax.add_patch(plt.Rectangle((xs.start, ys.start), xs.stop - xs.start, ys.stop - ys.start,
                                   fill=False, edgecolor="k", lw=1.0, ls="--"))
        ax.text(xs.start + 2, ys.start - 4, f"{name} ({WIDTHS_UM.get(name, '')} um)".replace(" ( um)", ""),
                color="k", fontsize=9)
    nz, ny, nx = solid.shape
    ax.set_title(f"drainage_fill domain (Geometry 2) — {nz}x{ny}x{nx} @ {DX_UM:.0f} um/vox, arms {arm * DX_UM:.0f} um\n"
                 "NWP piston at x=0 (51 um throat); T4 107 um opposite; T1 33 um top; T3 65 um bottom")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


if __name__ == "__main__":
    from pathlib import Path

    out = Path("demo/capillary_fill/output_fig4_ca1e-4")
    out.mkdir(parents=True, exist_ok=True)
    solid, regions, view = build_domain()
    print("domain", solid.shape, "pore cells", int((~solid).sum()))
    render(out / "domain.png")
    print("wrote", out / "domain.png")
