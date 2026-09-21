"""Publication-grade Washburn L(t) figure from the frozen case output.

Re-plots demo/washburn/output/washburn_Lt.csv (no re-simulation): LBM dots
vs the two-fluid Washburn analytic, direct-labeled at the right edge. Styled to
match the capillary_pressure publication set: STIX serif typography (matches the
mathtext), viridis per-angle colors, inward major+minor ticks on a full box,
no grid, white-edged markers. No in-axes title (the paper caption carries it).

Artifacts: washburn_Lt_pub.pdf (vector), washburn_Lt_pub.png (300 dpi)

Usage:  uv run python demo/washburn/publication_plot.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bob import d3q19
from bob.utils import washburn

plt.switch_backend("Agg")

HERE = Path(__file__).parent

# frozen run constants of washburn_demo.py (r=5, 12x12x101, reservoir 6)
GAMMA = 1.0 / 45
OM2 = 1.0 / 0.6
MU_W = float(d3q19.CS2) * 0.5
MU_NW = float(d3q19.CS2) * (1.0 / OM2 - 0.5)
R, NX, RES = 5, 101, 6
L_TUBE = NX - 2 * RES

INK = "0.15"

_PUB_RC = {
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "axes.linewidth": 1.4,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 7, "ytick.major.size": 7,
    "xtick.minor.size": 3.5, "ytick.minor.size": 3.5,
    "xtick.major.width": 1.4, "ytick.major.width": 1.4,
    "xtick.minor.width": 1.1, "ytick.minor.width": 1.1,
    "text.color": INK, "axes.edgecolor": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
}


def load_series(path):
    """CSV (theta,t,L) -> [(theta, ts, Ls)] in file order."""
    per = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            per.setdefault(float(row["theta"]), []).append((float(row["t"]), float(row["L"])))
    return [(th, np.array([t for t, _ in pts]), np.array([length for _, length in pts])) for th, pts in per.items()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=HERE / "output")
    args = ap.parse_args()

    per_theta = load_series(args.out / "washburn_Lt.csv")
    colors = plt.cm.viridis(np.linspace(0.04, 0.78, len(per_theta)))
    with plt.rc_context(_PUB_RC):
        fig, ax = plt.subplots(figsize=(7.2, 10.4))
        for i, ((theta, ts, Ls), col) in enumerate(zip(per_theta, colors)):
            A_an = washburn.capillary_A(3, R, GAMMA, theta)
            td = np.linspace(ts[0], ts[-1], 200)
            Ld = washburn.two_fluid_Lt(td - ts[0], A_an, MU_W, MU_NW, L_TUBE, L0=Ls[0])
            ax.plot(td, Ld, color=col, lw=2.6, zorder=2)
            ax.plot(ts, Ls, "o", color=col, ms=8.0, mec="white", mew=0.9, zorder=3)
            label = rf"$\theta = {theta:.0f}^\circ$" if i == 0 else rf"${theta:.0f}^\circ$"
            ax.annotate(
                label, (float(td[-1]), float(Ld[-1])),
                xytext=(9, 0), textcoords="offset points", va="center", ha="left",
                color=INK, fontsize=21,
            )
        t_max = max(float(np.max(ts)) for _, ts, _ in per_theta)
        y_max = max(float(np.max(Ls)) for _, _, Ls in per_theta)
        ax.set_xlim(0, t_max * 1.19)
        ax.set_ylim(0, y_max * 1.10)
        ax.minorticks_on()
        ax.tick_params(labelsize=21)
        ax.xaxis.set_major_locator(plt.MultipleLocator(10_000))
        # x origin label blanked — it collides with the y-axis "0" at the corner
        ax.xaxis.set_major_formatter(lambda x, _pos: "" if x == 0 else f"{x / 1000:g}")
        ax.set_xlabel(r"time step  ($\times 10^3$)", fontsize=24)
        ax.set_ylabel(r"penetration  $L$  (cells)", fontsize=24)
        handles = [
            plt.Line2D([], [], color="0.35", lw=0, marker="o", ms=8.0, mec="white", mew=0.9, label="LBM"),
            plt.Line2D([], [], color="0.35", lw=2.6, label="two-fluid analytic"),
        ]
        ax.legend(handles=handles, fontsize=19, loc="upper left", frameon=False,
                  borderpad=0.7, labelspacing=0.45, handletextpad=0.8)
        fig.tight_layout()
        fig.savefig(args.out / "washburn_Lt_pub.pdf")
        fig.savefig(args.out / "washburn_Lt_pub.png", dpi=300)
        plt.close(fig)
    print(f"wrote {args.out / 'washburn_Lt_pub.pdf'} and .png")


if __name__ == "__main__":
    main()
