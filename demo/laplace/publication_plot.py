"""Publication-grade Laplace-law figure from the frozen case output.

Re-plots demo/laplace/output/plots/laplace_fit.csv (no re-simulation): measured
pressure jumps vs 1/R with the per-sigma linear fits, styled to match the
capillary_pressure publication set — STIX serif typography (matches the mathtext),
viridis per-sigma colors (dark = largest sigma, top line), inward major+minor ticks
on a full box, no grid, white-edged markers. The fitted sigma (slope/2) is folded
into each legend label, and the y axis carries an explicit x10^-2 scale instead of
matplotlib's floating offset text.

Artifacts: laplace_fit_pub.pdf (vector), laplace_fit_pub.png (300 dpi)

Usage:  uv run python demo/laplace/publication_plot.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")

HERE = Path(__file__).parent

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

SCALE = 1e2  # plot dp * 10^2 so the axis reads O(1) with an explicit x10^-2 in the label


def load_series(path):
    """CSV (sigma,R,inv_R,dp) -> [(sigma, inv_R[], dp[])] sorted by DESCENDING sigma."""
    per = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            per.setdefault(float(row["sigma"]), []).append((float(row["inv_R"]), float(row["dp"])))
    out = []
    for sig in sorted(per, reverse=True):
        pts = sorted(per[sig])
        out.append((sig, np.array([x for x, _ in pts]), np.array([y for _, y in pts])))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=HERE / "output" / "plots")
    args = ap.parse_args()

    series = load_series(args.out / "laplace_fit.csv")
    colors = plt.cm.viridis(np.linspace(0.03, 0.8, len(series)))
    with plt.rc_context(_PUB_RC):
        fig, ax = plt.subplots(figsize=(10.0, 7.0))
        x_max = max(float(np.max(ix)) for _, ix, _ in series) * 1.08
        fit_x = np.linspace(0.0, x_max, 50)
        y_top = 0.0
        for (sig, inv_R, dps), col in zip(series, colors):
            slope, intercept = np.polyfit(inv_R, dps, 1)
            y_top = max(y_top, (slope * x_max + intercept) * SCALE)
            ax.plot(fit_x, (slope * fit_x + intercept) * SCALE, color=col, lw=2.6, zorder=2)
            ax.plot(inv_R, dps * SCALE, "o", color=col, ms=9, mec="white", mew=1.1, zorder=3,
                    label=rf"$\sigma={sig:g}$")
        handles, labels = ax.get_legend_handles_labels()
        handles = [plt.Line2D([], [], color=h.get_color(), lw=2.6, marker="o", ms=8,
                              mec="white", mew=1.0) for h in handles]
        ax.set_xlim(0.0, x_max)
        ax.set_ylim(0.0, y_top * 1.12)
        ax.minorticks_on()
        ax.tick_params(labelsize=21)
        # blank the x origin label — it collides with the y-axis "0.0" at the corner
        ax.xaxis.set_major_formatter(lambda v, _pos: "" if v == 0 else f"{v:g}")
        ax.set_xlabel(r"$1/R$  (lattice units$^{-1}$)", fontsize=24)
        ax.set_ylabel(r"$\Delta p$  ($\times 10^{-2}$)", fontsize=24)
        ax.set_title(r"Laplace Law (3D):  $\Delta p = 2\sigma/R$", fontsize=21, pad=12)
        ax.legend(handles, labels, fontsize=19, loc="upper left", frameon=False,
                  borderpad=0.7, labelspacing=0.45, handletextpad=0.8)
        fig.tight_layout()
        fig.savefig(args.out / "laplace_fit_pub.pdf")
        fig.savefig(args.out / "laplace_fit_pub.png", dpi=300)
        plt.close(fig)
    print(f"wrote {args.out / 'laplace_fit_pub.pdf'} and .png")


if __name__ == "__main__":
    main()
