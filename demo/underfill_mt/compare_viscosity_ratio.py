"""Viscosity-ratio ladder comparison for the MT6225A underfill — publication style,
replotted from the frozen metrics.csv of each finished run (never re-simulates).

Overlays the chip-gap fill curves for M = 1.5 / 10 / 20 / 30 (viridis, dark -> light
with increasing M) and plots steps-to-42 % vs M. Runs without a metrics.csv are
skipped with a note.

Usage:  uv run python demo/underfill_mt/compare_viscosity_ratio.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"

# (label M, run dir, nu_red, nu_blue) — dark -> light viridis in this fixed order
RUNS = [
    (1.5, OUT / "run", 0.05, 0.033),
    (10, OUT / "m10_sphere", 0.25, 0.025),
    (20, OUT / "m20_sphere", 0.5, 0.025),
    (30, OUT / "m30_sphere", 0.75, 0.025),
]

RC = {
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "axes.linewidth": 1.4,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
}


def load_rows(run):
    with open(run / "metrics.csv") as fh:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]


def steps_to(rows, fill):
    for r in rows:
        if r["fill"] >= fill:
            return r["step"]
    return None


def main():
    avail = [(m, d, nr, nb) for m, d, nr, nb in RUNS if (d / "metrics.csv").exists()]
    for m, d, _, _ in RUNS:
        if not (d / "metrics.csv").exists():
            print(f"[skip M={m:g}: {d} has no metrics.csv yet]")
    colors = plt.cm.viridis(np.linspace(0.03, 0.8, len(avail)))
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
        a = axes[0]
        pts = []
        for (m, d, nr, nb), c in zip(avail, colors):
            rows = load_rows(d)
            a.plot(
                [r["step"] / 1e3 for r in rows],
                [100 * r["fill"] for r in rows],
                "-",
                color=c,
                lw=2.2,
                label=rf"$M={m:g}$  ($\nu_r={nr:g}$, $\nu_b={nb:g}$)",
            )
            s42 = steps_to(rows, 0.42)
            if s42 is not None:
                pts.append((m, s42, c))
        a.axhline(42.0, color="gray", ls="--", lw=1.2)
        a.set_xlabel(r"time step ($\times 10^{3}$)", fontsize=19)
        a.set_ylabel("chip gap fill (%)", fontsize=19)
        a.set_title("Fill Vs Viscosity Ratio", fontsize=20)
        a.tick_params(labelsize=16)
        a.legend(fontsize=13, frameon=False)
        a = axes[1]
        for m, s42, c in pts:
            a.plot(m, s42 / 1e3, "o", ms=10, mfc=c, mec="white", mew=1.3)
        a.plot([m for m, s, _ in pts], [s / 1e3 for _, s, _ in pts], "-", color="gray", lw=1.2, zorder=0)
        a.set_xlabel(r"viscosity ratio $M=\nu_r/\nu_b$", fontsize=19)
        a.set_ylabel(r"steps to 42 % ($\times 10^{3}$)", fontsize=19)
        a.set_title("Fill Time Scaling", fontsize=20)
        a.tick_params(labelsize=16)
        fig.tight_layout()
        fig.savefig(OUT / "viscosity_ratio_ladder.png", dpi=300)
        fig.savefig(OUT / "viscosity_ratio_ladder.pdf")
        plt.close(fig)
    print(f"figure -> {OUT}/viscosity_ratio_ladder.(png|pdf)  ({len(avail)} runs included)")


if __name__ == "__main__":
    main()
