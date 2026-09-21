"""Publication figures for the MT6225A hi-res underfill run (STIX serif, inward
ticks, viridis series, PNG+PDF twins), replotted from the frozen run artifacts
(state42.npz, metrics.csv, run_meta.json); never re-simulates.

Usage:  uv run python demo/underfill_mt/publication_plot.py [--run output/hires]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from geometry import build_gap_mask, load_package

HERE = Path(__file__).resolve().parent

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


def load_run(run):
    meta = __import__("json").loads((run / "run_meta.json").read_text())
    dx = meta["extra"]["dx_mm"]
    buffer = 20 if dx < 0.02 else 10  # the --hires / default buffer presets of underfill_demo.py
    solid, info = build_gap_mask(load_package(), dx=dx, buffer_in=buffer, buffer_out=buffer)
    with open(run / "metrics.csv") as fh:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]
    return solid, info, rows, dx


def top_view_np(fR, fB, solid):
    rhoR, rhoB = fR.sum(0), fB.sum(0)
    rhoN = (rhoR - rhoB) / np.maximum(rhoR + rhoB, 1e-12)
    pore = ~solid
    w = pore.sum(axis=0)
    col = (rhoN * pore).sum(axis=0) / np.maximum(w, 1)
    return np.ma.masked_where(w == 0, col), w == 0


def front_profile_np(fR, fB, solid, chip_x, xs):
    red = ((fR.sum(0) > fB.sum(0)) & ~solid)[:, :, chip_x].any(axis=0)
    idx = np.where(red, np.arange(red.shape[1])[None, :], -1).max(axis=1)
    xf = xs[chip_x][np.maximum(idx, 0)].astype(float)
    xf[idx < 0] = np.nan
    return xf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=HERE / "output" / "hires")
    args = ap.parse_args()
    run = args.run
    solid, info, rows, _dx = load_run(run)
    xs, ys, chip_x = info["xs"], info["ys"], info["chip_x"]
    snap = np.load(run / "state42.npz")
    fR, fB, step42 = snap["fR"], snap["fB"], int(snap["step"])

    with plt.rc_context(RC):
        # Fig 1 — melt-front map at the paper's 42 % comparison point (Fig. 8 analogue)
        col, dead = top_view_np(fR, fB, solid)
        fig, ax = plt.subplots(figsize=(8.6, 8.0))
        ax.imshow(
            col,
            origin="lower",
            extent=(xs[0], xs[-1], ys[0], ys[-1]),
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            aspect="equal",
            interpolation="nearest",
        )
        ax.imshow(
            np.ma.masked_where(~dead, np.ones_like(col.data)),
            origin="lower",
            extent=(xs[0], xs[-1], ys[0], ys[-1]),
            cmap="Greys",
            vmin=0,
            vmax=2.2,
            aspect="equal",
        )
        ax.set_xlabel("$x$ (mm)", fontsize=21)
        ax.set_ylabel("$y$ (mm)", fontsize=21)
        ax.tick_params(labelsize=18)
        ax.set_title("Melt Front At 42 % Fill", fontsize=22)
        fig.tight_layout()
        fig.savefig(run / "publication_melt_front.png", dpi=300)
        fig.savefig(run / "publication_melt_front.pdf")
        plt.close(fig)

        # Fig 2 — fill curve + melt-front profile (theta = encapsulant contact angle, 30 deg)
        c_curve, c_prof = plt.cm.viridis(0.03), plt.cm.viridis(0.55)
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
        a = axes[0]
        a.plot([r["step"] / 1e3 for r in rows], [100 * r["fill"] for r in rows], "-", color=c_curve, lw=2.2)
        a.plot(step42 / 1e3, 42.0, "o", ms=9, mfc=c_prof, mec="white", mew=1.2, zorder=5)
        a.axhline(42.0, color="gray", ls="--", lw=1.2)
        a.set_xlabel(r"time step ($\times 10^{3}$)", fontsize=19)
        a.set_ylabel("chip gap fill (%)", fontsize=19)
        a.set_title("Fill Fraction", fontsize=20)
        a.tick_params(labelsize=16)
        prof = front_profile_np(fR, fB, solid, chip_x, xs)
        a = axes[1]
        a.plot(prof, ys, "-", color=c_prof, lw=1.8)
        span = 10.4 / 2 + 0.15  # outermost ball surface: the edge channels lie beyond
        for yb in (-span, span):
            a.axhline(yb, color="gray", ls=":", lw=1.2)
        a.set_xlabel(r"$x_{\mathrm{front}}$ (mm)", fontsize=19)
        a.set_ylabel("$y$ (mm)", fontsize=19)
        a.set_title("Melt-Front Profile At 42 % Fill", fontsize=20)
        a.tick_params(labelsize=16)
        fig.tight_layout()
        fig.savefig(run / "publication_fill_front.png", dpi=300)
        fig.savefig(run / "publication_fill_front.pdf")
        plt.close(fig)
    print(f"publication figures -> {run}/publication_melt_front.(png|pdf), publication_fill_front.(png|pdf)")


if __name__ == "__main__":
    main()
