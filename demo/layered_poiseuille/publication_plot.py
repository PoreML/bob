"""Publication-grade layered-Poiseuille profile figure.

Re-runs the five viscosity ratios of layered_demo.py at its default
parameters (steady profiles only — no GIF/frames), caches them to
profile_pub.csv, and draws the combined figure in the publication style of
the washburn / capillary_pressure sets: STIX serif typography, viridis
per-M colors, inward major+minor ticks on a full box, no grid, white-edged
markers, no in-axes title (the paper caption carries it). If profile_pub.csv
already exists the sim is skipped (delete it to force a re-run).

Artifacts: profile_pub.pdf (vector), profile_pub.png (300 dpi), profile_pub.csv

Usage:  uv run python demo/layered_poiseuille/publication_plot.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

plt.switch_backend("Agg")

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

INK = "0.15"
NOTE = "0.35"

_PUB_RC = {
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "axes.linewidth": 1.4,
    # explicit: layered_demo mutates global rcParams (grid on, box spines off)
    # at import time, and rc_context only overrides listed keys
    "axes.grid": False,
    "axes.spines.top": True, "axes.spines.right": True,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 7, "ytick.major.size": 7,
    "xtick.minor.size": 3.5, "ytick.minor.size": 3.5,
    "xtick.major.width": 1.4, "ytick.major.width": 1.4,
    "xtick.minor.width": 1.1, "ytick.minor.width": 1.1,
    "text.color": INK, "axes.edgecolor": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
}


def simulate(out):
    """Demo-default run of all five ratios; returns [(M, u_lbm, u_an)] and writes the CSV."""
    from layered_demo import CS2, NU_B, analytic_layered, run_sim

    # mirror layered_demo.parse_args() defaults (no CLI passthrough)
    args = argparse.Namespace(
        ratios=[5, 10, 15, 20, 30], nz=64, ny=8, nx=128,
        umax=0.025, steps=600_000, block=2000, rtol=1e-6,
    )
    zz = np.arange(args.nz)
    band = (zz >= args.nz // 4) & (zz < 3 * args.nz // 4)
    m_max = max(args.ratios)
    u_g1 = analytic_layered(args.nz, np.where(band, NU_B / m_max, NU_B), 1.0)
    drho = (args.umax / float(u_g1.max())) * (args.nx - 1) / CS2

    series = []
    with (out / "profile_pub.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["M", "z", "u_lbm", "u_an"])
        for m in sorted(args.ratios):
            c = run_sim(m, args, drho)
            ux = c["hist"][-1][1]
            print(f"M={m}: max rel err {c['rel']:.2%}  ({c['steps']} steps, {c['seconds']:.0f}s)")
            for z, ul, ua in zip(zz, ux, c["u_an"]):
                w.writerow([m, int(z), f"{ul:.8e}", f"{ua:.8e}"])
            series.append((m, np.asarray(ux), np.asarray(c["u_an"])))
    return series


def load_series(path):
    """profile_pub.csv -> [(M, u_lbm, u_an)] ordered by M."""
    per = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            per.setdefault(int(float(row["M"])), []).append((float(row["u_lbm"]), float(row["u_an"])))
    return [(m, np.array([u for u, _ in pts]), np.array([a for _, a in pts])) for m, pts in sorted(per.items())]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=HERE / "output")
    args = ap.parse_args()

    cached = args.out / "profile_pub.csv"
    series = load_series(cached) if cached.exists() else simulate(args.out)
    nz = len(series[0][1])
    zz = np.arange(nz)
    band_lo, band_hi = nz // 4, 3 * nz // 4
    colors = plt.cm.viridis(np.linspace(0.04, 0.78, len(series)))
    y_max = 1.18 * max(float(max(ua.max(), ux.max())) for _, ux, ua in series)

    with plt.rc_context(_PUB_RC):
        fig, ax = plt.subplots(figsize=(11.0, 7.2))
        ax.axvspan(band_lo - 0.5, band_hi - 0.5, color=INK, alpha=0.05, lw=0)
        handles = []
        for (m, ux, ua), col in zip(series, colors):
            ax.plot(zz, ua, color=col, lw=2.6, zorder=2)
            ax.plot(zz[1:-1:2], ux[1:-1:2], "o", color=col, ms=8.0, mec="white", mew=0.9, zorder=3)
            handles.append(Line2D([], [], color=col, lw=2.6, marker="o", ms=8.0,
                                  mec="white", mew=0.9, label=rf"$M = {m}$"))
        ax.set_xlim(0, nz - 1)
        ax.set_ylim(0, y_max)
        ax.minorticks_on()
        ax.tick_params(labelsize=21)
        ax.xaxis.set_major_locator(plt.MultipleLocator(16))  # majors land on the band edges
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
        ax.yaxis.get_offset_text().set_fontsize(21)
        ax.set_xlabel(r"$z$  (lattice units, walls at both ends)", fontsize=24)
        ax.set_ylabel(r"$u_x$", fontsize=24)
        ax.text(0.5 * band_lo, 0.10 * y_max, "viscous\n" + r"($\nu_{\mathrm{b}}$)",
                ha="center", va="center", color=NOTE, fontsize=19)
        ax.text(0.5 * (band_lo + band_hi), 0.035 * y_max, r"thin fluid ($\nu_{\mathrm{r}}$)",
                ha="center", va="center", color=NOTE, fontsize=19)
        ax.text(0.98, 0.97, "lines — analytic\ndots — LBM", transform=ax.transAxes,
                ha="right", va="top", fontsize=19, color=NOTE)
        ax.legend(handles=handles, fontsize=19, loc="upper left", frameon=False,
                  borderpad=0.7, labelspacing=0.45, handletextpad=0.8, borderaxespad=1.0)
        fig.tight_layout()
        fig.savefig(args.out / "profile_pub.pdf")
        fig.savefig(args.out / "profile_pub.png", dpi=300)
        plt.close(fig)
    print(f"wrote {args.out / 'profile_pub.pdf'} and .png")


if __name__ == "__main__":
    main()
