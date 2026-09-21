"""Shared rendering helpers for the demos and campaigns (matplotlib + Pillow)."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

plt.switch_backend("Agg")  # headless: render straight to files

# Analysis curves (capillary-pressure, saturation, ledger, fit plots ...) are OPT-IN:
# a run writes its data — metrics.csv, report.md, fields, checkpoints — and its field
# renders and GIFs by default, and the curves only when plots are enabled. Drivers take
# ``--plots``; ``BOB_PLOTS=1`` enables them for a whole session. Standalone figure tools
# (publication_plot.py and friends) call ``enable_plots()`` themselves: making figures is
# their only job.
_PLOTS_ENABLED = os.environ.get("BOB_PLOTS", "") == "1"


def plots_enabled() -> bool:
    """True when the optional analysis curves should be written."""
    return _PLOTS_ENABLED


def enable_plots(on: bool = True) -> None:
    """Turn the optional analysis curves on (or off) for this process."""
    global _PLOTS_ENABLED
    _PLOTS_ENABLED = bool(on)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_gif(frame_paths: list[Path], out: Path, fps: int) -> None:
    frames = [Image.open(p).convert("RGB") for p in frame_paths]
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)


def field_frame(path: Path, rhoN, solid=None, dpi=130, scale=45):
    """Full-bleed RdBu_r render of the color field; solid cells drawn dark gray."""
    ny, nx = rhoN.shape
    img = np.ma.masked_where(solid, rhoN) if solid is not None else rhoN
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#3a3a3a")
    fig, ax = plt.subplots(figsize=(nx / scale, ny / scale))
    ax.imshow(img, origin="lower", cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def line_plot(path: Path, series, title, xlabel, ylabel, logy=False):
    """series: list of (xs, ys, label, kwargs) tuples drawn on one axes.

    An analysis curve: writes nothing unless plots are enabled (``plots_enabled``)."""
    if not _PLOTS_ENABLED:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot = ax.semilogy if logy else ax.plot
    for xs, ys, label, kw in series:
        if kw.pop("scatter", False):
            ax.scatter(xs, ys, label=label, **kw)
        else:
            plot(xs, ys, label=label, **kw)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    if any(s[2] for s in series):
        ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
