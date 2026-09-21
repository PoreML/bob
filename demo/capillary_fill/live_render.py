"""On-the-fly rendering for a running capillary_fill / drainage_fill demo.

Watches <out>/frames/ and renders every new z-midplane phi snapshot to a PNG in
<out>/live/ (solid green, NWP blue, WP white), stamped with the step and the
per-region phase fractions from metrics.csv; latest.png always holds the newest view
and capillary_fill.gif the animation so far.

The run type is read from <out>/domain.json: a Geometry-2 run (drainage_fill_demo.py,
key "arm") rebuilds the solid with geometry2.build_domain(arm) and treats red as the
NWP; otherwise the Geometry-3 imbibition run (capillary_fill_demo.py) is assumed, with
red as the WP. Pure numpy/matplotlib — runs on a CPU node while the GPU job is going.

Usage:
  uv run python demo/capillary_fill/live_render.py            # watch output_oh006 until idle 10 min
  uv run python demo/capillary_fill/live_render.py --once     # render backlog and exit
  uv run python demo/capillary_fill/live_render.py --out demo/capillary_fill/output_fig4_ca1e-4
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path("demo/capillary_fill")
OUT = HERE / "output_oh006"

GREEN = np.array([28, 172, 66], np.uint8)
BLUE = np.array([32, 48, 160], np.uint8)
WHITE = np.array([252, 252, 252], np.uint8)


RED_IS_NWP = False          # set from domain.json in main()

# csv columns stamped on each frame, per run type (whichever exist)
COLS_G3 = ("red_T2", "red_T1", "red_T4", "red_junction")
COLS_G2 = ("inlet", "body", "T4", "T3", "T1")


def paint(phi, solid_mid):
    red = phi[:, :, None] > 0
    red_col, blue_col = (BLUE, WHITE) if RED_IS_NWP else (WHITE, BLUE)
    return np.where(solid_mid[:, :, None], GREEN[None, None],
                    np.where(red, red_col[None, None], blue_col[None, None]))


def load_metrics():
    rows = np.atleast_1d(np.genfromtxt(OUT / "metrics.csv", delimiter=",", names=True))
    cols = [n for n in (COLS_G2 if RED_IS_NWP else COLS_G3) if n in rows.dtype.names]
    return rows, cols


def metrics_row(step):
    try:
        rows, cols = load_metrics()
        hit = rows[rows["step"] == step]
        if len(hit):
            r = hit[0]
            txt = "  ".join(f"{n.replace('red_', '')}={r[n]:.2f}" for n in cols)
            if "dp_in" in rows.dtype.names:
                txt += f"  dp_in={r['dp_in']:.4f}"
            return txt
    except Exception:
        pass
    return ""


def main():
    global OUT, RED_IS_NWP
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT, help="demo output dir to watch")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--idle-exit", type=float, default=600, help="exit after this many idle seconds")
    ap.add_argument("--gif-stride", type=int, default=1, help="use every Nth rendered frame in the GIF")
    args = ap.parse_args()
    OUT = args.out
    live = OUT / "live"
    live.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(HERE))
    dom = {}
    if (OUT / "domain.json").exists():
        dom = json.loads((OUT / "domain.json").read_text())
    if "arm" in dom:                                   # Geometry-2 drainage run
        from geometry2 import build_domain

        RED_IS_NWP = True
        solid, _, _ = build_domain(dom["arm"])
        print(f"geometry2 (arm={dom['arm']}), red = NWP -> blue in the render")
    else:                                              # Geometry-3 imbibition run
        from geometry import build_domain

        solid, _, _ = build_domain()
        print("geometry (G3), red = WP -> white in the render")
    solid_mid = np.asarray(solid)[solid.shape[0] // 2]

    done, last_new, gif_at = set(), time.time(), 0
    while True:
        fresh = sorted(p for p in (OUT / "frames").glob("mid_*.npz") if p.name not in done)
        for p in fresh:
            step = int(p.stem.split("_")[1])
            try:
                phi = np.load(p)["phi"].astype(np.float32)
            except Exception as e:
                print(f"{p.name} not readable yet ({e}); retrying next pass", file=sys.stderr)
                continue
            done.add(p.name)
            img = paint(phi, solid_mid)
            fig, ax = plt.subplots(figsize=(9, 8.2))
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"step {step:,}   {metrics_row(step)}", fontsize=11, family="monospace")
            fig.tight_layout()
            fig.savefig(live / f"mid_{step:08d}.png", dpi=110)
            plt.close(fig)
            shutil.copyfile(live / f"mid_{step:08d}.png", live / "latest.png")
            last_new = time.time()
        if fresh:
            print(f"rendered {len(fresh)} frame(s), latest {max(done)}")
        if len(done) - gif_at >= 8:          # refresh the animation every ~8 new frames
            gif(live, stride=args.gif_stride)
            gif_at = len(done)
        if args.once or (time.time() - last_new > args.idle_exit):
            break
        time.sleep(10)
    gif(live, stride=args.gif_stride)


def gif(live, duration_ms=120, width=560, stride=1):
    """Rebuild the on-the-fly animation from the rendered live frames (every `stride`-th one)."""
    try:
        from PIL import Image

        paths = sorted(live.glob("mid_*.png"))[::max(1, stride)]
        if len(paths) < 2:
            return
        frames = []
        for p in paths:
            im = Image.open(p).convert("P", palette=Image.ADAPTIVE)
            im.thumbnail((width, width))
            frames.append(im)
        frames[0].save(live / "capillary_fill.gif", save_all=True, append_images=frames[1:],
                       duration=duration_ms, loop=0)
        print(f"gif refreshed ({len(frames)} frames)")
    except Exception as e:
        print("gif refresh failed:", e, file=sys.stderr)


if __name__ == "__main__":
    main()
