"""Flip-chip solder-ball underfill on the flipchip geometry dataset (single-run driver).

Capillary underfill of a thin die/substrate gap crossed by a square staggered
array of truncated-sphere solder balls (geometry dataset `data/flipchip/samples/`:
uniform stencil balls, 2 D side-wall clearance, per-sample D 32-54 / gap 20-50 /
pitch 1.5-2.1 D, in cells). The drive is capillary suction only (Zou-He dP=0
pressure inlet and outlet, `porous3d.drain(inlet="zouhe")`), sigma 0.05,
beta 0.99 (thin interface), solver defaults MRT + akai + CSF, fp32 — with two
knobs set per run:

  * viscosity ratio M = nu_red/nu_blue: nu_blue is pinned at 0.025 (the
    stability floor in these thin gaps; nu_blue <= 0.02 diverges) and
    nu_red = M * 0.025, so fill time is ~linear in M;
  * encapsulant contact angle theta (deg) on all walls.

Logging (every --block = 1000 steps): metrics.csv row + top-view GIF frame +
3D PyVista frame (red melt opaque, solid translucent, `flow_camera`) +
fields .h5/.xdmf append (phi, rho, p, umag, u) + run_meta.json heartbeat.
Checkpoints every --ckpt-every blocks + fill-target + end; GIFs (2D top view
and 3D) refresh with the checkpoints. The run stops when the ball-array fill
fraction reaches --fill-target (default 0.99) plus --extra-blocks settle blocks
(finish_type "filled") — the intended terminator for every run. The step cap
(--steps, default 3M; finish_type "step_cap") is a safety net for a gap that
stalls, not a run length: the longest campaign run ended at 1,265,000 steps. A
run can also end on divergence (non-finite saturation, finish_type "diverged").

The dataset is read from `$GEOMETRY_DATA/flipchip/samples`, defaulting to the
sibling checkout `<repo>/../geometry/data`.

Usage (from the repository root):
  .venv/bin/python case/underfill/underfill_demo.py \
      --sample 7 --m-ratio 10 --theta 40 \
      --out case/underfill/runs/fc0007_M10_th40 --name fc0007_M10_th40
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import jax

# fp32 is the solver default (bandwidth-bound, ~2x steps/s, validated at parity).
FP32 = os.environ.get("BOB_FP64") != "1"
jax.config.update("jax_enable_x64", not FP32)

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

from bob import color3d, porous3d
from bob.utils import stats, viz, viz3d
from bob.utils.file import FieldWriter, RunMeta

HERE = Path(__file__).resolve().parent
# geometry dataset tree: $GEOMETRY_DATA, else the sibling checkout <repo>/../geometry/data.
GEOMETRY_DATA = Path(os.environ.get("GEOMETRY_DATA") or HERE.parent.parent.parent / "geometry" / "data")
DATASET = GEOMETRY_DATA / "flipchip" / "samples"

# ---- the standardized setting (module constants = the contract) -----------------
SIGMA = 0.05
BETA = 0.99  # recoloring sharpness: thin interface
NU_BLUE = 0.025  # air kinematic viscosity: the thin-gap stability floor (<= 0.02 diverges)
BLOCK = 1000  # the log interval: metrics + 2D/3D frames + fields.h5 cadence
FILL_TARGET = 0.99  # ball-array fill fraction that ends the run (+ settle blocks)
FIELD_VARS = "phi,rho,p,umag,u"


def omega_of(nu: float) -> float:
    return 1.0 / (3.0 * nu + 0.5)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", type=int, required=True, help="flipchip dataset sample index")
    p.add_argument("--m-ratio", type=float, required=True, help="viscosity ratio M = nu_red/nu_blue (campaign menu: 5, 10, 20, 30)")
    p.add_argument("--theta", type=float, required=True, help="encapsulant contact angle on all walls (deg)")
    p.add_argument("--out", type=Path, required=True, help="run directory")
    p.add_argument("--name", type=str, required=True, help="run name (folder/h5/gif stem)")
    p.add_argument("--sigma", type=float, default=SIGMA)
    p.add_argument("--beta", type=float, default=BETA)
    p.add_argument("--nu-blue", type=float, default=NU_BLUE)
    p.add_argument(
        "--steps",
        type=int,
        default=3_000_000,
        help="safety cap only — the run is meant to end at --fill-target + settle ('filled'). Sized so the "
        "fill target always governs: the longest campaign run ended at 1,265,000 steps (fill time is "
        "~linear in M)",
    )
    p.add_argument("--block", type=int, default=BLOCK, help="steps per block = the log/frame/fields.h5 interval")
    p.add_argument("--fill-target", type=float, default=FILL_TARGET, help="ball-array fill fraction that ends the run")
    p.add_argument("--extra-blocks", type=int, default=10, help="blocks run past the fill target to let trailing pockets settle")
    p.add_argument("--min-void", type=int, default=8, help="minimum trapped-air component size (cells)")
    p.add_argument("--ckpt-every", type=int, default=25, help="blocks between checkpoints + GIF refresh (0 = end only)")
    p.add_argument("--field-vars", type=str, default=FIELD_VARS, help="comma-separated fields.h5 variables")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--plots", action="store_true", help="also write the analysis curve (fill.svg); off by default")
    p.add_argument("--resume", type=Path, default=None, help="checkpoint .npz (or dir: newest) to continue")
    return p.parse_args()


def stage_geometry(args, out):
    """Stage the dataset sample into the run folder (or verify the staged copy):
    the .npy sha256 must match the dataset JSON recorded at generation time."""
    stem = f"flipchip_{args.sample:04d}"
    npy, meta_json = out / f"{args.name}.npy", out / f"{args.name}_geometry.json"
    if not meta_json.exists():
        out.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(DATASET / f"{stem}.json", meta_json)
    info = json.loads(meta_json.read_text())
    if not npy.exists():
        shutil.copyfile(DATASET / f"{stem}.npy", npy)
    solid = np.load(npy)
    sha = hashlib.sha256(solid.tobytes()).hexdigest()
    if sha != info["sha256"]:
        sys.exit(f"staged geometry {npy} sha {sha[:12]} != dataset {info['sha256'][:12]} — wrong/corrupt stage?")
    return solid, info


def find_voids(state, solid_np, min_vol, threshs=(0.5, 0.25, 0.1)):
    """Trapped-air components at several blue-fraction thresholds; components
    touching the inlet/outlet x planes are excluded.
    Returns ({thresh: (count, cells, rhoB mass)}, majority mask, mist mask)."""
    rhoR, rhoB = (np.asarray(f) for f in color3d.densities(state))
    pore = ~solid_np
    frac = np.where(pore, rhoB / np.maximum(rhoR + rhoB, 1e-12), 0.0)
    out, masks = {}, {}
    for th in threshs:
        blue = frac > th
        lab, n = ndimage.label(blue)
        if n == 0:
            out[th], masks[th] = (0, 0, 0.0), np.zeros_like(blue)
            continue
        edge = np.unique(np.concatenate([lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        keep = np.ones(n + 1, bool)
        keep[0] = False
        keep[edge] = False
        keep &= sizes >= min_vol
        mask = keep[lab]
        out[th], masks[th] = (int(keep.sum()), int(sizes[keep].sum()), float(rhoB[mask].sum())), mask
    return out, masks[threshs[0]], masks[threshs[-1]]


def top_view(state, solid_np):
    """z-averaged color field over the pore columns (the melt-front map)."""
    rhoR, rhoB = (np.asarray(f) for f in color3d.densities(state))
    rhoN = np.asarray(color3d.color_field(rhoR, rhoB))
    pore = ~solid_np
    w = pore.sum(axis=0)
    col = (rhoN * pore).sum(axis=0) / np.maximum(w, 1)
    return np.ma.masked_where(w == 0, col), w == 0


def main():
    args = parse_args()
    if args.plots:
        viz.enable_plots(True)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    solid_np, info = stage_geometry(args, out)
    nz, ny, nx = solid_np.shape
    x0, x1 = info["array_x"]
    solid = jnp.asarray(solid_np)
    nw = color3d.wall_normals(solid_np)
    frames = out / "frames"
    frames3d = out / "frames3d"
    frames.mkdir(parents=True, exist_ok=True)
    frames3d.mkdir(parents=True, exist_ok=True)

    nu_red = args.m_ratio * args.nu_blue
    p = color3d.Params(
        omega=omega_of(nu_red), omega2=omega_of(args.nu_blue), sigma=args.sigma, beta=args.beta, theta=args.theta
    )
    n_in = 20  # open inlet buffer slabs (dataset buffer_cells)
    region = ~solid_np[1:-1, 1:-1, x0:x1]  # square ball-array pore region for the fill metric
    cam = viz3d.flow_camera(solid_np.shape)

    def fill_fraction(state):
        red = porous3d._red_pore(state, solid_np)[1:-1, 1:-1, x0:x1]
        return float(red.sum() / max(region.sum(), 1))

    # ---- init / resume --------------------------------------------------------
    if args.resume is not None:
        ckpt_path = stats.latest_checkpoint(args.resume) if args.resume.is_dir() else args.resume
        if ckpt_path is None:
            sys.exit(f"no checkpoint found under {args.resume}")
        ck = np.load(ckpt_path)
        state = color3d.State(jnp.asarray(ck["fR"]), jnp.asarray(ck["fB"]))
        step0 = int(ck["step"])
        print(f"resumed {ckpt_path.name}: step {step0}")
    else:
        state = porous3d.init_drainage(nz, ny, nx, n_red=n_in + 2)
        step0 = 0
        for stale in (*frames.glob("frame_*.png"), *frames3d.glob("frame3d_*.png")):
            stale.unlink()

    geom = f"sample {args.sample:04d} (D {info['D']}, gap {info['gap']}, pitch {info['pitch_realized']:.2f}D)"
    title = f"{args.name} ({geom}, {info['n_balls']} balls, M {args.m_ratio:g}, theta {args.theta:.0f} deg, capillary dP=0)"
    meta = RunMeta(
        out / "run_meta.json",
        params=p,
        solid=solid_np,
        geometry_source=f"geometry data/flipchip/samples/flipchip_{args.sample:04d}",
        target_steps=args.steps,
        resume_step=step0,
        extra={
            "sample": args.sample,
            "D": info["D"],
            "gap": info["gap"],
            "pitch_realized": info["pitch_realized"],
            "n_balls": info["n_balls"],
            "gap_porosity": info["gap_porosity"],
            "m_ratio": args.m_ratio,
            "nu_red": nu_red,
            "nu_blue": args.nu_blue,
            "theta": args.theta,
            "inlet": "zouhe",
            "block": int(args.block),
        },
    )
    fields = ("step", "fill", "front", "voids", "void_cells", "voids25", "cells25", "mass25", "voids10", "cells10", "mass10")
    rows = stats.carry_rows(out, step0, fields=fields) if step0 else None
    monitor = stats.RunMonitor(out, title=title, curve_field="fill", rows=rows, meta=meta)
    fw = FieldWriter(
        out / f"uf_{args.name}.h5",
        solid_np,
        fields=tuple(v for v in args.field_vars.split(",") if v),
        mask_solid=True,
        attrs={
            "sample": int(args.sample),
            "D": int(info["D"]),
            "gap": int(info["gap"]),
            "pitch_realized": float(info["pitch_realized"]),
            "m_ratio": float(args.m_ratio),
            "nu_red": float(nu_red),
            "nu_blue": float(args.nu_blue),
            "theta": float(args.theta),
            "sigma": float(args.sigma),
            "inlet": "zouhe",
        },
        meta=meta,
    )
    runner = jax.jit(lambda s: porous3d.drain(s, p, solid, n_in, 0.0, args.block, nw=nw, inlet="zouhe"))

    def frame(step, fill, voids, vmask, mist):
        (nv, vv, _), (nm, mm, _) = voids[0.5], voids[0.1]
        col, dead = top_view(state, solid_np)
        fig, ax = plt.subplots(figsize=(10, 8.4))
        ax.imshow(col, origin="lower", cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
        ax.imshow(
            np.ma.masked_where(~dead, np.ones_like(col.data)), origin="lower", cmap="Greys", vmin=0, vmax=2.5, aspect="equal"
        )
        mcol = (mist & ~vmask).any(axis=0)
        ax.imshow(
            np.ma.masked_where(~mcol, np.ones_like(col.data)), origin="lower", cmap="autumn", vmin=0, vmax=1, aspect="equal"
        )  # sub-majority mist pockets -> yellow
        vcol = vmask.any(axis=0)
        ax.imshow(
            np.ma.masked_where(~vcol, np.ones_like(col.data)), origin="lower", cmap="brg", vmin=0, vmax=1, aspect="equal"
        )  # trapped air -> green
        ax.set_title(
            f"{title}\nstep {step} — array fill {fill * 100:.0f} %, voids {nv} ({vv} cells, green), "
            f"mist fB>0.1: {nm} ({mm} cells, yellow)"
        )
        ax.set_xlabel("x (flow, cells)")
        ax.set_ylabel("y (cells)")
        fig.tight_layout()
        fig.savefig(frames / f"frame_{step:08d}.png", dpi=110)
        fig.savefig(out / "latest.png", dpi=110)
        plt.close(fig)

    def frame3d(step, fill):
        rhoR, rhoB = (np.asarray(f) for f in color3d.densities(state))
        rhoN = np.asarray(color3d.color_field(rhoR, rhoB))
        path = frames3d / f"frame3d_{step:08d}.png"
        viz3d.phase_frame(
            path, rhoN, solid_np, camera=cam, zoom=1.0,
            text=f"{args.name}  step {step}  fill {fill * 100:.0f} %",
        )
        shutil.copyfile(path, out / "latest3d.png")

    def log_block(step, fill, voids):
        (nv, vv, _), (n25, c25, m25), (n10, c10, m10) = voids[0.5], voids[0.25], voids[0.1]
        monitor.log(
            step=step,
            fill=fill,
            front=porous3d.invasion_front(state, solid_np),
            voids=nv,
            void_cells=vv,
            voids25=n25,
            cells25=c25,
            mass25=m25,
            voids10=n10,
            cells10=c10,
            mass10=m10,
        )

    def gifs():
        for pat, gif_path in ((frames, out / f"uf_{args.name}.gif"), (frames3d, out / f"uf3d_{args.name}.gif")):
            paths = sorted(pat.glob("frame*.png"))
            if not paths:
                continue
            if len(paths) > 200:
                idx = np.unique(np.linspace(0, len(paths) - 1, 200).round().astype(int))
                paths = [paths[i] for i in idx]
            viz.write_gif(paths, gif_path, args.fps)

    # ---- main loop ------------------------------------------------------------
    start = time.perf_counter()
    step, tail = step0, None
    finish_type, finish_detail = "step_cap", f"hit the --steps safety cap ({args.steps})"
    fill = fill_fraction(state)
    voids, vmask, mist = find_voids(state, solid_np, args.min_void)
    log_block(step, fill, voids)
    frame(step, fill, voids, vmask, mist)
    frame3d(step, fill)
    fw.append(step, state)
    blocks_done = 0
    while step < args.steps:
        state = runner(state)
        step += args.block
        blocks_done += 1
        fill = fill_fraction(state)
        voids, vmask, mist = find_voids(state, solid_np, args.min_void)
        log_block(step, fill, voids)
        frame(step, fill, voids, vmask, mist)
        frame3d(step, fill)
        fw.append(step, state)
        if not np.isfinite(porous3d.saturation(state, solid_np)):
            finish_type, finish_detail = "diverged", f"non-finite saturation at step {step}"
            break
        if tail is None and fill >= args.fill_target:
            tail = args.extra_blocks  # array filled; let trailing pockets settle
            monitor.checkpoint(step, fill, fR=np.asarray(state.fR), fB=np.asarray(state.fB))
        elif tail is not None:
            tail -= 1
            if tail <= 0:
                finish_type, finish_detail = (
                    "filled",
                    f"array fill >= {args.fill_target:g} + {args.extra_blocks} settle blocks at step {step}",
                )
                break
        if args.ckpt_every and blocks_done % args.ckpt_every == 0:
            monitor.checkpoint(step, fill, fR=np.asarray(state.fR), fB=np.asarray(state.fB))
            gifs()
    elapsed = time.perf_counter() - start

    monitor.checkpoint(step, fill, fR=np.asarray(state.fR), fB=np.asarray(state.fB))
    fw.close()
    gifs()
    meta.finish(step, finish_type=finish_type, finish_detail=finish_detail)

    rs = monitor.rows
    (out / "report.md").write_text(
        f"# {title}\n\n"
        f"- Geometry {geom} | {info['n_balls']} balls | wall margin {info['wall_margin']} cells | "
        f"domain {nz}x{ny}x{nx} | gap porosity {info['gap_porosity']:.3f}\n"
        f"- Capillary Zou-He dP=0 inlet/outlet, theta {args.theta:.0f} deg all walls, sigma {args.sigma:g}, "
        f"beta {args.beta:g}, M = {args.m_ratio:g} (nu_red {nu_red:g} / nu_blue {args.nu_blue:g}), fp32\n"
        f"- Steps {step} | runtime {elapsed / 3600:.2f} h this segment ({step / max(elapsed, 1e-9):.0f} steps/s) | "
        f"log/frames/h5 interval {args.block} steps\n"
        f"- Finished: **{finish_type}** — {finish_detail}\n\n"
        f"## Result\n\n"
        f"- Array fill {rs[0]['fill']:.3f} -> {rs[-1]['fill']:.3f}\n"
        f"- Trapped voids (fB>0.5): {rs[-1]['voids']} ({rs[-1]['void_cells']} cells) | "
        f"mist fB>0.25: {rs[-1]['voids25']} ({rs[-1]['cells25']} cells, mass {rs[-1]['mass25']:.1f}) | "
        f"fB>0.1: {rs[-1]['voids10']} ({rs[-1]['cells10']} cells, mass {rs[-1]['mass10']:.1f})\n"
        f"- Artifacts: frames/ + uf_{args.name}.gif (top view), frames3d/ + uf3d_{args.name}.gif (PyVista), "
        f"metrics.csv, uf_{args.name}.h5/.xdmf ({args.field_vars})\n",
        encoding="utf-8",
    )
    print(
        f"{args.name}: fill {rs[0]['fill']:.3f} -> {rs[-1]['fill']:.3f}, voids {rs[-1]['voids']} "
        f"({rs[-1]['void_cells']} cells) at step {step} ({finish_type}); "
        f"{step / max(elapsed, 1e-9):.0f} steps/s; output: {out}"
    )


if __name__ == "__main__":
    main()
