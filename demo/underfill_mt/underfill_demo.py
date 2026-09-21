"""Capillary underfill of the MT6225A flip-chip package (paper case 3, Wang et al.,
Microelectron. Eng. 2016, PII S0167931715300575).

I-type dispensation: encapsulant (red, wetting, the more viscous phase) is seeded
along the low-x edge of the voxelized 0.21 mm die/substrate gap (`geometry.py`) and
imbibes across the 264-ball array under capillary suction alone — Zou-He ΔP=0
pressure inlet and outlet (the capillary-driven BC pair validated in
demo/washburn; `porous3d.drain(inlet="zouhe")`), no imposed velocity or pressure drop.

Setup (following the paper wherever the solver allows):
  * geometry   : datasheet-exact MT6225A ball map, truncated-sphere bumps, 0.21 mm gap
  * dispensing : I-type, instantaneous (paper Fig. 5) — red slug along the low-x edge
  * drive      : capillary action only (Zou-He dP=0 pressure inlet + outlet)
  * wettability: case 3 gives substrate/die/bump the same interaction strength
                 (g^w = -1.8/-1.8/-1.8, paper Table 3), so one global theta is faithful
  * comparison : the paper's Fig. 8 snapshot is at 42 % fill — state42.npz freezes it
Deliberate deviations: bob's color-gradient MRT+akai+CSF stack replaces the paper's
Shan-Chen GIPM + LBGK (the GIPM g^w has no exact theta mapping — theta 30 deg stands
in for a strongly wetting encapsulant), and the viscosity ratio M = nu_red/nu_blue = 1.5
is a lattice-stability compromise (real epoxy/air ~ 1e4).

The observable to compare against Fig. 8: the bump region fills slightly ahead of
the depopulated center channel, because the ball surfaces add wetted area that
pulls the meniscus forward.

Artifacts (demo/underfill_mt/output/run/): frames/ top-view melt-front maps + GIF,
metrics.csv, run_meta.json, report.md; with --plots also fill_curve.png and
front_profile.png (x_front(y) at the 42 % snapshot).

Usage:  uv run python demo/underfill_mt/underfill_demo.py            # GPU if available
        uv run python demo/underfill_mt/underfill_demo.py --steps 2000 --block 200  # smoke
        uv run python demo/underfill_mt/underfill_demo.py --margin 0.6  # paper Fig. 4 flow margin
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax
import matplotlib.pyplot as plt
import numpy as np
from geometry import DX_MM, DX_MM_HIRES, build_gap_mask, load_package

from bob import color3d, d3q19, porous3d
from bob.utils import stats, viz
from bob.utils.file import RunMeta

HERE = Path(__file__).resolve().parent


def nu(omega):
    """Kinematic viscosity of a relaxation rate: nu = cs^2 (1/omega - 1/2)."""
    return float(d3q19.CS2) * (1.0 / omega - 0.5)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--hires",
        action="store_true",
        help="2x-resolution preset: dx 0.015 mm (14 gap layers, ball dia 20 cells = 4-5 interface widths), "
        "buffer 20, steps cap 400k, out output/hires. This is the void-resolving configuration — at the "
        "ordinary dx 0.03 wake bubbles dissolve through the diffuse interface; at 0.015 they persist "
        "(mid-gap, on ball downstream faces, advected with the melt). ~8x cells, ~9x slower (20 steps/s "
        "on an H200, ~2.5 h to 45 %% fill vs ~9 min ordinary). Explicit --dx/--buffer/--steps/--out win",
    )
    p.add_argument("--dx", type=float, default=DX_MM, help="lattice pitch in mm (0.03 -> 7 cells across the gap)")
    p.add_argument(
        "--theta",
        type=float,
        default=30.0,
        help="encapsulant contact angle on all walls (deg, <90 wetting; "
        "typical underfill epoxies wet at ~10-30 deg, and cos(theta) sets the fill speed)",
    )
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--beta", type=float, default=0.95)
    p.add_argument("--omega", type=float, default=1.538, help="red (encapsulant) relaxation: nu_red = 0.05")
    p.add_argument(
        "--omega2",
        type=float,
        default=1.0 / 0.6,
        help="blue (air) relaxation: nu_blue = 0.033 (M = 1.5 vs the default omega). "
        "The displaced-air drag through the long thin gap dominates the fill time, but a thinner air "
        "phase diverges here: nu_blue <= 0.02 (omega2 >= 1.786) goes NaN within 1000 steps in this "
        "7-cell gap at sigma = 0.05 (nu_blue = 0.025 is stable) — a tighter floor than in porous drainage",
    )
    p.add_argument(
        "--bump-shape",
        choices=("sphere", "cylinder"),
        default="sphere",
        help="solder joints as plate-truncated spheres (datasheet-faithful) or full-gap cylinders "
        "(no near-tangent plate wedges)",
    )
    p.add_argument("--buffer", type=int, default=10, help="open inlet/outlet slabs outside the chip footprint")
    p.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="flow-margin width in mm (paper Fig. 4): bump-free side strips along both y edges at gap height, "
        "enabling the edge detour flow around the chip side-edges; 0 = sealed side walls at the chip edge "
        "(default). The paper gives no width; ~0.5-1.0 mm is a reasonable range. Fill %% and the "
        "front profile always measure the chip footprint only",
    )
    p.add_argument("--fill-target", type=float, default=0.50, help="stop once the chip gap fill fraction passes this")
    p.add_argument("--steps", type=int, default=150_000, help="safety cap; the run stops at --fill-target/breakthrough")
    p.add_argument("--block", type=int, default=1000, help="steps per jitted block (metrics + frame cadence)")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--plots", action="store_true",
                   help="also write the analysis curves (fill_curve.png, front_profile.png, fill.svg); off by default")
    p.add_argument("--out", type=Path, default=None, help="output dir (default output/run, or output/hires with --hires)")
    p.add_argument("--resume", type=Path, default=None, help="checkpoint .npz (or dir: newest) to continue")
    p.add_argument("--ckpt-every", type=int, default=20, help="blocks between periodic checkpoints (0 = end only)")
    args = p.parse_args()
    if args.hires:  # preset fills only what the user left at defaults
        if args.dx == DX_MM:
            args.dx = DX_MM_HIRES
        if args.buffer == 10:
            args.buffer = 20
        if args.steps == 150_000:
            args.steps = 400_000
    if args.out is None:
        args.out = HERE / "output" / ("hires" if args.hires else "run")
    return args


def top_view(state, solid_np):
    """z-averaged color field over the pore columns: the melt-front map the paper
    shows. Returns (rhoN_top masked ndarray, fully-solid-column mask)."""
    rhoR, rhoB = (np.asarray(f) for f in color3d.densities(state))
    rhoN = np.asarray(color3d.color_field(rhoR, rhoB))
    pore = ~solid_np
    w = pore.sum(axis=0)
    col = (rhoN * pore).sum(axis=0) / np.maximum(w, 1)
    return np.ma.masked_where(w == 0, col), w == 0


def fill_fraction(state, solid_np, chip_x, chip_y):
    """Red-majority fraction of the chip-interior pore space (the paper's fill %) —
    chip footprint only, excluding buffers and any flow margin."""
    red = porous3d._red_pore(state, solid_np)[:, chip_y, chip_x]
    pore = ~solid_np[:, chip_y, chip_x]
    return float(red.sum() / max(pore.sum(), 1))


def front_profile(state, solid_np, chip_x, chip_y, xs_mm):
    """Furthest red-majority x (mm) per y column inside the chip — the melt-front
    shape x_front(y); NaN where a column holds no red yet."""
    red = porous3d._red_pore(state, solid_np)[:, chip_y, chip_x].any(axis=0)  # (ny_chip, nx_chip)
    idx = np.where(red, np.arange(red.shape[1])[None, :], -1).max(axis=1)
    xf = xs_mm[chip_x][np.maximum(idx, 0)].astype(float)
    xf[idx < 0] = np.nan
    return xf


def main():
    args = parse_args()
    if args.plots:
        viz.enable_plots(True)
    pkg = load_package()
    solid_np, info = build_gap_mask(
        pkg, dx=args.dx, buffer_in=args.buffer, buffer_out=args.buffer, bump_shape=args.bump_shape, margin_mm=args.margin
    )
    nz, ny, nx = solid_np.shape
    xs, ys, chip_x, chip_y = info["xs"], info["ys"], info["chip_x"], info["chip_y"]
    ext = (xs[0], xs[-1], ys[0], ys[-1])
    solid = jax.numpy.asarray(solid_np)
    nw = color3d.wall_normals(solid_np)

    out = args.out
    frames = out / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    if args.resume is None:
        for stale in frames.glob("frame_*.png"):
            stale.unlink()

    p = color3d.Params(omega=args.omega, omega2=args.omega2, sigma=args.sigma, beta=args.beta, theta=args.theta)
    n_in = args.buffer
    if args.resume is not None:
        ckpt_path = stats.latest_checkpoint(args.resume) if args.resume.is_dir() else args.resume
        if ckpt_path is None:
            sys.exit(f"no checkpoint found under {args.resume}")
        ck = np.load(ckpt_path)
        state = color3d.State(jax.numpy.asarray(ck["fR"]), jax.numpy.asarray(ck["fB"]))
        step0 = int(ck["step"])
        if args.steps <= step0:
            sys.exit(f"--steps {args.steps} <= checkpoint step {step0}: nothing to run (raise --steps)")
    else:
        state = porous3d.init_drainage(nz, ny, nx, n_red=n_in + 2)
        step0 = 0
    title = (
        f"MT6225A underfill (theta={args.theta:.0f} deg, sigma={args.sigma:g}, dx={args.dx:g} mm, "
        f"{args.bump_shape} bumps, capillary dP=0)"
    )
    meta = RunMeta(
        out / "run_meta.json",
        params=p,
        solid=solid_np,
        geometry_source="mt6225a_package_geometry.json",
        target_steps=args.steps,
        resume_step=step0,
        extra={
            "dx_mm": args.dx,
            "gap_porosity": info["gap_porosity"],
            "fill_target": args.fill_target,
            "inlet": "zouhe",
            "bump_shape": args.bump_shape,
            "margin_mm": args.margin,
        },
    )
    monitor = stats.RunMonitor(
        out,
        title=title,
        curve_field="fill",
        rows=stats.carry_rows(out, step0, fields=("step", "fill", "front")) if step0 else None,
        meta=meta,
    )
    runner = jax.jit(lambda s: porous3d.drain(s, p, solid, n_in, 0.0, args.block, nw=nw, inlet="zouhe"))

    def frame(step, fill):
        col, dead = top_view(state, solid_np)
        fig, ax = plt.subplots(figsize=(8, 7.2))
        ax.imshow(col, origin="lower", extent=ext, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
        ax.imshow(
            np.ma.masked_where(~dead, np.ones_like(col.data)),
            origin="lower",
            extent=ext,
            cmap="Greys",
            vmin=0,
            vmax=2.5,
            aspect="equal",
        )
        ax.set_title(f"{title}\nstep {step} — chip fill {fill * 100:.1f} % (red = encapsulant)")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        fig.tight_layout()
        path = frames / f"frame_{step:08d}.png"
        fig.savefig(path, dpi=110)
        fig.savefig(out / "latest.png", dpi=110)
        plt.close(fig)
        return path

    def fill_curve(rows):
        if not viz.plots_enabled():
            return
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([r["step"] for r in rows], [100 * r["fill"] for r in rows], "-o", ms=3)
        ax.axhline(42.0, color="gray", ls="--", lw=1, label="paper Fig. 8 snapshot (42 %)")
        ax.set_xlabel("step")
        ax.set_ylabel("chip gap fill (%)")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "fill_curve.png", dpi=130)
        plt.close(fig)

    start = time.perf_counter()
    step, fill = step0, fill_fraction(state, solid_np, chip_x, chip_y)
    monitor.log(step=step, fill=fill, front=porous3d.invasion_front(state, solid_np))
    frame(step, fill)
    snap42 = None  # (step, fill, front_profile) nearest the paper's 42 % comparison point
    reason = "step cap"
    blocks_done = 0
    while step < args.steps:
        state = runner(state)
        step += args.block
        blocks_done += 1
        fill = fill_fraction(state, solid_np, chip_x, chip_y)
        sat = porous3d.saturation(state, solid_np)  # NaN-propagating divergence sentinel
        monitor.log(step=step, fill=fill, front=porous3d.invasion_front(state, solid_np))
        frame(step, fill)
        if not np.isfinite(sat):
            reason = "DIVERGED (non-finite saturation)"
            break
        if snap42 is None and fill >= 0.42:
            snap42 = (step, fill, front_profile(state, solid_np, chip_x, chip_y, xs))
            # freeze the paper-comparison state for replotting/rendering without re-simulation
            np.savez(out / "state42.npz", fR=np.asarray(state.fR), fB=np.asarray(state.fB), step=step, fill=fill)
        if fill >= args.fill_target:
            reason = f"fill target {args.fill_target:.0%} reached"
            break
        if porous3d.breakthrough(state, solid_np, x=nx - 2):
            reason = "breakthrough (front reached vent edge)"
            break
        if args.ckpt_every and blocks_done % args.ckpt_every == 0:
            monitor.checkpoint(step, fill, fR=np.asarray(state.fR), fB=np.asarray(state.fB))
            fill_curve(monitor.rows)
    elapsed = time.perf_counter() - start
    monitor.checkpoint(step, fill, fR=np.asarray(state.fR), fB=np.asarray(state.fB))
    meta.finish(step)
    rows = monitor.rows
    fill_curve(rows)

    if viz.plots_enabled():
        prof = front_profile(state, solid_np, chip_x, chip_y, xs)
        ys_chip = ys[chip_y]  # profile is chip-footprint only; ys spans margin too
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(prof, ys_chip, "-", lw=1.2, label=f"final (step {step}, {fill * 100:.0f} %)")
        if snap42 is not None:
            ax.plot(snap42[2], ys_chip, "-", lw=1.2, color="tab:orange", label=f"at 42 % (step {snap42[0]})")
        span = 10.4 / 2 + 0.15
        for yb in (-span, span):
            ax.axhline(yb, color="gray", ls=":", lw=1)
        ax.text(ax.get_xlim()[0], span, " edge channel above", va="bottom", fontsize=8, color="gray")
        ax.set_xlabel("melt-front position x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title("Melt-front profile x_front(y)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "front_profile.png", dpi=130)
        plt.close(fig)

    frame_paths = sorted(frames.glob("frame_*.png"), key=lambda q: int(q.stem.split("_")[1]))
    if len(frame_paths) > 200:
        idx = np.unique(np.linspace(0, len(frame_paths) - 1, 200).round().astype(int))
        frame_paths = [frame_paths[i] for i in idx]
    viz.write_gif(frame_paths, out / "underfill.gif", args.fps)

    connected = porous3d.connected_to_inlet(state, solid_np)
    (out / "report.md").write_text(
        f"# {title}\n\n"
        f"- Geometry mt6225a_package_geometry.json | dx {args.dx:g} mm | domain {nz}x{ny}x{nx} "
        f"| gap porosity {info['gap_porosity']:.3f} | 264 {args.bump_shape} balls\n"
        f"- Capillary-driven (Zou-He dP=0 inlet+outlet), theta {args.theta:.0f} deg on all walls "
        f"(paper case 3: uniform wettability), sigma {args.sigma:g}, beta {args.beta:g}, "
        f"nu_red {nu(args.omega):.3f} / nu_blue {nu(args.omega2):.3f} "
        f"(encapsulant more viscous)\n"
        f"- Steps {step} | runtime {elapsed:.0f}s ({step / max(elapsed, 1e-9):.0f} steps/s) | stop: {reason}\n\n"
        f"## Result\n\n"
        f"- Chip fill {rows[0]['fill']:.3f} -> {fill:.3f}"
        f"{f' | 42 % snapshot at step {snap42[0]}' if snap42 else ''}\n"
        f"- Inlet-connected red fraction {connected:.4f}\n"
        f"- Artifacts: frames/ + underfill.gif (top-view melt front); fill_curve.png and front_profile.png with --plots\n",
        encoding="utf-8",
    )
    print(
        f"underfill: fill {rows[0]['fill']:.3f} -> {fill:.3f} at step {step} ({reason}); "
        f"{step / max(elapsed, 1e-9):.0f} steps/s; output: {out}"
    )


if __name__ == "__main__":
    main()
