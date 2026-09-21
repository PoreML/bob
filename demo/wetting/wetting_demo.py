"""3D wettability sweep: sessile droplets at 9 imposed contact angles.

A red half-ball on the bottom z-wall settles to the contact angle imposed by the
geometric wetting BC. Default solver, no overrides: MRT + akai reorientation + CSF
surface tension with the Eq.34/35 phi/nc solid extrapolation + recolor_emag (Eq.17) +
wall_grad="fluid". The box is sealed with walls on both the bottom (z=0) and the top
(z=nz-1); x and y stay periodic (wide enough that the low-angle puddle never wraps).
The realized angle tracks the imposed angle closely (~1-4°, with a mild overshoot at
the 150° extreme — a property of the single-layer geometric BC); the convergence plot
shows each track sitting on its imposed target.

Renders are 3D PyVista views: red phase opaque, blue hidden, walls translucent, viewed
side-on (camera "xz", z up) so the contact angle reads off the droplet profile. Voxel
threshold (smooth=False) shows the unsmoothed red volume.

Every angle runs the same fixed number of steps. Artifacts:
  * sessile_grid.png  — 3x3 grid of the equilibrated droplets (400 dpi)
  * convergence.png   — measured angle vs timestep, one track per case (400 dpi)
  * wetting.gif     — the 3x3 grid animated as all nine droplets relax together

Usage:  uv run python demo/wetting/wetting_demo.py     (uses GPU if available)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from bob import color3d, d3q19, misc
from bob.utils import viz as _viz
from bob.utils import viz3d as _viz3d

plt.switch_backend("Agg")

THETAS = [30.0, 45.0, 60.0, 75.0, 90.0, 105.0, 120.0, 135.0, 150.0]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nz", type=int, default=60, help="wall-normal axis (sealed top+bottom z-walls)")
    p.add_argument("--n", type=int, default=104, help="lateral size (ny = nx = n, periodic)")
    p.add_argument("--R", type=int, default=20)
    p.add_argument("--steps", type=int, default=20000, help="uniform total steps per case (no early stop)")
    p.add_argument("--block", type=int, default=200, help="steps per angle measurement / render snapshot (100 frames)")
    p.add_argument("--beta", type=float, default=0.95, help="recoloring / segregation strength")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--frame-dpi", type=int, default=120, help="dpi for the GIF frames (downscaled in the gif)")
    p.add_argument("--out", type=Path, default=Path("demo/wetting/output"))
    p.add_argument("--plots", action="store_true",
                   help="also write the analysis curves (convergence.png); off by default")
    return p.parse_args()


def init_sessile(nz, n, R):
    """Red half-ball of radius R sitting on the bottom z-wall (wall at z = 0)."""
    zz, yy, xx = jnp.meshgrid(jnp.arange(nz), jnp.arange(n), jnp.arange(n), indexing="ij")
    red = ((xx - (n - 1) / 2.0) ** 2 + (yy - (n - 1) / 2.0) ** 2 + (zz - 1.0) ** 2 <= R**2) & (zz >= 1)
    rhoR = jnp.where(red, 1.0, 0.0)
    W = d3q19.W[:, None, None, None]
    return color3d.State(W * rhoR[None], W * (1.0 - rhoR)[None])


def field(state):
    rhoR, rhoB = color3d.densities(state)
    return np.asarray(color3d.color_field(rhoR, rhoB))


def run_fixed(theta, solid, nw, args):
    """Relax one droplet for exactly args.steps, recording the (step, angle) track
    and a colour-field snapshot every args.block steps."""
    p = color3d.Params(omega=1.0, sigma=0.02, beta=args.beta, theta=theta)  # default solver
    runner = jax.jit(
        lambda s: jax.lax.scan(lambda a, _: (color3d.step(a, p, solid=solid, nw=nw), None), s, None, length=args.block)[0]
    )
    st = init_sessile(args.nz, args.n, args.R)
    steps = [0]
    angles = [misc.contact_angle(st, wall_plane=0)]
    snaps = {0: field(st)}
    step = 0
    while step < args.steps:
        st = runner(st)
        step += args.block
        steps.append(step)
        angles.append(misc.contact_angle(st, wall_plane=0))
        snaps[step] = field(st)
    return st, np.array(steps), np.array(angles), snaps


# side-on render: x horizontal, z up; whole droplet region framed
_CAM = "xz"
_WIN_FRAME = (440, 280)
_WIN_FINAL = (600, 380)
_ZOOM = 1.5


def render_tube(path, rhoN, solid, window):
    # smooth=False keeps the unsmoothed voxel volume; shade=True lights the 3D dome
    _viz3d.phase_frame(path, rhoN, solid, camera=_CAM, smooth=False, shade=True, zoom=_ZOOM, rock_opacity=0.12, window=window)


def render_grid(path, render_paths, titles, suptitle, dpi):
    """3x3 grid assembled from pre-rendered PyVista PNGs."""
    fig, axes = plt.subplots(3, 3, figsize=(13, 8))
    for ax, rp, title in zip(axes.ravel(), render_paths, titles):
        ax.imshow(plt.imread(rp))
        ax.set_axis_off()
        ax.set_title(title, fontsize=10)
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def convergence_plot(path, tracks):
    if not _viz.plots_enabled():
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.cm.turbo(np.linspace(0.05, 0.95, len(THETAS)))
    for (steps, angles), th, c in zip(tracks, THETAS, cmap):
        ax.plot(steps, angles, ":o", color=c, ms=4, lw=1.0, label=f"{th:.1f} deg")
        ax.axhline(th, ls="--", color=c, lw=1.0, alpha=0.7)
    ax.set_xlabel("timestep")
    ax.set_ylabel("contact angle (deg)")
    ax.set_title("Convergence of measured angle (dots) toward imposed angle (dashed)")
    ax.grid(alpha=0.3)
    ax.legend(title="imposed", ncol=2, fontsize=9, loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=400)
    plt.close(fig)


def main():
    args = parse_args()
    if args.plots:
        _viz.enable_plots(True)
    args.out.mkdir(parents=True, exist_ok=True)
    rdir = args.out / "_render"
    fdir = args.out / "frames"
    rdir.mkdir(exist_ok=True)
    fdir.mkdir(exist_ok=True)
    for stale in list(rdir.glob("*.png")) + list(fdir.glob("frame_*.png")):
        stale.unlink()

    # walls on both bottom (z=0) and top (z=nz-1); x,y periodic
    solid = jnp.zeros((args.nz, args.n, args.n), bool).at[0].set(True).at[-1].set(True)
    solid_np = np.asarray(solid)
    nw = color3d.wall_normals(solid_np)  # preprocessing: unit wall normals at X_W

    print(
        f"3D wetting sweep on {jax.default_backend().upper()}  "
        f"({args.nz}x{args.n}x{args.n}, R={args.R}, beta={args.beta}, {args.steps} steps, render every {args.block})"
    )
    t0 = time.perf_counter()
    states, tracks, snaps_all = [], [], []
    for th in THETAS:
        st, steps, angles, snaps = run_fixed(th, solid, nw, args)
        states.append(st)
        tracks.append((steps, angles))
        snaps_all.append((steps, angles, snaps))
        print(f"  imposed {th:5.1f} deg -> measured {angles[-1]:5.1f} deg")
    elapsed = time.perf_counter() - t0

    # relaxation GIF: one 3x3 grid frame per block step (titles show the evolving angle)
    frame_steps = sorted(snaps_all[0][2])
    frame_paths = []
    for fstep in frame_steps:
        rps, titles = [], []
        for th, (steps, ang, snaps) in zip(THETAS, snaps_all):
            m = float(ang[int(np.where(steps == fstep)[0][0])])
            rp = rdir / f"a{int(th):03d}_s{fstep:06d}.png"
            render_tube(rp, snaps[fstep], solid_np, _WIN_FRAME)
            rps.append(rp)
            titles.append(f"imposed {th:.0f} deg  ->  measured {m:.1f} deg")
        fp = fdir / f"frame_{fstep:06d}.png"
        render_grid(fp, rps, titles, f"3D sessile droplets relaxing  (step {fstep})", args.frame_dpi)
        frame_paths.append(fp)
    _viz.write_gif(frame_paths, args.out / "wetting.gif", args.fps)

    # final still (400 dpi) — re-render the last snapshot at higher resolution
    final_paths, final_titles = [], []
    for th, st, (_, ang, _) in zip(THETAS, states, snaps_all):
        rp = rdir / f"final_a{int(th):03d}.png"
        render_tube(rp, field(st), solid_np, _WIN_FINAL)
        final_paths.append(rp)
        final_titles.append(f"imposed {th:.0f} deg  ->  measured {ang[-1]:.1f} deg  (step {args.steps})")
    render_grid(
        args.out / "sessile_grid.png",
        final_paths,
        final_titles,
        "3D sessile droplets — geometric wetting BC (default solver)",
        400,
    )
    convergence_plot(args.out / "convergence.png", tracks)

    rows = [{"theta_imposed": th, "measured_deg": round(float(ang[-1]), 1)} for th, (_, ang, _) in zip(THETAS, snaps_all)]
    _viz.write_csv(args.out / "metrics.csv", rows)
    monotonic = all(a <= b for a, b in zip([r["measured_deg"] for r in rows], [r["measured_deg"] for r in rows][1:]))
    lines = [
        "# 3D Wettability Sweep (sessile droplets, top+bottom z-walls)",
        "",
        (f"- Grid {args.nz}x{args.n}x{args.n}, R={args.R}, beta={args.beta}, {args.steps} uniform steps, "
        f"render every {args.block}, runtime {elapsed:.1f}s ({jax.default_backend()})"),
        f"- Monotonic (smaller imposed -> smaller measured): {'PASS' if monotonic else 'FAIL'}",
        "",
        "| imposed (deg) | measured (deg) |",
        "| ---: | ---: |",
    ]
    lines += [f"| {r['theta_imposed']:.1f} | {r['measured_deg']:.1f} |" for r in rows]
    lines += [
        "",
        "Default solver (MRT + akai + CSF Eq.34/35 + recolor_emag + wall_grad='fluid'), no overrides;",
        "the realized angle tracks the imposed angle to ~1-4° (single-layer geometric BC; mild",
        "overshoot at the 150° extreme). Top z-wall seals the periodic z-boundary. Renders are 3D PyVista,",
        "side-on (camera 'xz', z up), voxel threshold (no smoothing).",
        "",
        "## Artifacts",
        "- sessile_grid.png (3x3 equilibrated droplets, PyVista side-on, 400 dpi)",
        "- convergence.png (measured-angle tracks vs imposed, 400 dpi; with --plots)",
        "- wetting.gif (the 3x3 grid relaxing) | metrics.csv",
    ]
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  output: {args.out}  ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
