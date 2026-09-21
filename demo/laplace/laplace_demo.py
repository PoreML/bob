"""Laplace-law demo — 3D color-gradient droplets on the D3Q19 lattice.

Three short 3D experiments, rendered in 3D with PyVista (red phase opaque,
blue hidden, walls translucent):
  1. rounding — an elongated red box pulls itself into a ball (rounding.gif)
  2. sessile  — a droplet on the bottom wall relaxes to its contact angle (sessile.gif)
  3. laplace  — dp vs 1/R across radii, one fit per sigma in a scan
     (default 0.01..0.05, the range the porous cases use), confirms
     dp = 2*sigma/R (plots/laplace_fit.svg)

Writes GIFs, plots, CSV metrics and a report to ``demo/laplace/output``.

Usage:
    uv run python demo/laplace/laplace_demo.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)  # must run before importing bob

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from bob import color3d, d3q19, misc
from bob.utils import viz as _viz
from bob.utils import viz3d as _viz3d

plt.switch_backend("Agg")

# Plot style (shared with demo/layered_poiseuille): categorical palette, dark ink
# text colors, recessive grid, mathtext, publication font sizes.
# SERIES = categorical colors in fixed order (adjacent pairs chosen for contrast and
# color-vision-deficiency safety).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.grid": True,
        "grid.color": INK,
        "grid.alpha": 0.10,
        "grid.linewidth": 0.6,
        "mathtext.fontset": "stixsans",
    }
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=80, help="rounding box size (n^3); sessile grid scales from this")
    p.add_argument("--steps", type=int, default=5_000, help="steps for the rounding and sessile runs")
    p.add_argument("--block", type=int, default=100, help="JAX scan block size / metric cadence")
    p.add_argument("--frame-interval", type=int, default=100, help="steps between saved GIF frames")
    p.add_argument("--fps", type=int, default=12, help="GIF frames per second")
    p.add_argument("--omega", type=float, default=1.0, help="relaxation rate omega (sets the viscosity)")
    p.add_argument("--sigma", type=float, default=0.02, help="surface-tension parameter")
    p.add_argument("--beta", type=float, default=0.7, help="recoloring strength")
    p.add_argument(
        "--theta",
        type=float,
        default=107.0,
        help="imposed sessile red contact angle in degrees (geometric wetting BC)",
    )
    p.add_argument(
        "--laplace-radii",
        type=float,
        nargs="+",
        default=None,
        help="droplet radii for the Laplace fit; default: 7 radii evenly spaced in 1/R from R=18 to R=7",
    )
    p.add_argument("--laplace-steps", type=int, default=2_500)
    p.add_argument(
        "--laplace-sigmas",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.03, 0.04, 0.05],
        help="surface tensions for the Laplace scan (one fitted line each)",
    )
    p.add_argument(
        "--only",
        choices=["all", "laplace"],
        default="all",
        help="'laplace' regenerates only the Laplace scan/plot, skipping the GIF phases",
    )
    p.add_argument("--out", type=Path, default=Path("demo/laplace/output"), help="output directory")
    p.add_argument("--plots", action="store_true",
                   help="also write the analysis curves (plots/laplace_fit.svg, plots/laplace_fit.png); off by default")
    return p.parse_args()


def make_runner(params: color3d.Params, solid, length: int, nw=None):
    """jit-compiled scan advancing the 3D state by ``length`` steps."""

    @jax.jit
    def run(state):
        state, _ = jax.lax.scan(
            lambda s, _: (color3d.step(s, params, solid=solid, nw=nw), None), state, None, length=length
        )
        return state

    return run


def rho_n(state) -> np.ndarray:
    """Full 3D color field rhoN, (nz, ny, nx)."""
    rhoR, rhoB = color3d.densities(state)
    return np.asarray(color3d.color_field(rhoR, rhoB))


def run_phase(state, params, solid, steps, block, frame_interval, frames_dir, tag, solid_np=None, nw=None):
    """Advance ``state`` in blocks, saving PyVista 3D frames; returns (state, frame_paths, rows)."""
    runners: dict[int, callable] = {}
    frame_paths: list[Path] = []
    rows: list[dict] = []

    def record(step: int) -> None:
        umax = float(jnp.abs(color3d.velocity(state)).max())
        rows.append({"step": step, "umax": umax})
        if step % frame_interval == 0 or step == steps:
            path = frames_dir / f"{tag}_{step:06d}.png"
            # smooth shaded isosurface (droplets are smooth analytic shapes), whole
            # domain box in frame, large label
            _viz3d.phase_frame(
                path, rho_n(state), solid=solid_np, text=f"{tag}  step {step}", smooth=True, shade=True, zoom=1.0,
                font_size=18,
            )
            frame_paths.append(path)

    step = 0
    record(step)
    while step < steps:
        length = min(block, steps - step)
        if length not in runners:  # setdefault would build (and discard) a runner every block
            runners[length] = make_runner(params, solid, length, nw=nw)
        runner = runners[length]
        state = runner(state)
        step += length
        record(step)
    return state, frame_paths, rows


def main() -> None:
    args = parse_args()
    if args.plots:
        _viz.enable_plots(True)
    frames_dir, plots_dir = args.out / "frames", args.out / "plots"
    frames_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    if args.only == "all":
        for stale in frames_dir.glob("*.png"):
            stale.unlink()

        # 1. rounding: elongated box -> ball
        params = color3d.Params(omega=args.omega, sigma=args.sigma, beta=args.beta)
        st = misc.init_box(args.n, args.n, args.n, d=args.n // 5, h=args.n // 4, w=args.n // 2)
        s0 = misc.sphericity(np.asarray(color3d.densities(st)[0]))
        st, round_frames, round_rows = run_phase(
            st, params, None, args.steps, args.block, args.frame_interval, frames_dir, "round"
        )
        s1 = misc.sphericity(np.asarray(color3d.densities(st)[0]))
        _viz.write_gif(round_frames, args.out / "rounding.gif", args.fps)

        # 2. sessile droplet relaxing on the bottom z-wall
        # scale the sessile grid from --n (n=40 -> nz/n/R = 26/40/9)
        n = args.n
        nz, R = max(20, round(n * 0.65)), max(6, round(n * 0.225))
        wet_params = color3d.Params(omega=args.omega, sigma=args.sigma, beta=args.beta, theta=args.theta)
        solid = jnp.zeros((nz, n, n), dtype=bool).at[0].set(True)
        nw = color3d.wall_normals(np.asarray(solid))  # preprocessing: unit wall normals at X_W
        zz, yy, xx = jnp.meshgrid(jnp.arange(nz), jnp.arange(n), jnp.arange(n), indexing="ij")
        red = ((xx - (n - 1) / 2.0) ** 2 + (yy - (n - 1) / 2.0) ** 2 + (zz - 1.0) ** 2 <= R**2) & (zz >= 1)
        rhoR = jnp.where(red, 1.0, 0.0)
        st2 = color3d.State(
            d3q19.W[:, None, None, None] * rhoR[None], d3q19.W[:, None, None, None] * (1.0 - rhoR)[None]
        )
        st2, sessile_frames, _ = run_phase(
            st2,
            wet_params,
            solid,
            args.steps,
            args.block,
            args.frame_interval,
            frames_dir,
            "sessile",
            solid_np=np.asarray(solid),
            nw=nw,
        )
        theta = misc.contact_angle(st2, wall_plane=0)
        _viz.write_gif(sessile_frames, args.out / "sessile.gif", args.fps)

    # 3. Laplace: dp vs 1/R for each sigma in the scan, radii evenly spaced in 1/R (the fit abscissa)
    radii = args.laplace_radii or (1.0 / np.linspace(1.0 / 18.0, 1.0 / 7.0, 7)).tolist()
    inv_R = 1.0 / np.array(radii, dtype=float)
    fits = []  # per sigma: (sigma, dPs, slope, intercept, r2, sigma_eff)
    laplace_rows = []
    for sig in args.laplace_sigmas:
        params_s = color3d.Params(omega=args.omega, sigma=sig, beta=args.beta)
        dPs = []
        for R_lap in radii:
            L = int(np.ceil(3 * R_lap + 8))
            stL = misc.init_sphere(L, L, L, R=R_lap)
            runner = make_runner(params_s, None, args.laplace_steps)
            stL = runner(stL)
            dPs.append(misc.pressure_jump(stL, R_lap))
        dPs = np.array(dPs)
        slope, intercept = np.polyfit(inv_R, dPs, 1)
        resid = dPs - (slope * inv_R + intercept)
        r2 = 1.0 - np.sum(resid**2) / np.sum((dPs - dPs.mean()) ** 2)
        fits.append((sig, dPs, slope, intercept, r2, slope / 2.0))
        laplace_rows += [
            {"sigma": sig, "R": r, "inv_R": ir, "dp": dp} for r, ir, dp in zip(radii, inv_R, dPs)
        ]
    _viz.write_csv(plots_dir / "laplace_fit.csv", laplace_rows)  # table view for the plot

    if _viz.plots_enabled():  # analysis curve: opt-in via --plots (the CSV above is always written)
        fit_x = np.linspace(0, inv_R.max() * 1.08, 50)
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        handles = []
        for (sig, dPs, slope, intercept, _r2, _se), col in zip(fits, SERIES):
            ax.plot(fit_x, slope * fit_x + intercept, color=col, lw=2.2, zorder=2)
            ax.plot(inv_R, dPs, "o", color=col, ms=8, mec="white", mew=1.2, zorder=3)
            handles.append(
                plt.Line2D([], [], color=col, lw=2.2, marker="o", ms=7, mec="white", mew=1.1, label=rf"$\sigma = {sig:g}$")
            )
        ax.set_xlim(0, fit_x.max())
        ax.set_ylim(0, max(f[2] * fit_x.max() + f[3] for f in fits) * 1.12)
        ax.set_xlabel(r"$1/R$  (lattice units$^{-1}$)")
        ax.set_ylabel(r"$\Delta p$")
        ax.set_title(r"Laplace law (3D):  $\Delta p = 2\sigma/R$", pad=10)
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        ax.legend(handles=handles[::-1], loc="upper left", frameon=False)  # descending sigma = top line first
        fig.tight_layout()
        fig.savefig(plots_dir / "laplace_fit.svg")
        fig.savefig(plots_dir / "laplace_fit.png", dpi=150)
        plt.close(fig)

    elapsed = time.perf_counter() - start

    laplace_ok = all(r2 > 0.99 and 0.7 * sig < sig_eff < 1.4 * sig for sig, _, _, _, r2, sig_eff in fits)
    laplace_lines = [
        f"- Laplace sigma={sig:g}: sigma_eff={sig_eff:.4f} ({sig_eff / sig:.2f}x, R^2={r2:.4f})"
        for sig, _, _, _, r2, sig_eff in fits
    ]
    if args.only == "all":
        _viz.write_csv(args.out / "metrics.csv", round_rows)
        round_ok = s1 < 1.15
        angle_ok = abs(theta - args.theta) < 12.0  # geometric BC: measured tracks imposed
        status = "PASS" if (round_ok and angle_ok and laplace_ok) else "FAIL"
        report = [
            "# Droplet3D Demo Report",
            "",
            f"- Rounding box: {args.n}^3, steps {args.steps} | omega {args.omega}, sigma {args.sigma}, beta {args.beta}",
            f"- Runtime: {elapsed:.1f} s",
            f"- Overall: {status}",
            "",
            "## Checks",
            "",
            f"- Rounding: sphericity {s0:.3f} -> {s1:.3f} ({'PASS' if round_ok else 'FAIL'}: < 1.15)",
            (f"- Sessile angle: imposed {args.theta:.0f} deg, measured {theta:.1f} deg "
            f"({'PASS' if angle_ok else 'FAIL'}: within 12 deg)"),
            f"- Laplace scan ({'PASS' if laplace_ok else 'FAIL'}: each fit linear + sigma_eff within [0.7, 1.4]x):",
            *laplace_lines,
            "",
            "## Artifacts",
            "",
            "- rounding.gif, sessile.gif (PyVista 3D renders: red opaque, blue hidden, wall translucent)",
            "- plots/laplace_fit.csv (the fitted series; plots/laplace_fit.svg|png with --plots)",
            "- frames/*.png, metrics.csv",
        ]
        (args.out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        print(f"Droplet3D: {status}")
        print(f"  sphericity      : {s0:.3f} -> {s1:.3f}")
        print(f"  sessile angle   : imposed {args.theta:.0f} -> measured {theta:.1f} deg")
    else:
        print(f"Droplet3D (laplace only): {'PASS' if laplace_ok else 'FAIL'} ({elapsed:.1f} s)")
    for line in laplace_lines:
        print(f"  {line[2:]}")
    print(f"  output directory: {args.out}")


if __name__ == "__main__":
    main()
