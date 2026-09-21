"""3D layered Poiseuille — pressure-driven validation of the default color3d stack.

Three-band channel (viscous blue against both z-walls, thin red in the center),
driven along x by per-color Zou-He pressure boundaries (not a body force): inlet
x=0 at rho_in = 1 + drho with a per-cell sa_red mask feeding each layer its own
fluid, outlet x=-1 at rho_out = 1 with the zero-gradient color split. At steady
state u_x(z) is the analytic layered-Poiseuille profile: nearly flat in the
viscous wall layers, a tall parabola in the thin center band.

The wall fluid viscosity is fixed (nu_b = 0.3, tau = 1.4) and the center fluid
is thinned to realize each ratio M = nu_b/nu_r; all ratios share the same
pressure drop (set so the largest M peaks near u_max), so in the combined
figure the wall segments coincide and the center peak fans upward with M.

The theory curve solves d/dz(nu du/dz) = -G on the realized diffuse interface
(harmonic face viscosities from the measured phi at mid-x) with the realized
pressure gradient G (linear fit of cs2*rho(x)). One combined
figure (theory lines + LBM dots, one color per M) and one combined GIF.

Usage:  uv run python demo/layered_poiseuille/layered_demo.py            # all ratios
        uv run python demo/layered_poiseuille/layered_demo.py --ratios 5 --steps 8000 --block 2000
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
from matplotlib.lines import Line2D

from bob import bc, color3d, d3q19
from bob.utils import viz as _viz

plt.switch_backend("Agg")

# Plot style: categorical palette in fixed order (one color per M, line and dots
# share it), dark ink text colors, recessive grid, mathtext.
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update(
    {
        # publication sizing: readable after column-width shrink, without crowding
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

CS2 = float(d3q19.CS2)
NU_B = 0.3  # fixed wall-fluid viscosity (tau = 1.4); nu_r = NU_B / M


def analytic_layered(nz, nu_z, g):
    """u(z) solving d/dz(nu du/dz) = -g, u=0 at wall nodes, harmonic face viscosities."""
    A, b = np.zeros((nz, nz)), np.zeros(nz)
    A[0, 0] = A[-1, -1] = 1.0
    for j in range(1, nz - 1):
        nu_m = 2 * nu_z[j] * nu_z[j - 1] / (nu_z[j] + nu_z[j - 1])
        nu_p = 2 * nu_z[j] * nu_z[j + 1] / (nu_z[j] + nu_z[j + 1])
        A[j, j - 1], A[j, j + 1], A[j, j] = nu_m, nu_p, -(nu_m + nu_p)
        b[j] = -g
    return np.linalg.solve(A, b)


def run_sim(m_ratio, args, drho):
    """Run one ratio to steady state; return theory, profile history, and metrics."""
    nz, ny, nx = args.nz, args.ny, args.nx
    nu_r = NU_B / m_ratio
    omega_r, omega_b = 1.0 / (3.0 * nu_r + 0.5), 1.0 / (3.0 * NU_B + 0.5)
    zz = np.arange(nz)
    band_lo, band_hi = nz // 4, 3 * nz // 4
    band = (zz >= band_lo) & (zz < band_hi)

    solid = jnp.zeros((nz, ny, nx), bool).at[0].set(True).at[-1].set(True)
    rhoR = jnp.broadcast_to(jnp.asarray(band, float)[:, None, None], (nz, ny, nx))
    rhoR = jnp.where(solid, 0.0, rhoR)
    W = d3q19.W[:, None, None, None]
    st = color3d.State(W * rhoR[None], W * (1.0 - rhoR)[None])
    sa_mask = jnp.broadcast_to(jnp.asarray(band, float)[:, None], (nz, ny))
    p = color3d.Params(omega=omega_r, sigma=0.0, beta=0.7, omega2=omega_b)

    def one(s):
        s = color3d.step(s, p, solid=solid)
        s = bc.zou_he_inlet(s, solid=solid, rho_in=1.0 + drho, sa_red=sa_mask)
        return bc.zou_he_outlet(s, solid=solid, rho_out=1.0)

    runner = jax.jit(lambda s: jax.lax.scan(lambda a, _: (one(a), None), s, None, length=args.block)[0])

    def mid_profile(s):
        return np.asarray(color3d.velocity(s)[0, :, ny // 2, nx // 2])

    start = time.perf_counter()
    hist = [(0, mid_profile(st))]
    prev, steps_run = hist[0][1], 0
    for it in range(1, args.steps // args.block + 1):
        st = runner(st)
        steps_run = it * args.block
        ux = mid_profile(st)
        hist.append((steps_run, ux))
        if np.max(np.abs(ux - prev)) < args.rtol * max(float(np.max(np.abs(ux))), 1e-30):
            break
        prev = ux
    elapsed = time.perf_counter() - start
    ux = hist[-1][1]

    # realized pressure gradient: linear fit of cs2 * <rho>(x) over interior planes
    rho_x = np.asarray((st.fR + st.fB).sum(0)[1:-1].mean(axis=(0, 1)))
    xs = np.arange(nx)[3 : nx - 3]
    g_real = -CS2 * float(np.polyfit(xs, rho_x[3 : nx - 3], 1)[0])

    # realized diffuse-interface viscosity profile at mid-x
    rhoR_f, rhoB_f = color3d.densities(st)
    phi = np.asarray(color3d.color_field(rhoR_f, rhoB_f))[:, :, nx // 2].mean(axis=1)
    psi = np.clip((1.0 + phi) / 2.0, 0.0, 1.0)
    nu_z = 1.0 / (psi / nu_r + (1.0 - psi) / NU_B)
    u_an = analytic_layered(nz, nu_z, g_real)

    fluid = slice(1, nz - 1)
    rel = float(np.max(np.abs(ux[fluid] - u_an[fluid])) / np.max(np.abs(u_an[fluid])))
    return {
        "M": m_ratio,
        "nu_r": nu_r,
        "omega_r": omega_r,
        "g_real": g_real,
        "steps": steps_run,
        "seconds": elapsed,
        "rel": rel,
        "umax": float(np.max(ux)),
        "hist": hist,
        "u_an": u_an,
    }


def held(hist, step):
    """Profile at `step`, holding the last available block for finished runs."""
    out = hist[0][1]
    for s, pr in hist:
        if s > step:
            break
        out = pr
    return out


def draw_all(ax, zz, cases, band_lo, band_hi, ymax, note, at_step=None):
    """Combined axes: per-M theory line + LBM dots in the same palette color."""
    nz = len(zz)
    ax.axvspan(band_lo - 0.5, band_hi - 0.5, color=INK, alpha=0.05, lw=0)
    handles = []
    for c, col in zip(cases, PALETTE):
        ux = c["hist"][-1][1] if at_step is None else held(c["hist"], at_step)
        ax.plot(zz, c["u_an"], color=col, lw=2.4, zorder=2)
        ax.plot(zz[1:-1:2], ux[1:-1:2], "o", color=col, ms=7, mec="white", mew=1.2, zorder=3)
        handles.append(Line2D([], [], color=col, lw=2.4, marker="o", ms=7, mec="white", label=rf"$M = {c['M']}$"))
    ax.set_xlim(0, nz - 1)
    ax.set_ylim(0, ymax)
    ax.set_xlabel(r"$z$  (lattice units, walls at both ends)")
    ax.set_ylabel(r"$u_x$")
    ax.set_title(r"Layered Poiseuille (3D):  $M \equiv \nu_{\mathrm{b}}/\nu_{\mathrm{r}}$,  fixed $\nu_{\mathrm{b}}$", pad=10)
    ax.text(0.5 * band_lo, 0.12 * ymax, r"viscous ($\nu_{\mathrm{b}}$)", ha="center", color=INK2, fontsize=13)
    ax.text(0.5 * (band_lo + band_hi), 0.02 * ymax, r"thin fluid ($\nu_{\mathrm{r}}$)", ha="center", color=INK2, fontsize=13)
    ax.text(0.98, 0.97, note, transform=ax.transAxes, ha="right", va="top", fontsize=13, color=INK2)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.legend(handles=handles, loc="upper left", frameon=False, borderaxespad=1.0)


def render(path, zz, cases, band_lo, band_hi, ymax, note, at_step=None, dpi=None):
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    draw_all(ax, zz, cases, band_lo, band_hi, ymax, note, at_step)
    fig.tight_layout()
    fig.savefig(path, **({"dpi": dpi} if dpi else {}))
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ratios", type=int, nargs="*", default=[5, 10, 15, 20, 30], help="viscosity ratios M = nu_b/nu_r")
    p.add_argument("--nz", type=int, default=64)
    p.add_argument("--ny", type=int, default=8)
    p.add_argument("--nx", type=int, default=128)
    p.add_argument("--umax", type=float, default=0.025, help="target peak velocity at the largest ratio (sets drho)")
    p.add_argument("--steps", type=int, default=600000, help="step cap per ratio")
    p.add_argument("--block", type=int, default=2000, help="scan length per jitted block")
    p.add_argument("--frame-every", type=int, default=5, help="GIF frame cadence in blocks")
    p.add_argument("--rtol", type=float, default=1e-6, help="steady-state stop: max profile change per block")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--out", type=Path, default=Path("demo/layered_poiseuille/output"))
    p.add_argument("--plots", action="store_true",
                   help="also write the analysis curves (profile.svg); off by default")
    return p.parse_args()


def main():
    args = parse_args()
    if args.plots:
        _viz.enable_plots(True)
    frames_dir = args.out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in frames_dir.glob("frame_*.png"):
        stale.unlink()
    nz = args.nz
    zz = np.arange(nz)
    band_lo, band_hi = nz // 4, 3 * nz // 4
    band = (zz >= band_lo) & (zz < band_hi)

    # one pressure drop for all ratios, set so the largest M peaks near --umax
    m_max = max(args.ratios)
    u_g1 = analytic_layered(nz, np.where(band, NU_B / m_max, NU_B), 1.0)
    drho = (args.umax / float(u_g1.max())) * (args.nx - 1) / CS2

    cases = [run_sim(m, args, drho) for m in sorted(args.ratios)]
    for c in cases:
        status = "PASS" if c["rel"] < 0.08 else "FAIL"
        print(f"M={c['M']}: {status}  max rel err {c['rel']:.2%}  ({c['steps']} steps, {c['seconds']:.0f}s)")

    ymax = 1.22 * max(float(max(c["u_an"].max(), c["hist"][-1][1].max())) for c in cases)
    note = "lines — analytic\ndots — LBM"
    if _viz.plots_enabled():  # analysis curve: opt-in via --plots (the GIF frames below stay unconditional)
        render(args.out / "profile.svg", zz, cases, band_lo, band_hi, ymax, note)

    stride = args.block * args.frame_every
    grid = range(0, max(c["steps"] for c in cases) + 1, stride)
    frame_paths = []
    for t in grid:
        path = frames_dir / f"frame_{t:06d}.png"
        render(path, zz, cases, band_lo, band_hi, ymax, note + f"\nstep {t:,}", at_step=t, dpi=110)
        frame_paths.append(path)
    _viz.write_gif(frame_paths, args.out / "layered.gif", args.fps)

    lines = [
        "# 3D Layered Poiseuille (pressure-driven, default MRT+CSF stack)",
        "",
        f"- Grid: {args.nz}x{args.ny}x{args.nx} (walls z=0/{args.nz - 1}, y periodic, Zou-He pressure x=0/x=-1)",
        (
            f"- nu_b = {NU_B} fixed (wall layers), nu_r = nu_b/M (center band) | shared drho = {drho:.3e} "
            f"(u_max target {args.umax} at M = {m_max})"
        ),
        "",
        "| M | nu_r | omega_r | G (realized) | steps | runtime | u_max | max rel err | status |",
        "|--:|-----:|--------:|-------------:|------:|--------:|------:|------------:|:------:|",
    ]
    for c in cases:
        status = "PASS" if c["rel"] < 0.08 else "FAIL"
        lines.append(
            f"| {c['M']} | {c['nu_r']:.4f} | {c['omega_r']:.3f} | {c['g_real']:.3e} "
            f"| {c['steps']} | {c['seconds']:.0f}s | {c['umax']:.4f} | {c['rel']:.2%} | {status} |"
        )
    lines += ["", "Artifacts: `profile.svg` (all ratios: theory lines vs LBM dots), `layered.gif` (development).", ""]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"report: {args.out / 'report.md'}")


if __name__ == "__main__":
    main()
