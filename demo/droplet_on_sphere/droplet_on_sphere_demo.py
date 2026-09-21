"""Wetting on a curved 3D surface — droplet on a spherical solid (Akai et al. 2018, Fig. 9/10).

bob's default solver with the geometric wetting BC (Params.theta + wall normals,
default akai reorientation): a red droplet relaxes on a spherical solid object
until its shape matches the analytical solution. With droplet radius of
curvature R1, solid radius R2 and contact angle theta, the distance between
sphere centres follows the law of cosines

    R3 = sqrt(R1^2 + R2^2 - 2 R1 R2 cos(theta)),

so the analytical equilibrium droplet is the ball of radius R1 centred R3
above the solid centre. Initially the red fluid coats the upper hemisphere of
the solid as a half-shell collar ("part of a sphere", Fig. 10a) whose voxel
volume matches the analytical droplet's, and 30k steps relax it to the
spherical cap. Renders a 3D PyVista GIF and a 2D midplane-slice GIF with the
analytical circle overlaid as a white dotted line (Fig. 10 style) for each
contact angle, and validates R1, R3 and theta recovered from a least-squares
sphere fit of the free interface.

Note on R2: the paper's text says the solid radius is "fixed at 40 lattice
units", but with centre (50, 50, 30) in a 101^3 box that solid would poke out
of the floor and the theta=120 analytical droplet would top out at z ~ 107 —
outside the domain and unlike Fig. 10, where everything sits inside. The
figure's proportions (solid fully interior, droplet larger than solid) match
R2 = 20, so 20 is the default here; --R2 overrides it.

Usage:  uv run python demo/droplet_on_sphere/droplet_on_sphere_demo.py
        uv run python demo/droplet_on_sphere/droplet_on_sphere_demo.py --steps 4000 --block 500  # smoke
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import matplotlib

matplotlib.use("Agg")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from bob import color3d, d3q19
from bob.utils import viz as _viz
from bob.utils import viz3d as _viz3d


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=101, help="cubic domain size")
    p.add_argument("--R1", type=float, default=22.5, help="equilibrium droplet radius of curvature")
    p.add_argument("--R2", type=float, default=20.0, help="solid sphere radius (see module docstring re the paper's 40)")
    p.add_argument("--solid-z", type=float, default=30.0, help="z of the solid sphere centre")
    p.add_argument("--thetas", type=float, nargs="+", default=[60.0, 120.0], help="contact angles in degrees")
    p.add_argument("--steps", type=int, default=30000, help="relaxation steps per angle (paper: 30k to equilibrium)")
    p.add_argument("--block", type=int, default=500, help="JAX scan block size")
    p.add_argument("--frame-interval", type=int, default=500, help="steps between saved GIF frames")
    p.add_argument("--omega", type=float, default=1.0, help="relaxation rate omega (sets the viscosity)")
    p.add_argument("--sigma", type=float, default=0.02, help="surface-tension parameter")
    p.add_argument("--beta", type=float, default=0.7, help="recoloring strength")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--out", type=Path, default=Path("demo/droplet_on_sphere/output"))
    p.add_argument("--plots", action="store_true",
                   help="also write the analysis curves (plots/comparison.png); off by default")
    return p.parse_args()


def r3_analytic(R1, R2, theta_deg):
    """Centre-to-centre distance: law of cosines through the contact angle (Akai Fig. 9)."""
    th = np.deg2rad(theta_deg)
    return float(np.sqrt(R1**2 + R2**2 - 2.0 * R1 * R2 * np.cos(th)))


def make_runner(params, solid, length, nw):
    @jax.jit
    def run(state):
        state, _ = jax.lax.scan(lambda s, _: (color3d.step(s, params, solid=solid, nw=nw), None), state, None, length=length)
        return state

    return run


def rho_n(state):
    rhoR, rhoB = color3d.densities(state)
    return np.asarray(color3d.color_field(rhoR, rhoB))


def fit_interface_sphere(rhoN, solid_np, solid_c, R2):
    """LSQ sphere through the free red/blue interface (red-region boundary voxels
    more than 4 cells clear of the solid, where the diffuse contact zone lives).
    Returns (centre xyz, radius) or None when there is no fittable interface."""
    red = (rhoN > 0.0) & ~solid_np
    interior = red.copy()
    for ax in (0, 1, 2):
        interior &= np.roll(red, 1, axis=ax) & np.roll(red, -1, axis=ax)
    surf = red & ~interior
    pz, py, px = np.nonzero(surf)
    pts = np.column_stack([px, py, pz]).astype(float)
    keep = np.linalg.norm(pts - np.asarray(solid_c), axis=1) > R2 + 4.0
    pts = pts[keep]
    if len(pts) < 50:
        return None
    A = np.column_stack([2.0 * pts, np.ones(len(pts))])
    sol, *_ = np.linalg.lstsq(A, (pts**2).sum(axis=1), rcond=None)
    c = sol[:3]
    return c, float(np.sqrt(sol[3] + c @ c))


def theta_from_fit(R1f, R2, d):
    """Invert the law of cosines: the contact angle implied by the fitted droplet."""
    cos_t = (R1f**2 + R2**2 - d**2) / (2.0 * R1f * R2)
    return float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))


def slice_frame(path, rhoN, solid_np, n, cz, R2, zC, R1, title):
    """Fig.-10-style midplane slice (y = n//2): red/blue field, solid gray,
    analytical droplet circle as a white dotted overlay."""
    sl = rhoN[:, n // 2, :]
    sol = solid_np[:, n // 2, :]
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.imshow(np.where(sol, 0.0, sl), origin="lower", cmap="RdBu_r", vmin=-1.0, vmax=1.0, interpolation="nearest")
    gray = np.ma.masked_where(~sol, np.ones_like(sl, dtype=float))
    ax.imshow(gray, origin="lower", cmap=matplotlib.colors.ListedColormap(["#8a8a8a"]), vmin=0, vmax=1)
    ax.add_patch(plt.Circle(((n - 1) / 2.0, zC), R1, fill=False, ls=(0, (1.5, 2.0)), ec="white", lw=1.6))
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    args = parse_args()
    if args.plots:
        _viz.enable_plots(True)
    frames_dir, plots_dir = args.out / "frames", args.out / "plots"
    frames_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    for theta in args.thetas:  # only this run's angles: per-angle scripts must not clobber each other
        for stale in frames_dir.glob(f"t{int(theta):03d}_*.png"):
            stale.unlink()

    n, cz = args.n, args.solid_z
    c = (n - 1) / 2.0  # 50 for n = 101
    solid_c = (c, c, cz)
    zz, yy, xx = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    d2_solid = (xx - c) ** 2 + (yy - c) ** 2 + (zz - cz) ** 2
    solid_np = d2_solid <= args.R2**2
    for ax in (0, 1, 2):  # closed box: bounce-back walls mask the periodic wrap
        idx = [slice(None)] * 3
        for end in (0, -1):
            idx[ax] = end
            solid_np[tuple(idx)] = True
    solid = jnp.asarray(solid_np)
    nw = color3d.wall_normals(solid_np)  # preprocessing, shared by both angles

    rows, results = [], []
    start = time.perf_counter()
    for theta in args.thetas:
        R3 = r3_analytic(args.R1, args.R2, theta)
        zC = cz + R3
        # analytical droplet voxel volume -> matched half-shell collar (Fig. 10a)
        drop = ((xx - c) ** 2 + (yy - c) ** 2 + (zz - zC) ** 2 <= args.R1**2) & (d2_solid > args.R2**2) & ~solid_np
        target = int(drop.sum())
        lo, hi = args.R2 + 0.5, n / 2.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            count = int(((d2_solid > args.R2**2) & (d2_solid <= mid**2) & (zz >= cz) & ~solid_np).sum())
            lo, hi = (lo, mid) if count >= target else (mid, hi)
        collar = (d2_solid > args.R2**2) & (d2_solid <= hi**2) & (zz >= cz) & ~solid_np
        rhoR = jnp.asarray(np.where(collar, 1.0, 0.0))
        state = color3d.State(d3q19.W[:, None, None, None] * rhoR[None], d3q19.W[:, None, None, None] * (1.0 - rhoR)[None])

        params = color3d.Params(omega=args.omega, sigma=args.sigma, beta=args.beta, theta=theta)
        runners = {}
        tag = f"t{int(theta):03d}"
        frames3d, frames2d, step = [], [], 0

        def record(step, theta=theta, zC=zC, tag=tag, frames3d=frames3d, frames2d=frames2d, state_ref=None):
            rhoN = rho_n(state_ref)
            fit = fit_interface_sphere(rhoN, solid_np, solid_c, args.R2)
            if fit is not None:
                c_fit, R1f = fit
                d_fit = float(np.linalg.norm(c_fit - np.asarray(solid_c)))
                th_meas = theta_from_fit(R1f, args.R2, d_fit)
            else:
                R1f = d_fit = th_meas = float("nan")
            rows.append({"theta": theta, "step": step, "R1_fit": R1f, "R3_fit": d_fit, "theta_meas": th_meas})
            p3 = frames_dir / f"{tag}_3d_{step:06d}.png"
            # side-on view ("xz", z up): an iso view looks down into the box and the
            # droplet hides the solid sphere beneath it — side-on keeps both visible.
            # smooth isosurfaces: voxel faces all share one normal in an axis-aligned
            # view (flat silhouette); contoured surfaces shade like the analytical spheres
            _viz3d.phase_frame(
                p3, rhoN, solid=solid_np, text=f"theta={theta:.0f}  step {step}", camera="xz", smooth=True, rock_opacity=0.35
            )
            frames3d.append(p3)
            p2 = frames_dir / f"{tag}_slice_{step:06d}.png"
            slice_frame(p2, rhoN, solid_np, n, cz, args.R2, zC, args.R1, f"theta={theta:.0f}  step {step}")
            frames2d.append(p2)

        record(step, state_ref=state)
        while step < args.steps:
            length = min(args.block, args.steps - step)
            if length not in runners:
                runners[length] = make_runner(params, solid, length, nw)
            state = runners[length](state)
            step += length
            if step % args.frame_interval == 0 or step == args.steps:
                record(step, state_ref=state)

        _viz.write_gif(frames3d, args.out / f"{tag}_3d.gif", args.fps)
        _viz.write_gif(frames2d, args.out / f"{tag}_slice.gif", args.fps)
        last = rows[-1]
        results.append(
            {
                "theta": theta,
                "R3": R3,
                "R1_fit": last["R1_fit"],
                "R3_fit": last["R3_fit"],
                "theta_meas": last["theta_meas"],
                "final_slice": frames2d[-1],
            }
        )

    elapsed = time.perf_counter() - start
    _viz.write_csv(args.out / "metrics.csv", rows)

    # Fig.-10-style side-by-side of the final slices (opt-in via --plots)
    if _viz.plots_enabled():
        fig, axes = plt.subplots(1, len(results), figsize=(4.6 * len(results), 4.6))
        for ax, res in zip(np.atleast_1d(axes), results):
            ax.imshow(plt.imread(res["final_slice"]))
            ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(plots_dir / "comparison.png", dpi=110)
        plt.close(fig)

    ok = all(abs(r["theta_meas"] - r["theta"]) <= 12.0 and abs(r["R1_fit"] - args.R1) / args.R1 <= 0.10 for r in results)
    status = "PASS" if ok else "FAIL"
    lines = [
        "# Droplet on a Spherical Solid (Akai et al. 2018, Fig. 9/10)",
        "",
        (f"- Domain {n}^3 | solid R2={args.R2:.1f} at (50, 50, {cz:.0f}) | droplet R1={args.R1} | omega={args.omega}, "
        f"sigma={args.sigma}, beta={args.beta} | geometric wetting BC"),
        f"- {args.steps} steps per angle | runtime {elapsed:.1f}s",
        f"- Overall: {status} (theta within 12 deg, R1 within 10%)",
        "",
        "| imposed theta | R3 analytic | R1 fit | R3 fit | theta measured |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    lines += [
        f"| {r['theta']:.0f} | {r['R3']:.2f} | {r['R1_fit']:.2f} | {r['R3_fit']:.2f} | {r['theta_meas']:.1f} |" for r in results
    ]
    lines += [
        "",
        "## Artifacts",
        "",
        "- t{060,120}_3d.gif (PyVista renders), t{060,120}_slice.gif (midplane y=50, analytical circle dotted white)",
        "- metrics.csv | frames/*.png | plots/comparison.png (with --plots)",
    ]
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Droplet on sphere: {status}")
    for r in results:
        print(
            f"  theta={r['theta']:.0f}: R1 {r['R1_fit']:.2f} (target {args.R1}), "
            f"R3 {r['R3_fit']:.2f} (analytic {r['R3']:.2f}), theta_meas {r['theta_meas']:.1f} deg"
        )
    print(f"  output: {args.out}")


if __name__ == "__main__":
    main()
