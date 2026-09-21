"""Washburn capillary imbibition (3D cylindrical tube) — reproduction of
Sedahmed & Coelho, Phys. Fluids 36, 092117 (2024), DOI 10.1063/5.0228835, §IV.A: a wetting fluid spontaneously
imbibes a capillary tube (radius r=5), displacing the non-wetting fluid under
capillary suction (dP=0). L(t) is compared to the two-fluid Washburn law (Eq. 47).

Setup (following the paper):
  * wetting  : akai reorientation
  * surface tension : CSF body force F=½σκ∇φ + Eq.34/35 extrapolation (Params.csf=True)
  * recoloring : paper Eq.17 |e_i| factor (Params.recolor_emag=True), beta=0.95
  * BC       : per-color Zou-He pressure inlet and outlet (dP=0), as in the paper's
               App. B (bc.zou_he_inlet / zou_he_outlet)
  * reservoir: frictionless (periodic, no-wall) open chambers feeding the tube

Fluid settings of the paper: gamma=1/45, M=nu_nw/nu_w=1/5 (tau_w=1.0, tau_nw=0.6), beta=0.95.

Artifacts:
  * washburn_Lt.png      — L(t): LBM dots vs two-fluid analytic, one colour per angle
  * imbibition_grid.png  — final 3D state of all 6 tubes (2 rows x 3), wall transparent,
                           tube vertical, wetting rising bottom -> top
  * imbibition.gif       — the same 2x3 grid animated as the six tubes imbibe

Usage:  uv run python demo/washburn/washburn_demo.py     (uses GPU if available)
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

from bob import bc, color3d, d3q19, porous3d
from bob.utils import viz, viz3d, washburn

plt.switch_backend("Agg")

GAMMA = 1.0 / 45
OM2 = 1.0 / 0.6
MU_W = float(d3q19.CS2) * 0.5
MU_NW = float(d3q19.CS2) * (1.0 / OM2 - 0.5)


def cylinder_with_reservoir(nz, ny, nx, R, res):
    """R-radius cylinder along x for res<=x<nx-res; frictionless (no-solid -> periodic
    y,z) open reservoir chambers for x<res and x>=nx-res."""
    yc, zc = (ny - 1) / 2.0, (nz - 1) / 2.0
    zz, yy = np.indices((nz, ny))
    tube = ((yy - yc) ** 2 + (zz - zc) ** 2) >= R ** 2          # (nz,ny): solid outside cylinder
    mask = np.zeros((nz, ny, nx), bool)
    for x in range(nx):
        if res <= x < nx - res:
            mask[:, :, x] = tube
    return mask


def vertical_camera(shape):
    """Camera that stands the x-flow tube upright (flow axis -> screen up), viewed from
    -y. ``_render`` reset_camera reframes the whole domain along this direction, so the
    long tube fills the frame vertically and the inlet (x=0) is at the bottom."""
    nz, ny, nx = shape
    c = (nx / 2.0, ny / 2.0, nz / 2.0)          # PyVista coords: x<-nx (flow), y<-ny, z<-nz
    return [(c[0], c[1] - 2.0 * nx, c[2]), c, (1.0, 0.0, 0.0)]


def run_angle(theta, solid, nw, nz, ny, nx, R, res, block, nblocks):
    p = color3d.Params(omega=1.0, sigma=GAMMA, beta=0.95, theta=theta, omega2=OM2)  # default solver
    solid_j = jnp.asarray(solid)

    def body(s, _):
        s = color3d.step(s, p, solid=solid_j, nw=nw)
        s = bc.zou_he_inlet(s, solid=solid_j, rho_in=1.0, sa_red=1.0)   # paper inlet
        s = bc.zou_he_outlet(s, solid=solid_j, rho_out=1.0)             # paper outlet
        return s, None

    runner = jax.jit(lambda s: jax.lax.scan(body, s, None, length=block)[0])
    state = porous3d.init_drainage(nz, ny, nx, n_red=res)
    ts, zs, snaps, step = [], [], [], 0
    for _ in range(nblocks):
        state = runner(state)
        step += block
        ts.append(step)
        zs.append(porous3d.filled_length(state, solid, x0=res))
        snaps.append(np.asarray(color3d.color_field(*color3d.densities(state))))   # (nz,ny,nx) rhoN
    return np.array(ts, float), np.array(zs, float), snaps


def render_tube(path, rhoN, solid, cam, window=(240, 680)):
    """Vertical tube, transparent wall, wetting (red) opaque. Voxel threshold (no
    smoothing): shows the full red volume (rhoN>0), so the red reservoir feed and the
    risen column are both visible — not just the meniscus isosurface."""
    viz3d.phase_frame(path, rhoN, solid, camera=cam, smooth=False, shade=True, zoom=1.5, rock_opacity=0.10, window=window)


def grid23(path, render_paths, titles, suptitle, dpi):
    fig, axes = plt.subplots(2, 3, figsize=(12, 9))
    for ax, rp, title in zip(axes.ravel(), render_paths, titles):
        ax.imshow(plt.imread(rp))
        ax.set_axis_off()
        ax.set_title(title, fontsize=11)
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("demo/washburn/output"))
    ap.add_argument("--nz", type=int, default=12)
    ap.add_argument("--ny", type=int, default=12)
    ap.add_argument("--nx", type=int, default=101)
    ap.add_argument("--R", type=int, default=5)
    ap.add_argument("--res", type=int, default=6, help="frictionless reservoir length each end")
    ap.add_argument("--block", type=int, default=1500)
    ap.add_argument("--nblocks", type=int, default=20, help="20*1500 = 30000 steps (plot ends at 30000)")
    ap.add_argument("--thetas", type=float, nargs="+", default=[20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--curve-only", action="store_true", help="rerun the physics + L(t) plot/report, skip 3D renders/GIF")
    ap.add_argument("--plots", action="store_true",
                    help="also write the analysis curves (washburn_Lt.png/.svg); off by default")
    args = ap.parse_args()
    if args.plots:
        viz.enable_plots(True)
    args.out.mkdir(parents=True, exist_ok=True)
    rdir = args.out / "_render"
    rdir.mkdir(exist_ok=True)
    if not args.curve_only:
        for stale in rdir.glob("*.png"):
            stale.unlink()
    L_tube = args.nx - 2 * args.res

    solid = cylinder_with_reservoir(args.nz, args.ny, args.nx, args.R, args.res)
    nw = color3d.wall_normals(solid)
    cam = vertical_camera(solid.shape)

    print(f"Washburn 3D on {jax.default_backend().upper()}  (r={args.R}, {args.nz}x{args.ny}x{args.nx}, paper Zou-He inlet+outlet)")
    t0 = time.time()
    per_theta, analytic, rows, all_snaps = [], [], [], []
    for theta in args.thetas:
        ts, zs, snaps = run_angle(theta, solid, nw, args.nz, args.ny, args.nx, args.R, args.res, args.block, args.nblocks)
        per_theta.append((theta, ts, zs))
        all_snaps.append(snaps)
        A_an = washburn.capillary_A(3, args.R, GAMMA, theta)
        td = np.linspace(ts[0], ts[-1], 200)
        analytic.append((td, washburn.two_fluid_Lt(td - ts[0], A_an, MU_W, MU_NW, L_tube, L0=zs[0])))
        A_meas, _, r2 = washburn.fit_slope(ts, washburn.rectify(zs, MU_W, MU_NW, L_tube), 0.1, 0.95)
        rows.append((theta, A_meas / A_an, r2, zs[-1] / args.R, bool(np.isfinite(zs).all())))
        print(f"  theta={theta:.0f}: A_meas/A_an={A_meas / A_an:.2f}  R^2={r2:.3f}  L_final/r={zs[-1] / args.R:.1f}")

    # L(t) curve
    washburn.curve_Lt(args.out / "washburn_Lt.png", per_theta, analytic)

    if not args.curve_only:
        # 3D renders: vertical tube, transparent wall, wetting rising bottom -> top.
        # final still (2x3 grid)
        final_paths, final_titles = [], []
        for theta, snaps, (_, ar, _, _, _) in zip(args.thetas, all_snaps, rows):
            rp = rdir / f"final_t{int(theta):03d}.png"
            render_tube(rp, snaps[-1], solid, cam, window=(360, 1000))
            final_paths.append(rp)
            final_titles.append(f"theta = {theta:.0f} deg   (A/A_an = {ar:.2f})")
        grid23(args.out / "imbibition_grid.png", final_paths, final_titles,
               "3D Washburn imbibition — final state (tube vertical, wall transparent, wetting rises bottom->top)", 400)

        # animated 2x3 GIF over the imbibition
        frame_paths = []
        fdir = args.out / "frames"
        fdir.mkdir(exist_ok=True)
        for stale in fdir.glob("frame_*.png"):
            stale.unlink()
        for k in range(args.nblocks):
            rps = []
            for theta, snaps in zip(args.thetas, all_snaps):
                rp = rdir / f"t{int(theta):03d}_k{k:03d}.png"
                render_tube(rp, snaps[k], solid, cam)
                rps.append(rp)
            fp = fdir / f"frame_{k:03d}.png"
            grid23(fp, rps, [f"theta = {t:.0f} deg" for t in args.thetas],
                   f"3D Washburn imbibition  (step {per_theta[0][1][k]:.0f})", 400)
            frame_paths.append(fp)
        viz.write_gif(frame_paths, args.out / "imbibition.gif", args.fps)

    ok = all(0.6 < r[1] < 1.3 and r[2] > 0.97 and r[4] for r in rows)
    status = "PASS" if ok else "FAIL"
    lines = [
        "# Washburn capillary imbibition (3D cylindrical tube) — Sedahmed & Coelho, Phys. Fluids 36, 092117 (2024), Sec. IV.A",
        "",
        (f"**{status}**  (r={args.R}, domain {args.nz}x{args.ny}x{args.nx}, frictionless reservoir={args.res}; "
        "gamma=1/45, M=1/5, beta=0.95; default solver + per-color Zou-He pressure inlet and outlet, paper App. B)"),
        "",
        "Two-fluid Washburn (Eq. 47, 3D tube): A = r*gamma*cos(theta)/4. 1.0 = on the analytic law.",
        "",
        "| imposed theta | A_meas/A_analytic | R^2 | L_final/r | finite |",
        "|---:|---:|---:|---:|:---:|",
    ]
    for theta, ar, r2, zr, fin in rows:
        lines.append(f"| {theta:.0f} | {ar:.2f} | {r2:.3f} | {zr:.1f} | {'yes' if fin else 'NO'} |")
    lines += ["", "Artifacts: washburn_Lt.csv (washburn_Lt.png|svg with --plots), imbibition_grid.png (final 3D, 2x3, wall transparent, "
              "tube vertical), imbibition.gif (animated). Runtime %.0fs." % (time.time() - t0)]
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Washburn imbibition (3D tube, paper Zou-He inlet+outlet): {status}")
    print(f"  output: {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
