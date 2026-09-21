"""Spontaneous imbibition into a single pore junction with four unequal throats —
the Geometry-3 micro-model of Zacharoudiou, Chapman, Boek & Crawshaw, J. Fluid Mech. 824
(2017), doi:10.1017/jfm.2017.363, figure 8 — with bob's color-gradient solver.

PHENOMENON: the wetting phase (WP) entering the junction through throat T3 (60 um) fills
the adjacent throat T2 (47 um) first — not the narrowest throat T1 (27 um) that
quasi-static Young-Laplace filling rules predict — and the non-wetting phase (NWP)
retreats through the widest throat T4 (100 um). The reference simulations use a
free-energy binary-fluid LB model; this demo reproduces the same displacement sequence
with the color-gradient model.

GEOMETRY: geometry.py — the junction as a mathematical description (rectangles on the
native 3 um lattice of the reference). T3 inlet left, T1 right, T2 bottom, T4 top; 600 um
straight arms; 55 um etch depth (18 nodes) in z.

DRIVE (spontaneous, dP = 0): every port sits at the same ambient density — a per-color
Zou-He pressure inlet on x=0 (pure WP reservoir) and pressure outlets on x=-1 (T1) and on
the y faces (T4 at y=0, T2 at y=-1) through the demo-local ``bc_faces`` module. The three
vents open onto air, so their colour is anchored to the NWP (``sa_red=0``): backflow
through a vent is always NWP and a WP-filled arm cannot turn its port into a WP reservoir.
Capillary suction alone moves the interface (the x-face driver is validated against the
two-fluid Washburn law in demo/washburn).

FLUIDS: theta = 16 deg (the experimental contact angle), sigma = 0.05, nu = 0.06 (WP) /
0.01 (NWP): Ohnesorge number Oh = 0.059, matching the reference (Oh = 6e-2), at viscosity
ratio r_eta = 6. The NWP viscosity sits at the stability limit nu >= 0.01 for sigma = 0.05,
so Oh and r_eta (reference: 50) cannot both be matched; Oh is matched because it controls
the interface morphology, while the filling sequence is independent of r_eta over the
reference's 5..500 sweep. fp32, default solver (MRT chi=0.8, akai, CSF, recolor_emag,
beta=0.99).

PASS: T2 reaches WP majority before T1 and before T4.

Usage:  uv run python demo/capillary_fill/capillary_fill_demo.py   (GPU if available, ~1.2 h)
Resume: ... capillary_fill_demo.py --resume   (continues from <out>/checkpoint.npz, written
        every --ckpt-every blocks; metrics.csv and frames/ are kept and appended)
Live:   uv run python demo/capillary_fill/live_render.py           (frames + GIF on the fly)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from bob import bc, color3d, porous3d

HERE = Path("demo/capillary_fill")
sys.path.insert(0, str(HERE))
import bc_faces  # demo-local, after the path insert
import geometry

NU_RED = 0.06            # WP: Oh = 0.059 with NU_BLUE at its stability limit (reference: Oh = 6e-2)
NU_BLUE = 0.01           # NWP: nu < 0.01 is unstable at sigma = 0.05
SIGMA = 0.05
THETA = 16.0
BETA = 0.99


def red_fraction(rhoR, rhoB, pore, region):
    ys, xs = region
    red = (rhoR - rhoB)[:, ys, xs] > 0
    npore = pore[:, ys, xs].sum()
    return float((red & pore[:, ys, xs]).sum() / max(npore, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=HERE / "output_oh006")
    ap.add_argument("--block", type=int, default=2500)
    ap.add_argument("--max-steps", type=int, default=250_000, help="run length (the junction is fully WP-filled by ~200k)")
    ap.add_argument("--nu-red", type=float, default=NU_RED, help="WP kinematic viscosity (sets Oh)")
    ap.add_argument("--nu-blue", type=float, default=NU_BLUE, help="NWP kinematic viscosity (>= 0.01 at sigma 0.05)")
    ap.add_argument("--theta", type=float, default=THETA, help="equilibrium contact angle of the WP, degrees")
    ap.add_argument("--mrt-chi", type=float, default=0.8,
                    help="MRT spectrum (Leclaire 2017): non-shear rates = chi*omega (<=0 selects the solver default rates)")
    ap.add_argument("--ckpt-every", type=int, default=10, help="full-state checkpoint every N blocks (0 = off)")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/checkpoint.npz if present")
    args = ap.parse_args()
    nu_red, nu_blue, theta = args.nu_red, args.nu_blue, args.theta
    fdir = args.out / "frames"
    fdir.mkdir(parents=True, exist_ok=True)
    ckpt = args.out / "checkpoint.npz"
    resume = args.resume and ckpt.exists()
    if not resume:
        for stale in fdir.glob("mid_*.npz"):
            stale.unlink()

    solid_np, regions, view = geometry.build_domain()
    nz, ny, nx = solid_np.shape
    pore_np = ~solid_np
    geometry.render(args.out / "domain.png")
    print(f"capillary_fill on {jax.default_backend().upper()}  domain {nz}x{ny}x{nx} "
          f"({pore_np.sum() / 1e6:.2f}M pore of {solid_np.size / 1e6:.2f}M cells)  -> domain.png")
    (args.out / "domain.json").write_text(json.dumps(
        {"shape": [nz, ny, nx], **view,
         "regions": {k: [[s.start, s.stop] for s in v] for k, v in regions.items()},
         "params": {"nu_red": nu_red, "nu_blue": nu_blue, "sigma": SIGMA, "theta": theta,
                    "beta": BETA, "mrt_chi": args.mrt_chi, "block": args.block,
                    "vent_color": "nwp (sa_red=0)"}},
        indent=1))

    chi = args.mrt_chi if args.mrt_chi and args.mrt_chi > 0 else None
    p = color3d.Params(omega=1.0 / (3 * nu_red + 0.5), sigma=SIGMA, beta=BETA,
                       omega2=1.0 / (3 * nu_blue + 0.5), theta=theta, mrt_chi=chi)
    oh = nu_red / np.sqrt(1.13 * SIGMA * 2 * geometry.DEPTH * geometry.T3_W / (geometry.DEPTH + geometry.T3_W))
    print(f"  nu_red={nu_red} nu_blue={nu_blue} theta={theta}  r_eta={nu_red / nu_blue:.0f}  "
          f"Oh={oh:.3f} (gamma=1.13*sigma from the Laplace demo, L=D_h(T3))  MRT chi={chi}", flush=True)
    solid = jnp.asarray(solid_np)
    nw = color3d.wall_normals(solid_np)

    def body(s, _):
        s = color3d.step(s, p, solid=solid, nw=nw)
        s = bc.zou_he_inlet(s, solid=solid, rho_in=1.0, sa_red=1.0)                            # WP reservoir (x=0)
        # the three vents open onto air (NWP) at ambient pressure; sa_red=0 anchors their colour
        s = bc.zou_he_outlet(s, solid=solid, rho_out=1.0, sa_red=0.0)                          # T1 port (x=-1)
        s = bc_faces.zou_he_pressure_outlet(s, "y", 0, solid=solid, rho_out=1.0, sa_red=0.0)   # T4 port
        s = bc_faces.zou_he_pressure_outlet(s, "y", -1, solid=solid, rho_out=1.0, sa_red=0.0)  # T2 port
        return s, None

    runner = jax.jit(lambda s: jax.lax.scan(body, s, None, length=args.block)[0])
    names = list(regions)
    step, t2_done_step, rows = 0, None, []
    if resume:
        ck = np.load(ckpt)
        step = int(ck["step"])
        assert tuple(ck["fR"].shape) == (19, nz, ny, nx), f"checkpoint shape {ck['fR'].shape} != domain"
        state = color3d.State(jnp.asarray(ck["fR"]), jnp.asarray(ck["fB"]))
        # replay the metrics rows up to the checkpoint step (later rows are re-simulated)
        for line in (args.out / "metrics.csv").read_text().splitlines()[1:]:
            vals = line.split(",")
            if int(vals[0]) <= step:
                rows.append((int(vals[0]), dict(zip(names, map(float, vals[1:])))))
        for st, d in rows:
            if t2_done_step is None and d["T2"] > 0.90:
                t2_done_step = st
        print(f"RESUME from {ckpt} at step {step} ({len(rows)} metric rows kept)")
    else:
        state = porous3d.init_drainage(nz, ny, nx, n_red=geometry.RES_X)

    def save_checkpoint(state, step):
        tmp = ckpt.with_suffix(".tmp.npz")
        np.savez(tmp, step=step, fR=np.asarray(state.fR), fB=np.asarray(state.fB))
        tmp.replace(ckpt)  # atomic: a kill mid-write never corrupts the live checkpoint

    t0, step0 = time.time(), step
    # (re)write header + kept rows so metrics.csv ends exactly at the start step, then append
    with open(args.out / "metrics.csv", "w") as csv:
        csv.write("step," + ",".join(f"red_{n}" for n in names) + "\n")
        for st, d in rows:
            csv.write(f"{st}," + ",".join(f"{d[n]:.4f}" for n in names) + "\n")
    with open(args.out / "metrics.csv", "a") as csv:
        while step < args.max_steps:
            state = runner(state)
            step += args.block
            rhoR, rhoB = (np.asarray(a) for a in color3d.densities(state))
            fr = [red_fraction(rhoR, rhoB, pore_np, regions[n]) for n in names]
            rows.append((step, dict(zip(names, fr))))
            csv.write(f"{step}," + ",".join(f"{v:.4f}" for v in fr) + "\n")
            csv.flush()
            mid = (rhoR - rhoB)[nz // 2].astype(np.float16)
            np.savez_compressed(fdir / f"mid_{step:08d}.npz", phi=mid)
            if not np.isfinite(fr).all() or not np.isfinite(mid).all():
                print(f"DIVERGED at step {step}")
                break
            if args.ckpt_every and step % (args.block * args.ckpt_every) == 0:
                save_checkpoint(state, step)
            if step % (args.block * 20) == 0:
                print(f"  step {step:>8d}  " + "  ".join(f"{n}={v:.2f}" for n, v in zip(names, fr))
                      + f"  ({(step - step0) / (time.time() - t0):.0f} steps/s)", flush=True)
            if t2_done_step is None and fr[names.index("T2")] > 0.90:
                t2_done_step = step
                print(f"  T2 filled at step {step}")
    if args.ckpt_every:
        save_checkpoint(state, step)

    # ---- verdict + report ----
    def t50(name):
        for s, d in rows:
            if d[name] > 0.5:
                return s
        return None

    tt = {n: t50(n) for n in ("T1", "T2", "T4")}
    finite = np.isfinite([v for _, d in rows for v in d.values()]).all()
    order_ok = tt["T2"] is not None and (tt["T1"] is None or tt["T2"] < tt["T1"]) \
        and (tt["T4"] is None or tt["T2"] < tt["T4"])
    status = "PASS" if (order_ok and finite) else "FAIL"
    last = rows[-1][1]
    lines = [
        "# capillary_fill — spontaneous imbibition, Geometry 3 (Zacharoudiou et al., JFM 2017, fig 8)",
        "",
        f"**{status}** — the wetting phase must imbibe the adjacent throat T2 (47 um) before the",
        "narrowest throat T1 (27 um); the Young-Laplace quasi-static rules predict T1 first.",
        "",
        (f"domain {nz}x{ny}x{nx} @ {geometry.DX_UM:.0f} um/vox (native pitch of the reference), "
         f"arms {geometry.ARM * geometry.DX_UM:.0f} um, theta={theta} deg, nu={nu_red}/{nu_blue} "
         f"(r_eta={nu_red / nu_blue:.0f}, Oh={oh:.3f}), "
         f"sigma={SIGMA}, beta={BETA}, dP=0 Zou-He on all four ports (y faces via bc_faces), fp32"),
        "",
        "| throat | t(red majority) | red fraction at end |",
        "|---|---:|---:|",
    ] + [f"| {n} ({int(w)} um) | {tt.get(n) or '-'} | {last[n]:.2f} |"
         for n, w in (("T2", 47), ("T1", 27), ("T4", 100))] + [
        "",
        (f"T2 filled at step {t2_done_step}; total steps {step}; "
         f"this segment {(time.time() - t0) / 60:.0f} min at {(step - step0) / (time.time() - t0):.0f} steps/s"
         + (f" (resumed at step {step0})." if step0 else ".")),
        "",
        "Artifacts: domain.png, metrics.csv, frames/mid_*.npz (z-midplane phi), live/ (renders + GIF",
        "from live_render.py).",
    ]
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"{status}  t50: " + "  ".join(f"{k}={v}" for k, v in tt.items()))
    print(f"  output: {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
