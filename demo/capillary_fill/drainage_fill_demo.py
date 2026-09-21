"""Primary dynamic drainage of a single pore junction — the Geometry-2 micro-model of
Zacharoudiou, Chapman, Boek & Crawshaw, J. Fluid Mech. 824 (2017),
doi:10.1017/jfm.2017.363, figure 4 — with bob's color-gradient solver.

PHENOMENON: the non-wetting phase (NWP, air) is forced into a square pore through the
51 um throat at a constant rate (0.5 ul/min, Ca ~ 1e-4 in the feeding throat). It fills
the pore body and then leaves through the widest downstream throat T4 (107 um) first —
the Young-Laplace order of the entry pressures P = 2 gamma cos(theta) (1/d + 1/w)
(table 1 of the reference: T4 1357 Pa < T3 1620 Pa < T2 1794 Pa) — while the 33 um and
65 um side throats stay WP-filled. The reference LB is a free-energy binary fluid driven
by velocity boundaries; this demo reproduces the sequence with the color-gradient model
and a rate-controlled piston inlet (the inlet pressure floats).

GEOMETRY: geometry2.py — 238 um square body, throats 51 (inlet, x=0) / 107 (T4, x=-1) /
33 (T1, y=0) / 65 (T3, y=-1) um on the native 3 um lattice of the reference, 45 um etch
depth (15 nodes), straight 450 um arms to the domain faces.

DRIVE: ``bc.piston_inlet`` at x=1 behind the sealed x=0 plane injects pure red (NWP) at
u_in per node; the three WP ports are Zou-He pressure outlets at ambient density,
colour-anchored to the WP (``sa_red=0``: they open onto the n-decane reservoir, so any
backflow is WP). Ca = eta_w u_in / gamma follows the reference's definition (WP
viscosity); the default Ca = 1e-4 matches the experiment. ``--ca 1e-3`` shows the same
throat sequence in about a tenth of the steps and is still capillary-dominated (the viscous
drop along a filled arm is < 1% of the entry-pressure differences that select the throat).

FLUIDS: theta_wp = 26 deg (table 1 of the reference; bob's ``theta`` is the red = NWP
angle, 154 deg), sigma = 0.05, nu = 0.01 (NWP) / 0.05 (WP) -> r_eta = 5. The experimental
ratio (47) would need nu below the stability limit nu >= 0.01 at sigma = 0.05; at this Ca
the arm resistances are negligible, so r_eta does not affect the pathway. fp32, default
solver (MRT chi=0.8, akai, CSF, recolor_emag, beta=0.99).

PASS: the T4 mouth (107 um) reaches NWP majority before the T1 (33 um) and T3 (65 um)
mouths, and both side throats are still < 10% NWP when the run ends. Also reported: the
inlet over-pressure while the body fills against the Young-Laplace T4 entry pressure.

Usage:  uv run python demo/capillary_fill/drainage_fill_demo.py   (GPU if available, ~17 h)
        sbatch demo/capillary_fill/run_fig4.slurm                 (resumable SLURM job)
Resume: ... drainage_fill_demo.py --resume   (continues from <out>/checkpoint.npz)
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
import geometry2

NU_RED = 0.01            # NWP (invading, red): stability limit at sigma = 0.05
NU_BLUE = 0.05           # WP (defending, blue): r_eta = 5
SIGMA = 0.05
GAMMA_EFF = 1.13 * SIGMA  # realized surface tension per the Laplace demo calibration
THETA_WP = 26.0          # contact angle through the wetting phase (table 1 of the reference)
BETA = 0.99
CA = 1e-4                # eta_w u_in / gamma in the feeding throat (the experimental value)
PROBE = 30               # mouth probe length (nodes) for the filling-order verdict


def red_fraction(rhoR, rhoB, pore, region):
    ys, xs = region
    red = (rhoR - rhoB)[:, ys, xs] > 0
    npore = pore[:, ys, xs].sum()
    return float((red & pore[:, ys, xs]).sum() / max(npore, 1))


def entry_pressure(width, depth=geometry2.DEPTH, theta_wp=THETA_WP, gamma=GAMMA_EFF):
    """Young-Laplace piston entry pressure of a rectangular throat, lattice units."""
    return 2.0 * gamma * np.cos(np.radians(theta_wp)) * (1.0 / depth + 1.0 / width)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=HERE / "output_fig4_ca1e-4")
    ap.add_argument("--block", type=int, default=5000)
    ap.add_argument("--max-steps", type=int, default=8_000_000, help="step cap (the end state is reached at ~6.8M)")
    ap.add_argument("--ca", type=float, default=CA, help="capillary number eta_w u_in / gamma in the feeding throat")
    ap.add_argument("--nu-red", type=float, default=NU_RED, help="NWP (red, injected) kinematic viscosity")
    ap.add_argument("--nu-blue", type=float, default=NU_BLUE, help="WP (blue, defending) kinematic viscosity")
    ap.add_argument("--theta-wp", type=float, default=THETA_WP, help="contact angle through the WP, degrees")
    ap.add_argument("--arm", type=int, default=geometry2.ARM, help="straight arm length in nodes (3 um each)")
    ap.add_argument("--n-red", type=int, default=50, help="initial NWP slug length in the inlet arm (nodes)")
    ap.add_argument("--end-t4", type=float, default=0.85, help="stop once this fraction of the T4 arm is NWP")
    ap.add_argument("--mrt-chi", type=float, default=0.8,
                    help="MRT spectrum (Leclaire 2017): non-shear rates = chi*omega (<=0 selects the solver default rates)")
    ap.add_argument("--ckpt-every", type=int, default=10, help="full-state checkpoint every N blocks (0 = off)")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/checkpoint.npz if present")
    args = ap.parse_args()
    nu_red, nu_blue, theta_wp = args.nu_red, args.nu_blue, args.theta_wp
    theta_red = 180.0 - theta_wp
    u_in = args.ca * SIGMA / nu_blue                     # Ca = rho nu_w u / sigma, rho = 1
    fdir = args.out / "frames"
    fdir.mkdir(parents=True, exist_ok=True)
    ckpt = args.out / "checkpoint.npz"
    resume = args.resume and ckpt.exists()
    if not resume:
        for stale in fdir.glob("mid_*.npz"):
            stale.unlink()

    solid_np, regions, view = geometry2.build_domain(args.arm)
    nz, ny, nx = solid_np.shape
    pore_np = ~solid_np
    by, bx = view["body_origin"]
    B = geometry2.BODY
    # mouth probes: the first PROBE nodes of each downstream arm, for the filling-order verdict
    regions = dict(regions)
    regions["T1_mouth"] = (slice(by - PROBE, by), regions["T1"][1])
    regions["T3_mouth"] = (slice(by + B, by + B + PROBE), regions["T3"][1])
    regions["T4_mouth"] = (regions["T4"][0], slice(bx + B, bx + B + PROBE))
    names = list(regions)
    inlet_rows = regions["inlet"][0]
    geometry2.render(args.out / "domain.png", args.arm)
    print(f"drainage_fill on {jax.default_backend().upper()}  domain {nz}x{ny}x{nx} "
          f"({pore_np.sum() / 1e6:.2f}M pore of {solid_np.size / 1e6:.2f}M cells)  -> domain.png")
    widths = {"inlet": geometry2.W_IN, "body": geometry2.BODY, "T1": geometry2.W_T1,
              "T3": geometry2.W_T3, "T4": geometry2.W_T4}
    p_entry = {k: float(entry_pressure(w, theta_wp=theta_wp)) for k, w in widths.items()}
    (args.out / "domain.json").write_text(json.dumps(
        {"shape": [nz, ny, nx], **view,
         "regions": {k: [[s.start, s.stop] for s in v] for k, v in regions.items()},
         "params": {"nu_red": nu_red, "nu_blue": nu_blue, "sigma": SIGMA, "gamma_eff": GAMMA_EFF,
                    "theta_wp": theta_wp, "theta_red": theta_red, "beta": BETA, "ca": args.ca, "u_in": u_in,
                    "mrt_chi": args.mrt_chi, "block": args.block, "n_red": args.n_red,
                    "inlet": "piston (x=0 sealed, kick at x=1, sa_red=1)", "vents": "zou-he rho=1, sa_red=0 (WP)"},
         "entry_pressure_lu": p_entry, "widths_nodes": widths}, indent=1))

    chi = args.mrt_chi if args.mrt_chi and args.mrt_chi > 0 else None
    p = color3d.Params(omega=1.0 / (3 * nu_red + 0.5), sigma=SIGMA, beta=BETA,
                       omega2=1.0 / (3 * nu_blue + 0.5), theta=theta_red, mrt_chi=chi)
    print(f"  nu_red(NWP)={nu_red} nu_blue(WP)={nu_blue} r_eta={nu_blue / nu_red:.0f}  theta_wp={theta_wp} "
          f"(red {theta_red})  Ca_w={args.ca:.1e} -> u_in={u_in:.2e}  MRT chi={chi}  x64={jax.config.jax_enable_x64}")
    print("  Young-Laplace entry pressures (lu): " + "  ".join(f"{k}={v:.4f}" for k, v in p_entry.items()), flush=True)
    solid = jnp.asarray(solid_np)
    step_solid = solid.at[:, :, 0].set(True)              # sealed piston plane (halfway bounce = reflection half)
    nw = color3d.wall_normals(solid_np)                   # from the unsealed mask: piston face wetting-neutral

    def body(s, _):
        s = color3d.step(s, p, solid=step_solid, nw=nw)
        s = bc.piston_inlet(s, u_in, solid=step_solid, sa_red=1.0, plane=1)                 # NWP syringe
        s = bc.zou_he_outlet(s, solid=solid, rho_out=1.0, sa_red=0.0)                       # T4 port (x=-1)
        s = bc_faces.zou_he_pressure_outlet(s, "y", 0, solid=solid, rho_out=1.0, sa_red=0.0)   # T1 port
        s = bc_faces.zou_he_pressure_outlet(s, "y", -1, solid=solid, rho_out=1.0, sa_red=0.0)  # T3 port
        return s, None

    runner = jax.jit(lambda s: jax.lax.scan(body, s, None, length=args.block)[0])
    step, rows = 0, []
    cols = names + ["dp_in", "front_x"]
    if resume:
        ck = np.load(ckpt)
        step = int(ck["step"])
        assert tuple(ck["fR"].shape) == (19, nz, ny, nx), f"checkpoint shape {ck['fR'].shape} != domain"
        state = color3d.State(jnp.asarray(ck["fR"]), jnp.asarray(ck["fB"]))
        for line in (args.out / "metrics.csv").read_text().splitlines()[1:]:
            vals = line.split(",")
            if int(vals[0]) <= step:
                rows.append((int(vals[0]), dict(zip(cols, map(float, vals[1:])))))
        print(f"RESUME from {ckpt} at step {step} ({len(rows)} metric rows kept)")
    else:
        state = porous3d.init_drainage(nz, ny, nx, n_red=args.n_red)

    def save_checkpoint(state, step):
        tmp = ckpt.with_suffix(".tmp.npz")
        np.savez(tmp, step=step, fR=np.asarray(state.fR), fB=np.asarray(state.fB))
        tmp.replace(ckpt)

    def measure(state):
        rhoR, rhoB = (np.asarray(a) for a in color3d.densities(state))
        d = {n: red_fraction(rhoR, rhoB, pore_np, regions[n]) for n in names}
        rho_in = (rhoR + rhoB)[1:-1, inlet_rows, 2]                       # inlet plane just past the piston kick
        d["dp_in"] = float((rho_in.mean() - 1.0) / 3.0)                   # over-pressure vs the ambient ports
        mid = (rhoR - rhoB)[nz // 2]
        redx = np.where((mid > 0) & pore_np[nz // 2])[1]
        d["front_x"] = float(redx.max()) if redx.size else 0.0
        return d, mid.astype(np.float16)

    t0, step0 = time.time(), step
    with open(args.out / "metrics.csv", "w") as csv:
        csv.write("step," + ",".join(cols) + "\n")
        csv.writelines(f"{st}," + ",".join(f"{d[c]:.5g}" for c in cols) + "\n" for st, d in rows)
    with open(args.out / "metrics.csv", "a") as csv:
        while step < args.max_steps:
            state = runner(state)
            step += args.block
            d, mid = measure(state)
            rows.append((step, d))
            csv.write(f"{step}," + ",".join(f"{d[c]:.5g}" for c in cols) + "\n")
            csv.flush()
            np.savez_compressed(fdir / f"mid_{step:08d}.npz", phi=mid)
            if not np.isfinite(list(d.values())).all() or not np.isfinite(mid).all():
                print(f"DIVERGED at step {step}")
                break
            if args.ckpt_every and step % (args.block * args.ckpt_every) == 0:
                save_checkpoint(state, step)
            if step % (args.block * 20) == 0:
                print(f"  step {step:>8d}  " + "  ".join(f"{n}={d[n]:.2f}" for n in ("inlet", "body", "T1", "T3", "T4"))
                      + f"  dp_in={d['dp_in']:.4f}  front_x={d['front_x']:.0f}"
                      + f"  ({(step - step0) / (time.time() - t0):.0f} steps/s)", flush=True)
            if d["T4"] >= args.end_t4:
                print(f"  T4 arm {d['T4']:.2f} NWP at step {step}: end state reached")
                break
    if args.ckpt_every:
        save_checkpoint(state, step)

    # ---- verdict + report ----
    def t50(name):
        for s, d in rows:
            if d[name] > 0.5:
                return s
        return None

    tt = {n: t50(n) for n in ("T4_mouth", "T3_mouth", "T1_mouth", "body")}
    finite = np.isfinite([v for _, d in rows for v in d.values()]).all()
    last = rows[-1][1]
    order_ok = tt["T4_mouth"] is not None \
        and all(tt[n] is None or tt["T4_mouth"] < tt[n] for n in ("T3_mouth", "T1_mouth")) \
        and last["T1"] < 0.10 and last["T3"] < 0.10
    status = "PASS" if (order_ok and finite) else "FAIL"
    # inlet over-pressure while the body fills (body > 5% NWP, T4 mouth still < 5%): the
    # maximum is the pressure that opened T4 -> compare with the Young-Laplace T4 entry pressure
    fill = [d["dp_in"] for _, d in rows if d["body"] > 0.05 and d["T4_mouth"] < 0.05]
    dp_peak = max(fill) if fill else float("nan")
    lines = [
        "# drainage_fill — primary dynamic drainage, Geometry 2 (Zacharoudiou et al., JFM 2017, fig 4)",
        "",
        f"**{status}** — the NWP injected through the 51 um throat must leave the pore body through the",
        "widest downstream throat T4 (107 um) first (lowest Young-Laplace entry pressure), with the 33 um",
        "and 65 um side throats staying WP-filled.",
        "",
        (f"domain {nz}x{ny}x{nx} @ {geometry2.DX_UM:.0f} um/vox (native pitch of the reference), "
         f"arms {args.arm * geometry2.DX_UM:.0f} um, theta_wp={theta_wp} deg (red {theta_red}), "
         f"nu={nu_red}/{nu_blue} NWP/WP (r_eta={nu_blue / nu_red:.0f}), Ca_w={args.ca:.1e} (u_in={u_in:.2e}), "
         f"sigma={SIGMA}, beta={BETA}, piston inlet + WP-anchored Zou-He ports (y faces via bc_faces), fp32"),
        "",
        "| throat | width | YL entry pressure (lu) | t(mouth red majority) | NWP fraction of the arm at end |",
        "|---|---:|---:|---:|---:|",
    ] + [f"| {n} | {geometry2.WIDTHS_UM[n]} um ({widths[n]} nodes) | {p_entry[n]:.4f} | "
         f"{tt[n + '_mouth'] or '-'} | {last[n]:.2f} |" for n in ("T4", "T3", "T1")] + [
        "",
        (f"Inlet over-pressure peak while the body filled: {dp_peak:.4f} lu vs Young-Laplace T4 entry "
         f"{p_entry['T4']:.4f} (T3 {p_entry['T3']:.4f}, T1 {p_entry['T1']:.4f}; body entry {p_entry['body']:.4f})."),
        "",
        (f"Body 50% NWP at step {tt['body']}; total steps {step}; this segment {(time.time() - t0) / 60:.0f} min "
         f"at {(step - step0) / (time.time() - t0):.0f} steps/s" + (f" (resumed at step {step0})." if step0 else ".")),
        "",
        "Artifacts: domain.png, metrics.csv, frames/mid_*.npz (z-midplane phi), live/ (renders + GIF",
        "from live_render.py).",
    ]
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"{status}  t50: " + "  ".join(f"{k}={v}" for k, v in tt.items()) + f"  dp_peak={dp_peak:.4f} YL_T4={p_entry['T4']:.4f}")
    print(f"  output: {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
