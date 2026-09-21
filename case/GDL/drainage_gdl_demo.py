"""Constant-rate capillary drainage on GDL/fiber slabs — no porous plate.

The case/drainage/drainage_demo.py standardized driver with one change: the
porous-plate outlet (and its dp cap, which is defined by the plate entry
pressure) is removed, and termination is anchored on breakthrough. Physics,
geometry assembly, logging and metadata are otherwise identical, so the two
campaigns are directly comparable.

Physics:
  piston inlet (u_in*A red/step exact, inlet pressure floats), sigma 0.05,
  baseline nu0 0.04 split geometrically for the viscosity ratio
  M = nu_red/nu_blue (--m: nu_r = nu0*sqrt(M), nu_b = nu0/sqrt(M)),
  Ca = 1e-5 nu0-referenced (u_in = Ca*sigma/nu0, independent of M),
  theta_rock via --theta (red = invading non-wetting), MRT chi 0.8, fp32,
  solver defaults MRT + akai + CSF + recolor_emag + wall_grad="fluid". The
  recoloring sharpness --beta defaults to 0.7; the dataset campaign passes
  --beta 0.99 (case/run_array.slurm).

Geometry (core-holder layout, plate-free):
  [piston face x=0 | inlet reservoir | rock | open outlet buffer]
  along x, solid no-slip z/y walls over the whole domain. The rock is a
  gdl_ct or fiber slab of the geometry dataset (128 x 128 x 64 or 256 x 256 x 128, flow
  axis x = through-plane). With no plate the outlet buffer is an exit gallery
  in front of the Zou-He blue-pinned outlet; the per-block scrubber converts
  red that leaves the rock into blue (mass- and momentum-conserving, logged
  cumulatively as "scrubbed"), so broken-through red is produced, never
  returned or piled against the outlet BC. Walls keep the rock contact angle
  everywhere except the purely red-wetting piston face slab x=1.

Run protocol (termination — breakthrough-anchored, volume-capped):
  Breakthrough (red first reaching the last rock slab) is the anchor: the step
  it happens at, and the saturation there, are recorded in run_meta.json
  ("breakthrough_step" / "breakthrough_sat", also in report.md).
  The run then stops at whichever comes first:
  1. BT cap (finish_type "bt_cap"): step reaches --bt-x (default 1.2) times the
     breakthrough step, i.e. a fifth of the pre-breakthrough time again, over
     which the post-breakthrough saturation settles.
  2. PV cap (finish_type "pv_cap"): the injected red volume u_in*A*step reaches
     --pv-cap (default 0.5) of the rock pore volume. This also bounds runs whose
     breakthrough never fires.
  Until breakthrough the ETA in run_meta.json targets the PV cap; once it fires
  the target is retimed to min(PV cap, bt_x * breakthrough).
  There is no dp cap (that limit protects a porous plate; there is none).
  Abnormal exits record "diverged" or "step_cap".

Logging (every --block = 5000 steps): metrics.csv row + saturation curve +
3D phase-field frame (GIF) + fields .h5/.xdmf append (phi, rho, p, umag, u)
+ run_meta.json heartbeat. Status-named checkpoints every 10 blocks.

Usage (from the repository root):
  .venv/bin/python case/GDL/drainage_gdl_demo.py \
      --structure <slab.npy> --out case/GDL/runs/<name> --name <name> \
      --m 10 --theta 150 --beta 0.99
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import jax

# fp32 is the solver default (bandwidth-bound, ~2x steps/s, validated at parity).
# Precision is decided before importing bob.
FP32 = os.environ.get("BOB_FP64") != "1"
jax.config.update("jax_enable_x64", not FP32)

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bob import color3d, d3q19, multigpu, porous3d
from bob.utils import stats, viz, viz3d
from bob.utils.file import FieldWriter, RunMeta

CS2 = float(d3q19.CS2)

# ---- the standardized setting (module constants = the contract) -----------------
SIGMA = 0.05  # CSF surface tension
NU0 = 0.04  # baseline kinematic viscosity (geometric mean of the two phases)
CA = 1e-5  # capillary regime -> u_in = CA * SIGMA / NU0 = 1.25e-5
THETA_ROCK = 135.0  # red contact angle on the rock (drainage entry pressure)
THETA_FACE = 0.0  # piston-face slab x=1: purely red-wetting
CHI = 0.8  # MRT chi spectrum (Leclaire), the dual-viscosity fix
BETA = 0.7
BLOCK = 5000  # the log interval: metrics + GIF frame + fields.h5 cadence
PV_CAP = 0.5  # hard cap: stop after injecting this multiple of the rock pore volume
BT_X = 1.2  # stop at this multiple of the breakthrough step (whichever cap comes first)
FIELD_VARS = "phi,rho,p,umag,u"  # color field, total density, pressure, |u|, velocity


def omega_pair(M, nu0):
    """Split the baseline viscosity nu0 geometrically into (red, blue) for ratio
    M = nu_red/nu_blue (the convention of demo/lenormand). Returns
    (omega_red, omega_blue_or_None, nu_red, nu_blue); omega_blue is None when
    M == 1 (equal viscosity -> the solver's omega2=None fast path)."""
    nu_r = nu0 * M**0.5
    nu_b = nu0 / M**0.5
    omega_r = 1.0 / (nu_r / CS2 + 0.5)
    omega_b = None if abs(M - 1.0) < 1e-9 else 1.0 / (nu_b / CS2 + 0.5)
    return omega_r, omega_b, nu_r, nu_b


# ------------------------------------------------------------------ geometry
def build_domain(rock, buf_in=7, buf_out=6):
    """[piston reservoir | rock | open outlet buffer] along x, solid z/y boundary
    planes over the whole domain (core-holder convention). x=0 is the piston face
    (sealed by porous3d.drain(inlet="piston")). No porous plate.
    Returns (solid bool array, regions dict of half-open x-intervals)."""
    rock = np.asarray(rock, bool)
    nz, ny, nxr = rock.shape
    nx = buf_in + nxr + buf_out
    solid = np.zeros((nz, ny, nx), bool)
    solid[:, :, buf_in : buf_in + nxr] = rock
    solid[0, :, :] = solid[-1, :, :] = True
    solid[:, 0, :] = solid[:, -1, :] = True
    regions = {
        "inbuf": (0, buf_in),
        "rock": (buf_in, buf_in + nxr),
        "outbuf": (buf_in + nxr, nx),
    }
    return solid, regions


# ------------------------------------------------------------------ diagnostics
def spurious_metrics(state, solid_np):
    """(max|u| pore, mean|u| interface band, mean|u| single-phase bulk)."""
    pore = ~solid_np
    u = color3d.velocity(state)
    umag = np.asarray(jnp.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2))
    rhoR, rhoB = color3d.densities(state)
    rhoN = np.asarray(color3d.color_field(rhoR, rhoB))
    iface = pore & (np.abs(rhoN) < 0.9)
    bulk = pore & (np.abs(rhoN) >= 0.9)
    umax = float(umag[pore].max()) if pore.any() else 0.0
    imean = float(umag[iface].mean()) if iface.any() else 0.0
    bmean = float(umag[bulk].mean()) if bulk.any() else 0.0
    return umax, imean, bmean


# ------------------------------------------------------------------ CLI
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--structure", type=Path, required=True, help="gdl bool volume (True=solid)")
    p.add_argument("--out", type=Path, required=True, help="run directory")
    p.add_argument("--name", type=str, default=None, help="run name (default drain_<structure stem>)")
    p.add_argument("--buffer", type=int, default=7, help="inlet slabs: piston face x=0 + reservoir x=1..buffer-1")
    p.add_argument("--buf-out", type=int, default=6, help="open outlet-buffer slabs behind the rock")
    p.add_argument("--ca", type=float, default=CA)
    p.add_argument("--sigma", type=float, default=SIGMA)
    p.add_argument("--beta", type=float, default=BETA, help="recoloring sharpness (dataset campaign: 0.99)")
    p.add_argument("--nu0", type=float, default=NU0)
    p.add_argument("--m", type=float, default=1.0, help="viscosity ratio M = nu_red/nu_blue (geometric split of nu0)")
    p.add_argument("--theta", type=float, default=THETA_ROCK)
    p.add_argument("--chi", type=float, default=CHI)
    p.add_argument("--block", type=int, default=BLOCK, help="steps per block = the log interval (metrics + frame + fields.h5)")
    p.add_argument(
        "--pv-cap",
        type=float,
        default=PV_CAP,
        help="stop after injecting this fraction of the rock pore volume (u_in*A*step ledger)",
    )
    p.add_argument(
        "--bt-x",
        type=float,
        default=BT_X,
        help="stop at this multiple of the breakthrough step (default 1.2); the PV cap still applies",
    )
    p.add_argument("--steps", type=int, default=100_000_000, help="safety cap")
    p.add_argument(
        "--scrub",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="per-block mass/momentum-conserving red->blue swap in the outlet buffer "
        "(broken-through red is produced, never returned; logged cumulatively as 'scrubbed')",
    )
    p.add_argument(
        "--field-vars", type=str, default=FIELD_VARS, help="comma-separated fields.h5 variables (see bob.utils.file.FIELDS)"
    )
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--plots", action="store_true",
                   help="also write the analysis curves (pressure_curve.svg, ledger_curve.svg, spurious_curve.svg, "
                        "saturation.svg); off by default")
    p.add_argument("--rock-alpha", type=float, default=0.04)
    p.add_argument("--resume", type=Path, default=None, help="run dir (newest ckpt) or ckpt .npz")
    p.add_argument("--gpus", type=int, default=1, help="shard the domain along z across the first N devices (bob.multigpu)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.plots:
        viz.enable_plots(True)
    name = args.name or f"drain_{args.structure.stem}"
    u_in = args.ca * args.sigma / args.nu0  # nu0-referenced Ca: injection rate independent of M
    omega, omega2, nu_r, nu_b = omega_pair(args.m, args.nu0)
    out = args.out
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # ---- geometry -------------------------------------------------------------
    rock = porous3d.load_structure(args.structure)
    solid_np, reg = build_domain(rock, buf_in=args.buffer, buf_out=args.buf_out)
    solid = jnp.asarray(solid_np)
    nz, ny, nx = solid_np.shape
    phi_rock = 1.0 - float(rock.mean())
    x_rock0, x_rock1 = reg["rock"]
    x_ob = reg["outbuf"][0]
    title = f"{name} (Ca={args.ca:.0e}, M={args.m:g}, theta={args.theta:g}, piston, no plate)"

    # piston: seal x=0 for the wall normals so the face-adjacent slab x=1 carries
    # normals for the wetting BC; theta_face there makes the face purely red-wetting.
    sealed_np = solid_np.copy()
    sealed_np[:, :, 0] = True
    pore = ~sealed_np
    a_open = int(pore[:, :, 1].sum())
    pore_in = pore.copy()
    pore_in[:, :, 0] = False
    pore_in[:, :, -1] = False
    xx = np.arange(nx)[None, None, :]
    rock_m = pore & (xx >= x_rock0) & (xx < x_rock1)  # rock-only pore mask
    outbuf_m = pore & (xx >= x_ob)  # pore behind the rock

    # ---- termination cap (see module docstring) -------------------------------
    pv_rock = int(rock_m.sum())  # rock pore volume in the assembled (wall-sealed) domain — same denominator as saturation
    steps_pv = int(np.ceil(args.pv_cap * pv_rock / (u_in * a_open) / args.block)) * args.block
    print(f"termination: whichever comes first — {args.bt_x:g}x breakthrough, or {args.pv_cap:g} PV = {steps_pv} steps "
          f"(rock pore volume {pv_rock}); no plate, no dp cap")
    nw = color3d.wall_normals(jnp.asarray(sealed_np))
    np.savez_compressed(
        out / "wall_normals.npz", nw=np.asarray(nw), structure=args.structure.name, buffer=args.buffer, piston=True
    )
    theta_field = jnp.full((nz, ny, nx), float(args.theta)).at[:, :, 1].set(float(THETA_FACE))
    camera = viz3d.flow_camera(solid_np.shape)

    # ---- init / resume --------------------------------------------------------
    broke = None  # breakthrough step; rides inside every checkpoint (resume keeps the anchor)
    if args.resume is not None:
        ckpt_path = stats.latest_checkpoint(args.resume) if args.resume.is_dir() else args.resume
        if ckpt_path is None:
            sys.exit(f"no checkpoint under {args.resume}")
        ckpt = np.load(ckpt_path)
        state = color3d.State(jnp.asarray(ckpt["fR"]), jnp.asarray(ckpt["fB"]))
        step0 = int(ckpt["step"])
        if "broke" in ckpt and int(ckpt["broke"]) >= 0:
            broke = int(ckpt["broke"])
        print(f"resumed {ckpt_path.name}: step {step0}, breakthrough {broke}")
    else:
        state = porous3d.init_drainage(nz, ny, nx, n_red=args.buffer)
        step0 = 0
        for stale in frames_dir.glob("frame_*.png"):
            stale.unlink()

    # ---- standardized run practice: RunMeta + RunMonitor + FieldWriter --------
    fields = (
        "step",
        "saturation",
        "sat_all",
        "front",
        "m_red",
        "ideal",
        "p_in",
        "p_mid",
        "p_end",
        "dp",
        "u_max",
        "u_iface",
        "u_bulk",
        "outbuf_red",
        "scrubbed",
    )
    meta = RunMeta(
        out / "run_meta.json",
        params=color3d.Params(omega=omega, omega2=omega2, sigma=args.sigma, beta=args.beta, theta=float(args.theta), mrt_chi=args.chi),
        solid=solid_np,
        geometry_source=str(args.structure),
        resume_step=step0,
        target_steps=min(steps_pv, args.steps),
        extra={
            "protocol": f"constant-rate drainage to {args.bt_x:g}x breakthrough or {args.pv_cap:g} PV injected, "
            f"whichever first (no porous plate, no dp cap)",
            "bt_x": float(args.bt_x),
            "pv_rock": pv_rock,
            "steps_pv": steps_pv,
            "ca": float(args.ca),
            "M": float(args.m),
            "nu_red": float(nu_r),
            "nu_blue": float(nu_b),
            "u_in": float(u_in),
            "inlet": "piston",
            "outlet": "zou_he blue-pinned open outlet (no porous plate)",
            "theta_face": float(THETA_FACE),
            "breakthrough": "red-majority pore cell in the last rock slab (outlet contact) — the timing anchor; "
            "breakthrough_step / breakthrough_sat are filled in when it fires",
            "breakthrough_step": None,
            "breakthrough_sat": None,
            "block": int(args.block),
            "regions": {k: list(v) for k, v in reg.items()},
        },
    )
    rows = stats.carry_rows(out, step0, fields=fields) if step0 else None
    if rows:  # backfill fields added since the carried segment
        rows = [{**dict.fromkeys(fields, 0.0), **r} for r in rows]
    monitor = stats.RunMonitor(out, title=title, rows=rows, meta=meta)
    fw = FieldWriter(
        out / f"{name}.h5",  # h5/xdmf named after the run: geometry + M + theta in the filename
        solid_np,
        fields=tuple(v for v in args.field_vars.split(",") if v),
        mask_solid=True,
        attrs={
            "ca": float(args.ca),
            "M": float(args.m),
            "nu_red": float(nu_r),
            "nu_blue": float(nu_b),
            "u_in": float(u_in),
            "sigma": float(args.sigma),
            "nu0": float(args.nu0),
            "theta": float(args.theta),
            "chi": float(args.chi),
            "inlet": "piston",
        },
        meta=meta,
    )

    # ---- solver ---------------------------------------------------------------
    p = color3d.Params(omega=omega, omega2=omega2, sigma=args.sigma, beta=args.beta, theta=theta_field, mrt_chi=args.chi)
    step_fn = None
    if args.gpus > 1:
        m = multigpu.mesh(args.gpus)
        print(f"sharding along z over {args.gpus} devices: {[str(d) for d in m.devices.reshape(-1)]}")
        # default multi-GPU backend: staged thin-halo shard_map (2169 MLUPS / 90%
        # efficiency on 8xH100 vs ~63% for plain GSPMD — demo/multi_gpu_halo).
        # Statics go in unsharded (it pre-pads them); the piston seals x=0 inside
        # drain, so the step mask must be sealed the same way.
        step_solid = np.asarray(solid).copy()
        step_solid[:, :, 0] = True
        step_fn = multigpu.staged_halo_step(m, p, step_solid, nw=nw)
        state, solid, nw, theta_field = multigpu.shard((state, solid, nw, theta_field), m)

    sharding = jax.tree_util.tree_map(lambda x: x.sharding, state)
    blk = int(args.block)
    runner = jax.jit(
        lambda s: porous3d.drain(s, p, solid, 0, u_in, blk, nw=nw, inlet="piston", outlet_sa_red=0.0, step_fn=step_fn),
        out_shardings=sharding,
        donate_argnums=0,
    )

    scrub_tot = (monitor.rows[-1].get("scrubbed", 0.0) if monitor.rows else 0.0) or 0.0

    def scrub(state):
        """Production side (no membrane): swap red that left the rock into blue,
        mass- and momentum-conserving per population."""
        nonlocal scrub_tot
        red = float(np.asarray(state.fR[:, :, :, x_ob:]).sum())
        if red > 1e-12:
            fB = state.fB.at[:, :, :, x_ob:].add(state.fR[:, :, :, x_ob:])
            fR = state.fR.at[:, :, :, x_ob:].set(0.0)
            state = color3d.State(fR, fB)
            scrub_tot += red
        return state

    def record(step):
        rhoR, rhoB = color3d.densities(state)
        rR, rho = np.asarray(rhoR), np.asarray(rhoR + rhoB)
        sat_rock = float(rR[rock_m].sum() / max(rho[rock_m].sum(), 1e-9))
        p_in = float(rho[:, :, 1][pore[:, :, 1]].mean()) / 3.0
        x_mid = (x_rock0 + x_rock1) // 2
        p_mid = float(rho[:, :, x_mid][pore[:, :, x_mid]].mean()) / 3.0
        p_end = float(rho[:, :, -2][pore[:, :, -2]].mean()) / 3.0
        umax, u_iface, u_bulk = spurious_metrics(state, sealed_np)
        monitor.log(
            step=step,
            saturation=sat_rock,  # rock-only: the drainage curve
            sat_all=float(rR[pore_in].sum() / max(rho[pore_in].sum(), 1e-9)),
            front=porous3d.invasion_front(state, solid),
            m_red=float(rR[pore_in].sum()),
            ideal=u_in * a_open * step,
            p_in=p_in,
            p_mid=p_mid,
            p_end=p_end,
            dp=p_in - p_end,
            u_max=umax,
            u_iface=u_iface,
            u_bulk=u_bulk,
            outbuf_red=float(rR[outbuf_m].sum()),  # pre-scrub arrival behind the rock
            scrubbed=scrub_tot,
        )
        try:  # PyVista needs offscreen GL; rendering is best-effort
            viz3d.phase_frame(
                frames_dir / f"frame_{step:08d}.png",
                np.asarray(color3d.color_field(rhoR, rhoB)),
                sealed_np,
                rock_opacity=args.rock_alpha,
                camera=camera,
                zoom=1.0,
            )
        except Exception as e:
            if step == step0:
                print(f"  [3D render unavailable: {type(e).__name__}: {e}; metrics/field IO continues]")
        return sat_rock

    def gif():
        frames = sorted(frames_dir.glob("frame_*.png"))
        if not frames:
            return
        if len(frames) > 300:  # watchability cap; every frame stays on disk
            idx = np.unique(np.linspace(0, len(frames) - 1, 300).round().astype(int))
            frames = [frames[i] for i in idx]
        viz.write_gif(frames, out / f"{name}.gif", args.fps)

    def pressure_curve():
        if not viz.plots_enabled():
            return
        rs = monitor.rows
        steps_ = [r["step"] for r in rs]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for key, c in (("p_in", "tab:green"), ("p_mid", "tab:blue"), ("p_end", "tab:purple")):
            ax[0].plot(steps_, [r[key] for r in rs], color=c, lw=1.8, label=key)
        ax[0].set(xlabel="step", ylabel="pressure  cs²·ρ", title="pressure stations")
        ax[1].plot(steps_, [r["dp"] for r in rs], color="tab:red", lw=2)
        ax[1].set(xlabel="step", ylabel="p_in − p_end", title="entry pressure buildup")
        for a in ax:
            a.grid(True, alpha=0.3)
        ax[0].legend(fontsize=8)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out / "pressure_curve.svg")
        plt.close(fig)

    def ledger_curve():
        if not viz.plots_enabled():
            return
        rs = monitor.rows
        steps_ = np.array([r["step"] for r in rs], float)
        gained = np.array([r["m_red"] + r["scrubbed"] - rs[0]["m_red"] - rs[0]["scrubbed"] for r in rs])
        ideal = np.array([r["ideal"] - rs[0]["ideal"] for r in rs])
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(steps_, gained, color="tab:red", lw=2, label="actual  ΔM_red + scrubbed")
        ax[0].plot(steps_, ideal, "--", color="tab:green", lw=2, label="ideal  u_in·A·t")
        ax[0].set(xlabel="step", ylabel="red mass increase", title="colour ledger")
        ax[0].legend(fontsize=9)
        ratio = gained / np.where(ideal > 0, ideal, np.nan)
        ax[1].plot(steps_, ratio, color="tab:orange", lw=2)
        ax[1].axhline(1.0, color="0.3", ls="--", lw=1.2)
        ax[1].set_ylim(0.5, 1.5)
        ax[1].set(xlabel="step", ylabel="actual / ideal", title="ledger ratio (leak-free = 1.0)")
        for a in ax:
            a.grid(True, alpha=0.3)
        fig.suptitle(f"{title}: injected red vs ideal")
        fig.tight_layout()
        fig.savefig(out / "ledger_curve.svg")
        plt.close(fig)

    def spurious_curve():
        if not viz.plots_enabled():
            return
        rs = monitor.rows
        steps_ = [r["step"] for r in rs]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(steps_, [r["u_max"] for r in rs], "-o", ms=3, label="max|u| (pore)")
        ax.plot(steps_, [r["u_iface"] for r in rs], "-s", ms=3, label="mean|u| interface")
        ax.plot(steps_, [r["u_bulk"] for r in rs], "-^", ms=3, label="mean|u| bulk")
        ax.axhline(u_in, color="cyan", lw=1.5, ls="--", label=f"u_in={u_in:.2e}")
        ax.set(xlabel="step", ylabel="|u| (lattice units)", title=f"{title} — spurious current")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "spurious_curve.svg")
        plt.close(fig)

    def curves():
        ledger_curve()
        pressure_curve()
        spurious_curve()

    def checkpoint(step, sat):
        monitor.checkpoint(
            step, sat, fR=np.asarray(state.fR), fB=np.asarray(state.fB), broke=np.int64(broke if broke is not None else -1)
        )

    # ---- main loop: to 1.2x breakthrough or the PV cap, whichever first ------
    # rebuilt from the carried anchor on resume, so a resumed segment still honours the
    # bt cap instead of silently running on to the PV cap
    steps_bt = None if broke is None else int(np.ceil(args.bt_x * broke / blk)) * blk
    if steps_bt is not None:
        meta.target_steps = min(steps_pv, steps_bt, args.steps)
    step = step0
    sat = record(step)
    curves()
    fw.append(step, state)
    sat_at_breakthrough = None
    finish_type, finish_detail = "step_cap", f"hit the --steps safety cap ({args.steps})"
    blocks_done = 0
    start = time.perf_counter()
    while step < args.steps:
        state = runner(state)
        step += blk
        blocks_done += 1
        sat = record(step)
        curves()
        fw.append(step, state)  # every block = every args.block steps
        if args.scrub:
            state = scrub(state)
        if not np.isfinite(sat):
            finish_type = "diverged"
            finish_detail = f"non-finite saturation at step {step}"
            print(
                f"{name}: non-finite saturation at step {step} — diverged; stopping early. "
                f"Watch u_max in metrics.csv."
            )
            break
        if broke is None and porous3d.breakthrough(state, solid, x=x_rock1 - 1):
            broke = step  # red reached the last rock slab: outlet contact — the timing anchor
            sat_at_breakthrough = sat
            steps_bt = int(np.ceil(args.bt_x * broke / blk)) * blk
            meta.target_steps = min(steps_pv, steps_bt, args.steps)  # retime the ETA off the anchor
            print(f"outlet breakthrough at step {step} (rock sat {sat:.3f}); "
                  f"{args.bt_x:g}x -> stop at {steps_bt} (PV cap {steps_pv})")
            gif()
            checkpoint(step, sat)
        if steps_bt is not None and step >= steps_bt:
            finish_type = "bt_cap"
            finish_detail = f"{args.bt_x:g}x breakthrough ({broke}) = {steps_bt} steps reached at step {step}"
            print(f"{name}: breakthrough cap — {finish_detail}")
            break
        if step >= steps_pv:
            finish_type = "pv_cap"
            finish_detail = f"{args.pv_cap:g} rock pore volumes injected at step {step}"
            print(f"{name}: PV cap — {finish_detail}")
            break
        if blocks_done % 10 == 0:
            gif()
            checkpoint(step, sat)
    elapsed = time.perf_counter() - start

    checkpoint(step, sat)
    fw.close()
    gif()
    curves()
    meta.finish(
        step,
        finish_type=finish_type,
        finish_detail=finish_detail,
        breakthrough_step=None if broke is None else int(broke),
        breakthrough_sat=None if sat_at_breakthrough is None else float(sat_at_breakthrough),
        steps_bt=steps_bt,
    )

    rs = monitor.rows
    gained = rs[-1]["m_red"] + rs[-1]["scrubbed"] - rs[0]["m_red"] - rs[0]["scrubbed"]
    ideal = rs[-1]["ideal"] - rs[0]["ideal"]
    ratio_T = None
    if broke is not None:
        pre = [r for r in rs if r["step"] <= broke]
        if len(pre) > 1 and (pre[-1]["ideal"] - pre[0]["ideal"]) > 0:
            ratio_T = (pre[-1]["m_red"] - pre[0]["m_red"]) / (pre[-1]["ideal"] - pre[0]["ideal"])
    out_max = max((r["outbuf_red"] for r in rs), default=0.0)
    (out / "report.md").write_text(
        f"# {title}\n\n"
        f"- Ca {args.ca:.2e} | M = {args.m:g} (nu_red {nu_r:.4f}, nu_blue {nu_b:.4f}) | u_in {u_in:.3e} | "
        f"sigma {args.sigma} | nu0 {args.nu0} | mrt_chi {args.chi} | precision {'fp32' if FP32 else 'fp64'}\n"
        f"- Structure {args.structure.name} | rock porosity {phi_rock:.3f} | domain {nz}x{ny}x{nx} "
        f"(piston x=0, reservoir x=1..{args.buffer - 1}, rock x={x_rock0}..{x_rock1 - 1}, "
        f"outlet buffer x={x_ob}..{nx - 1}, no porous plate)\n"
        f"- Termination: {args.bt_x:g}x breakthrough"
        + (f" ({steps_bt} steps)" if steps_bt is not None else " (not reached)")
        + f" or {args.pv_cap:g} PV injected ({steps_pv} steps), whichever first; no dp cap | "
        f"log interval {blk} steps (metrics + frame + fields.h5: {args.field_vars})\n"
        f"- Finished: **{finish_type}** — {finish_detail}\n"
        f"- Steps {rs[-1]['step']} | runtime {elapsed / 3600:.2f} h this segment\n"
        f"- **Breakthrough: step {broke if broke is not None else 'not reached'}**"
        + (f" (rock saturation {sat_at_breakthrough:.3f})" if sat_at_breakthrough is not None else "")
        + "\n\n"
        f"## Result\n\n"
        f"- Rock saturation {rs[0]['saturation']:.3f} -> {rs[-1]['saturation']:.3f}"
        f"{f' | at breakthrough {sat_at_breakthrough:.3f}' if sat_at_breakthrough is not None else ''}\n"
        f"- Colour ledger (ΔM_red + scrubbed vs injected): {gained / max(ideal, 1e-12):.3f}x whole-run"
        + (f"; {ratio_T:.3f}x at breakthrough (leak-free check)\n" if ratio_T is not None else "\n")
        + f"- Entry pressure p_in - p_end: {rs[0]['dp']:+.5f} -> {rs[-1]['dp']:+.5f}\n"
        + f"- Max red in outlet buffer (per block, pre-scrub): {out_max:.3e} | total produced (scrubbed) {scrub_tot:.3e}\n",
        encoding="utf-8",
    )
    print(
        f"{name}: rock sat {rs[0]['saturation']:.3f}->{rs[-1]['saturation']:.3f}, "
        f"breakthrough {broke}, ledger {gained / max(ideal, 1e-12):.3f}x, "
        f"produced {scrub_tot:.2e}, output {out}"
    )


if __name__ == "__main__":
    main()
