"""Constant-rate capillary drainage of a porous rock with a porous-plate outlet.

The standardized drainage driver of the case/drainage campaign. One standalone
file holds the whole setting so every run is identical in physics, geometry
assembly, logging and metadata.

Physics:
  piston inlet (u_in*A red/step exact, inlet pressure floats), sigma 0.05,
  baseline nu0 0.04 split geometrically for the viscosity ratio
  M = nu_red/nu_blue (--m: nu_r = nu0*sqrt(M), nu_b = nu0/sqrt(M)),
  Ca = 1e-5 nu0-referenced (u_in = Ca*sigma/nu0, independent of M), theta_rock
  via --theta (red = invading non-wetting), MRT chi 0.8, fp32, solver defaults
  MRT + akai + CSF + recolor_emag + wall_grad="fluid". The recoloring sharpness
  --beta defaults to 0.7; the dataset campaign passes --beta 0.99
  (case/run_array.slurm).

Geometry (core-holder layout):
  [piston face x=0 | inlet reservoir | rock | porous plate | outlet buffer]
  along x, solid no-slip z/y walls over the whole domain. The plate (5x5-voxel
  square pores on a pitch-8 lattice, 20 thick, theta 150 deg water-wet, in
  direct contact with the rock outlet face) seals the outlet buffer against
  red like a laboratory semi-permeable membrane; a per-block scrubber converts
  any red mist past the plate to blue (mass- and momentum-conserving, logged
  cumulatively) so broken-through oil is produced, never returned. The pore
  width must exceed the ~4-5 voxel diffuse-interface width so the meniscus is
  resolved, and the plate must be long enough for the dissolved-red mist
  gradient to decay before the scrub/outlet sinks; a thinner plate with
  narrower pores holds capillary pressure but leaks the injected red as
  dissolved mist after breakthrough.

Run protocol (termination — volume- and pressure-capped):
  The plate seals the outlet, so injection cannot run past what the rock can
  hold; the run stops at whichever cap comes first:
  1. PV cap (finish_type "pv_cap"): the injected red volume u_in*A*step
     reaches --pv-cap (default 0.75) of the rock pore volume. The target step
     is known at t=0, so the ETA in run_meta.json is live from the start.
  2. Plate-pressure cap (finish_type "dp_cap"): dp = p_in - p_end reaches
     --dp-cap (default 0.5) of the plate entry pressure
     Pc_plate = 2*sigma*|cos(theta_plate)|/(hole/2); beyond that the plate
     itself is at risk of draining and the experiment stops being a valid
     porous-plate run.
  Breakthrough (red first reaching the last rock slab, in contact with the
  plate face) is detected, logged and checkpointed, but does not terminate the
  run. The finish type + detail land in run_meta.json (extra section) and
  report.md; abnormal exits record "diverged" or "step_cap".

Logging (every --block = 5000 steps): metrics.csv row + saturation curve +
3D phase-field frame (GIF) + fields .h5/.xdmf append (phi, rho, p, umag, u)
+ run_meta.json heartbeat (bob.utils.file.RunMeta). Status-named checkpoints
every 10 blocks.

Usage (from the repository root):
  .venv/bin/python case/drainage/drainage_demo.py \
      --structure <rock.npy> --out case/drainage/runs/<name> --name <name> \
      --m 0.2 --theta 140 --beta 0.99
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
THETA_PLATE = 150.0  # porous plate + outlet buffer: strongly water-wet (at 170 the red detaches from the plate face)
CHI = 0.8  # MRT chi spectrum (Leclaire), the dual-viscosity fix
BETA = 0.7
BLOCK = 5000  # the log interval: metrics + GIF frame + fields.h5 cadence
PV_CAP = 0.75  # stop after injecting this fraction of the rock pore volume
DP_CAP = 0.5  # stop when dp reaches this fraction of the plate entry pressure
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
def build_plate(nz, ny, thick=20, hole=5, pitch=8, offset=3):
    """Water-wet porous plate: solid slab pierced by hole x hole square through-pores
    on a pitch lattice starting at ``offset`` (so the mesh clears the z/y walls).
    Default 5-voxel pores, 20 long, theta 150: the meniscus must be resolved
    (pore width > interface width ~4-5 voxels) and the pore long enough that the
    dissolved-red (recolor mist) gradient decays before the outlet sinks. A
    2x2-pore, 3-thick, theta-170 plate holds capillary pressure but leaks the
    injected red as mist and stalls drainage after breakthrough."""
    plate = np.ones((nz, ny, thick), bool)
    for z0 in range(offset, nz, pitch):
        for y0 in range(offset, ny, pitch):
            plate[z0 : z0 + hole, y0 : y0 + hole, :] = False
    return plate


def build_domain(rock, buf_in=7, buf_out=6, thick=20, hole=5, pitch=8, offset=3):
    """[piston reservoir | rock | plate (direct contact) | outlet buffer] along x,
    solid z/y boundary planes over the whole domain (core-holder convention).
    x=0 is the piston face (sealed by porous3d.drain(inlet="piston")).
    Returns (solid bool array, regions dict of half-open x-intervals)."""
    rock = np.asarray(rock, bool)
    nz, ny, nxr = rock.shape
    nx = buf_in + nxr + thick + buf_out
    solid = np.zeros((nz, ny, nx), bool)
    solid[:, :, buf_in : buf_in + nxr] = rock
    solid[:, :, buf_in + nxr : buf_in + nxr + thick] = build_plate(nz, ny, thick, hole, pitch, offset)
    solid[0, :, :] = solid[-1, :, :] = True
    solid[:, 0, :] = solid[:, -1, :] = True
    regions = {
        "inbuf": (0, buf_in),
        "rock": (buf_in, buf_in + nxr),
        "plate": (buf_in + nxr, buf_in + nxr + thick),
        "outbuf": (buf_in + nxr + thick, nx),
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
    p.add_argument("--structure", type=Path, required=True, help="rock bool volume .npy (True=solid)")
    p.add_argument("--out", type=Path, required=True, help="run directory")
    p.add_argument("--name", type=str, default=None, help="run name (default drain_<structure stem>)")
    p.add_argument("--buffer", type=int, default=7, help="inlet slabs: piston face x=0 + reservoir x=1..buffer-1")
    p.add_argument("--buf-out", type=int, default=6, help="open outlet-buffer slabs behind the plate")
    p.add_argument("--plate-thick", type=int, default=20)
    p.add_argument("--hole", type=int, default=5, help="plate pore width (voxels; must exceed the ~4-5 vox interface width)")
    p.add_argument("--pitch", type=int, default=8, help="plate pore pitch (voxels)")
    p.add_argument("--hole-offset", type=int, default=3, help="first pore corner (clears the z/y walls)")
    p.add_argument("--ca", type=float, default=CA)
    p.add_argument("--sigma", type=float, default=SIGMA)
    p.add_argument("--beta", type=float, default=BETA, help="recoloring sharpness (dataset campaign: 0.99)")
    p.add_argument("--nu0", type=float, default=NU0)
    p.add_argument("--m", type=float, default=1.0, help="viscosity ratio M = nu_red/nu_blue (geometric split of nu0)")
    p.add_argument("--theta", type=float, default=THETA_ROCK)
    p.add_argument("--theta-plate", type=float, default=THETA_PLATE)
    p.add_argument("--chi", type=float, default=CHI)
    p.add_argument("--block", type=int, default=BLOCK, help="steps per block = the log interval (metrics + frame + fields.h5)")
    p.add_argument(
        "--pv-cap",
        type=float,
        default=PV_CAP,
        help="stop after injecting this fraction of the rock pore volume (u_in*A*step ledger)",
    )
    p.add_argument(
        "--dp-cap",
        type=float,
        default=DP_CAP,
        help="stop when dp = p_in - p_end reaches this fraction of the plate entry "
        "pressure Pc = 2*sigma*|cos(theta_plate)|/(hole/2)",
    )
    p.add_argument("--steps", type=int, default=100_000_000, help="safety cap")
    p.add_argument(
        "--scrub",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="per-block mass/momentum-conserving red->blue swap past the plate "
        "(membrane suction side; logged cumulatively as 'scrubbed')",
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
    solid_np, reg = build_domain(
        rock, buf_in=args.buffer, buf_out=args.buf_out, thick=args.plate_thick, hole=args.hole, pitch=args.pitch,
        offset=args.hole_offset,
    )
    solid = jnp.asarray(solid_np)
    nz, ny, nx = solid_np.shape
    phi_rock = 1.0 - float(rock.mean())
    x_rock0, x_rock1 = reg["rock"]
    x_plate0 = reg["plate"][0]
    x_ob = reg["outbuf"][0]
    title = f"{name} (Ca={args.ca:.0e}, M={args.m:g}, theta={args.theta:g}, piston + porous plate)"

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
    outbuf_m = pore & (xx >= x_ob)  # pore past the plate

    # ---- termination caps (see module docstring) ------------------------------
    pv_rock = int(rock_m.sum())  # rock pore volume in the assembled (wall-sealed) domain — same denominator as saturation
    steps_pv = int(np.ceil(args.pv_cap * pv_rock / (u_in * a_open) / args.block)) * args.block
    pc_plate = 2.0 * args.sigma * abs(np.cos(np.radians(args.theta_plate))) / (args.hole / 2.0)
    dp_stop = args.dp_cap * pc_plate
    print(
        f"termination: {args.pv_cap:g} PV cap = {steps_pv} steps (rock pore volume {pv_rock}), "
        f"or dp >= {dp_stop:.5f} ({args.dp_cap:g} x plate entry Pc {pc_plate:.5f})"
    )
    nw = color3d.wall_normals(jnp.asarray(sealed_np))
    np.savez_compressed(
        out / "wall_normals.npz", nw=np.asarray(nw), structure=args.structure.name, buffer=args.buffer, piston=True
    )
    theta_field = (
        jnp.full((nz, ny, nx), float(args.theta))
        .at[:, :, 1]
        .set(float(THETA_FACE))
        .at[:, :, x_plate0:]
        .set(float(args.theta_plate))
    )
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
        "plate_leak",
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
            "protocol": f"constant-rate drainage to {args.pv_cap:g} PV injected or dp >= {args.dp_cap:g}x plate entry Pc",
            "pv_rock": pv_rock,
            "steps_pv": steps_pv,
            "pc_plate": pc_plate,
            "dp_stop": dp_stop,
            "ca": float(args.ca),
            "M": float(args.m),
            "nu_red": float(nu_r),
            "nu_blue": float(nu_b),
            "u_in": float(u_in),
            "inlet": "piston",
            "outlet": "zou_he blue-pinned behind porous plate",
            "plate": f"{args.hole}x{args.hole} holes pitch {args.pitch} thick {args.plate_thick}",
            "theta_plate": float(args.theta_plate),
            "theta_face": float(THETA_FACE),
            "breakthrough": "red-majority pore cell in the last rock slab (plate contact) — logged, not a stop criterion",
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
            "theta_plate": float(args.theta_plate),
            "chi": float(args.chi),
            "inlet": "piston",
            "plate_x0": x_plate0,
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
        """Membrane suction side (as in demo/scal): swap red past the plate into blue,
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
            plate_leak=float(rR[outbuf_m].sum()),  # pre-scrub arrival past the plate
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
        gained = np.array([r["m_red"] - rs[0]["m_red"] for r in rs])
        ideal = np.array([r["ideal"] - rs[0]["ideal"] for r in rs])
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(steps_, gained, color="tab:red", lw=2, label="actual  ΔM_red downstream")
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

    # ---- main loop: run to the PV cap or the plate-pressure cap ----------------
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
            broke = step  # red reached the last rock slab: plate contact (logged, not a stop)
            sat_at_breakthrough = sat
            print(f"plate breakthrough at step {step} (rock sat {sat:.3f})")
            gif()
            checkpoint(step, sat)
        dp_now = monitor.rows[-1]["dp"]
        if dp_now >= dp_stop:
            finish_type = "dp_cap"
            finish_detail = f"dp {dp_now:.5f} >= {args.dp_cap:g} x plate entry Pc {pc_plate:.5f} at step {step}"
            print(f"{name}: plate-pressure cap — {finish_detail}")
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
    meta.finish(step, finish_type=finish_type, finish_detail=finish_detail)

    rs = monitor.rows
    gained = rs[-1]["m_red"] - rs[0]["m_red"]
    ideal = rs[-1]["ideal"] - rs[0]["ideal"]
    ratio_T = None
    if broke is not None:
        pre = [r for r in rs if r["step"] <= broke]
        if len(pre) > 1 and (pre[-1]["ideal"] - pre[0]["ideal"]) > 0:
            ratio_T = (pre[-1]["m_red"] - pre[0]["m_red"]) / (pre[-1]["ideal"] - pre[0]["ideal"])
    leak_max = max((r["plate_leak"] for r in rs), default=0.0)
    (out / "report.md").write_text(
        f"# {title}\n\n"
        f"- Ca {args.ca:.2e} | M = {args.m:g} (nu_red {nu_r:.4f}, nu_blue {nu_b:.4f}) | u_in {u_in:.3e} | "
        f"sigma {args.sigma} | nu0 {args.nu0} | mrt_chi {args.chi} | precision {'fp32' if FP32 else 'fp64'}\n"
        f"- Structure {args.structure.name} | rock porosity {phi_rock:.3f} | domain {nz}x{ny}x{nx} "
        f"(piston x=0, reservoir x=1..{args.buffer - 1}, rock x={x_rock0}..{x_rock1 - 1}, "
        f"plate x={x_plate0}..{x_ob - 1}, outlet buffer x={x_ob}..{nx - 1})\n"
        f"- Porous plate: {args.hole}x{args.hole} holes, pitch {args.pitch}, thick {args.plate_thick}, "
        f"theta {args.theta_plate:.0f} deg | rock theta {args.theta:.0f} deg\n"
        f"- Termination: {args.pv_cap:g} PV injected ({steps_pv} steps) or dp >= {dp_stop:.5f} "
        f"({args.dp_cap:g} x plate entry Pc {pc_plate:.5f}) | log interval {blk} steps "
        f"(metrics + frame + fields.h5: {args.field_vars})\n"
        f"- Finished: **{finish_type}** — {finish_detail}\n"
        f"- Steps {rs[-1]['step']} | runtime {elapsed / 3600:.2f} h this segment | plate breakthrough at "
        f"{broke if broke is not None else 'not reached'}\n\n"
        f"## Result\n\n"
        f"- Rock saturation {rs[0]['saturation']:.3f} -> {rs[-1]['saturation']:.3f}"
        f"{f' | at breakthrough {sat_at_breakthrough:.3f}' if sat_at_breakthrough is not None else ''}\n"
        f"- Colour ledger: {gained / max(ideal, 1e-12):.3f}x whole-run"
        + (f"; {ratio_T:.3f}x at breakthrough (leak-free check)\n" if ratio_T is not None else "\n")
        + f"- Entry pressure p_in - p_end: {rs[0]['dp']:+.5f} -> {rs[-1]['dp']:+.5f}\n"
        + f"- Max plate leak (red mass past plate, per block): {leak_max:.3e} | total scrubbed {scrub_tot:.3e}\n",
        encoding="utf-8",
    )
    print(
        f"{name}: rock sat {rs[0]['saturation']:.3f}->{rs[-1]['saturation']:.3f}, "
        f"breakthrough {broke}, ledger {gained / max(ideal, 1e-12):.3f}x, "
        f"plate_leak_max {leak_max:.2e}, output {out}"
    )


if __name__ == "__main__":
    main()
