"""Lenormand drainage phase-diagram regimes through the 256^3 synthetic sphere pack (3D),
piston-inlet driver.

Reproduces the three immiscible-displacement regimes of Lenormand, Touboul & Zarcone
(JFM 1988) on one full geometry — the monodisperse sphere pack
``test/assets/sphere256.npy`` — by moving only two control parameters:

  * capillary number   Ca = nu0 * u_in / sigma          (x-axis; via injection speed u_in)
  * viscosity ratio    M  = nu_red / nu_blue            (y-axis; via params.omega (invading
                          = mu_invading / mu_defending           red) vs params.omega2 (defending blue))

  regime       Ca     M        front morphology
  ----------   ----   ------   -----------------------------------------------------
  stable       high   >> 1     compact, ~flat front (viscous invader sweeps everything)
  viscous      high   << 1     thin branched fingers (low-viscosity invader screens)
  capillary    low    ~  1     isotropic invasion-percolation clusters (capillarity rules)

Boundary conditions and solver settings:

  - **piston inlet** (``drain(inlet="piston")``, MF-LBM's default bounce-back velocity
    BC, colour-leak-free): the x=0 plane is sealed as the piston face, the Ladd kick
    injects exactly u_in*A red/step and zero blue, and the inlet pressure is the free
    output that builds against the entry barriers;
  - **blue-pinned pressure outlet** (``outlet_sa_red=0.0``): the default zero-gradient
    colour split is self-referential and accumulates a spurious red pool at the outlet —
    the pin keeps the colour ledger at 1.00x;
  - reservoir layers x=1..buffer-1 seeded pure red at t=0 only (never re-pinned; the
    piston sustains the column physically), piston face purely red-wetting via a
    per-cell theta field (theta_face=0 at the face-adjacent slab x=1);
  - **mrt_chi=0.8** (Leclaire MRT spectrum): ties the bulk/ghost moments to 0.8*omega.
    Required for the M=30/0.01 dual-viscosity presets: the default MRT spectrum leaves
    ghost rates at 1.98, which are under-damped when one phase's omega approaches 2.

Ca=1e-1 is not reachable (u_in=0.05, Ma~0.087 — a compressibility blow-up that chi does
not address), so the high-Ca regimes sit at Ca=1e-3 and the capillary regime at Ca=1e-5
(two decades below). The run stops at breakthrough (invading red reaches the outlet) —
the canonical point at which Lenormand morphologies are compared; --steps is only a
safety cap.

Usage:
  uv run python demo/lenormand/lenormand_demo.py --regime stable      # Ca 1e-3, M=30:   compact front
  uv run python demo/lenormand/lenormand_demo.py --regime viscous     # Ca 1e-3, M=0.01: viscous fingers
  uv run python demo/lenormand/lenormand_demo.py --regime capillary   # Ca 1e-5, M=30:   capillary fingers (long)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import jax

# float32 is the default (matching bob's global default): LBM is bandwidth-bound,
# so halving the word size is ~2x steps/s and half the state memory. Validated
# at parity with float64 on a 64^3 crop of the pack (breakthrough step, morphology,
# u_max all within run-to-run noise; ledger drift ~0.3%/1M steps). BOB_FP64=1 opts
# back into float64. Decided before importing bob (d3q19 builds its arrays at import).
FP32 = os.environ.get("BOB_FP64") != "1"
jax.config.update("jax_enable_x64", not FP32)

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bob import color3d, d3q19, multigpu, porous3d
from bob.utils import (
    stats,
    viz,
    viz3d,
)
from bob.utils.file import FieldWriter

ASSET = Path(__file__).resolve().parents[2] / "test" / "assets" / "sphere256.npy"
CS2 = float(d3q19.CS2)
NU0 = 0.1  # baseline kinematic viscosity (geometric-mean of the M split); omega(NU0) ~ 1.25
SIGMA = 0.05  # surface-tension parameter (CSF)

# Canonical phase-diagram corner points, stable with mrt_chi=0.8:
# stable<->viscous isolate the M axis at Ca=1e-3;
# stable<->capillary isolate the Ca axis at M=30. Ca is the nu0-referenced capillary number
# (u_in = Ca*sigma/nu0; with the piston this injection rate is exact by construction); M is
# the independent viscosity-ratio axis. These M land both omega inside the (0, 2) band
# (M=30 -> omega in [0.47, 1.80]; M=0.01 -> [0.29, 1.89]); mrt_chi=0.8 keeps the near-edge
# ghost modes damped. Ca=1e-1 is not reachable (u_in=0.05 -> Ma~0.087, compressibility
# blow-up independent of chi); --ca overrides the preset. ``block``
# is the per-regime cadence: Ca=1e-3 breaks through in ~1e5 steps (block 500), Ca=1e-5 in
# ~1e7 (block 2000); the GIF is capped at 300 frames regardless.
REGIMES = {
    "stable": {"ca": 1e-3, "M": 30.0, "block": 500},  # high Ca, viscous invader  (M >> 1)
    "viscous": {"ca": 1e-3, "M": 0.01, "block": 500},  # high Ca, inviscid invader (M << 1)
    "capillary": {"ca": 1e-5, "M": 30.0, "block": 2000},  # low Ca, capillary-dominated (favorable M)
}


# --------------------------------------------------------------------------- knob mapping
def nu_to_omega(nu: float) -> float:
    """BGK/MRT relaxation rate for a target kinematic viscosity: nu = cs^2 (1/omega - 0.5).
    omega is bounded to (0, 2) by definition (omega -> 2 is zero viscosity); the MRT
    chi-spectrum keeps the near-edge splits (M ~ 30 / 0.01) stable."""
    return 1.0 / (nu / CS2 + 0.5)


def omega_pair(M: float, nu0: float = NU0):
    """Split the baseline viscosity nu0 geometrically into (red, blue) for ratio M = nu_red/nu_blue.
    Returns (omega_red, omega_blue_or_None, nu_red, nu_blue). omega_blue is None when M == 1
    (equal viscosity -> single relaxation rate, the solver's omega2=None fast path)."""
    nu_r = nu0 * M**0.5
    nu_b = nu0 / M**0.5
    omega_r = nu_to_omega(nu_r)
    omega_b = None if abs(M - 1.0) < 1e-9 else nu_to_omega(nu_b)
    return omega_r, omega_b, nu_r, nu_b


# --------------------------------------------------------------------------- diagnostics
def spurious_metrics(state, solid_np):
    """(max|u| pore, mean|u| interface band, mean|u| single-phase bulk) — the parasitic
    interface velocity shows as the excess of the meniscus band over the bulk. u_max is
    also the divergence early-warning for the near-edge omega splits (growing past ~0.1
    = LBM stability limit)."""
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


def front_morphology(state, solid, x_lo: int, x_hi: int) -> dict:
    """Host-side front descriptors over the interior pore [x_lo, x_hi) that separate the
    three regimes quantitatively (not just by eye):

      invaded_frac   red-occupied fraction of the interior pore (overall sweep efficiency)
      tip_mean       mean furthest-invaded x per (z, y) column (front position)
      tip_rough      std of those tips = front roughness (stable: small; fingering: large)
      fill_behind_tip  invaded fraction of pore *behind* the mean tip
                       (compactness: stable ~ 1; fingering < 1, fingers leave pore behind)
    """
    rhoR, rhoB = (np.asarray(z) for z in color3d.densities(state))
    solid_np = np.asarray(solid)
    red = (rhoR > rhoB) & ~solid_np
    sub = red[:, :, x_lo:x_hi]  # (nz, ny, Lx)
    pore = ~solid_np[:, :, x_lo:x_hi]
    lx = sub.shape[2]
    invaded_frac = float(sub.sum() / max(pore.sum(), 1))
    has = sub.any(axis=2)
    tip = np.where(sub, np.arange(lx)[None, None, :], -1).max(axis=2)
    tips = tip[has]
    if tips.size == 0:
        return {"invaded_frac": invaded_frac, "tip_mean": 0.0, "tip_rough": 0.0, "fill_behind_tip": 0.0}
    tip_mean = float(tips.mean())
    cut = round(tip_mean) + 1
    behind_pore = pore[:, :, :cut]
    fill = float((sub[:, :, :cut] & behind_pore).sum() / max(behind_pore.sum(), 1))
    return {"invaded_frac": invaded_frac, "tip_mean": tip_mean, "tip_rough": float(tips.std()), "fill_behind_tip": fill}


# --------------------------------------------------------------------------- CLI
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--regime", choices=sorted(REGIMES), required=True, help="Lenormand phase-diagram corner")
    p.add_argument("--structure", type=Path, default=ASSET, help="bool volume (True=solid)")
    p.add_argument("--buffer", type=int, default=7,
                   help="open slabs on each x side; piston face at x=0 leaves buffer-1 reservoir layers")
    p.add_argument("--ca", type=float, default=None, help="override the regime's nu0-referenced capillary number")
    p.add_argument("--m", type=float, default=None, help="override the regime's viscosity ratio M = nu_red/nu_blue")
    p.add_argument("--sigma", type=float, default=SIGMA)
    p.add_argument("--nu0", type=float, default=NU0,
                   help="baseline kinematic viscosity (geometric mean of the M split). Lower it to "
                        "speed up at fixed Ca: u_in=Ca*sigma/nu0 rises. Bounded by the low-viscosity "
                        "phase nu=nu0/sqrt(M) staying stable and u_in staying << 0.1 (Ma).")
    p.add_argument("--theta", type=float, default=135.0,
                   help="red contact angle (deg) on the rock; >90 = drainage entry pressure")
    p.add_argument("--theta-face", type=float, default=0.0,
                   help="red contact angle on the piston-face slab x=1 (0 = purely red-wetting face)")
    p.add_argument("--chi", type=float, default=0.8,
                   help="MRT chi spectrum (Params.mrt_chi): non-shear moments relax at chi*omega — "
                        "required for the M=30/0.01 splits; 0 = off (default MRT spectrum)")
    p.add_argument("--steps", type=int, default=100_000_000, help="safety cap; the run stops at breakthrough first")
    p.add_argument("--after-breakthrough", type=int, default=0,
                   help="extra steps to keep draining past breakthrough toward the saturation plateau "
                        "(0 = stop at breakthrough, the Lenormand-canonical comparison point)")
    p.add_argument("--after-breakthrough-x", type=float, default=0.0,
                   help="extra steps as a fraction of the breakthrough step (1.0 = run to 2x the "
                        "breakthrough step); combined with --after-breakthrough the larger wins. "
                        "Note: on a resume that lands past breakthrough, the factor anchors to the "
                        "re-detected (resumed) step, running somewhat longer than the original anchor")
    p.add_argument("--plateau", type=float, default=0.0,
                   help="with --after-breakthrough: stop early once rock saturation gains less than this "
                        "over the trailing 10 blocks (0 = run the full tail)")
    p.add_argument("--name", type=str, default=None,
                   help="run name for the title and GIF filename (default lenormand_<regime>); "
                        "set it when driving a non-Lenormand structure via --structure")
    p.add_argument("--block", type=int, default=None, help="render/scan block; default is per-regime (see REGIMES)")
    p.add_argument("--rock-alpha", type=float, default=0.04)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--out", type=Path, default=None, help="output dir (default demo/lenormand/output/<regime>)")
    p.add_argument("--plots", action="store_true",
                   help="also write the analysis curves (ledger_curve.svg, pressure_curve.svg, "
                        "spurious_curve.svg); off by default")
    p.add_argument("--resume", type=Path, default=None, help="checkpoint .npz (or dir: newest) to continue")
    p.add_argument("--gpus", type=int, default=1,
                   help="shard the domain along z across the first N devices (bob.multigpu; "
                        "256^3 -> 1.5x/2.65x/4.46x on 2/4/8 H100s, bit-stable)")
    p.add_argument("--field-every", type=int, default=20,
                   help="blocks between HDF5 field dumps (fields.h5 + .xdmf for ParaView); 0 = off")
    p.add_argument("--field-vars", type=str, default="phi,p,umag,u",
                   help="comma-separated fields to log (phi=rhoN, p=pressure, umag=|u|, u=velocity vector)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.plots:
        viz.enable_plots(True)
    preset = REGIMES[args.regime]
    ca = args.ca if args.ca is not None else preset["ca"]
    M = args.m if args.m is not None else preset["M"]
    blk = args.block if args.block is not None else preset["block"]
    chi = None if args.chi == 0 else args.chi
    omega_r, omega_b, nu_r, nu_b = omega_pair(M, args.nu0)
    u_in = ca * args.sigma / args.nu0
    name = args.name or f"lenormand_{args.regime}"
    title = f"{name} (Ca={ca:.1e}, M={M:g}, piston)"
    out = args.out or Path("demo/lenormand/output") / args.regime
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    if args.resume is None:
        for stale in frames_dir.glob("frame_*.png"):
            stale.unlink()

    crop = porous3d.load_structure(args.structure)
    solid = porous3d.with_buffers(crop, buffer=args.buffer)
    solid_np = np.asarray(solid)
    nz, ny, nx = solid_np.shape
    phi = 1.0 - float(crop.mean())
    x_lo, x_hi = args.buffer, nx - args.buffer
    camera = viz3d.flow_camera(solid_np.shape)

    # piston geometry: sealed x=0 face (drain seals it for the dynamics too); the wall
    # normals come from the sealed mask so the face-adjacent slab x=1 carries normals
    # for the wetting BC, where the per-cell theta field imposes theta_face.
    sealed_np = solid_np.copy()
    sealed_np[:, :, 0] = True
    pore = ~sealed_np
    a_open = int(pore[:, :, 1].sum())
    pore_in = pore.copy()
    pore_in[:, :, 0] = False
    pore_in[:, :, -1] = False
    xx = np.arange(nx)[None, None, :]
    rock = pore & (xx >= x_lo) & (xx < x_hi)
    nw = color3d.wall_normals(sealed_np)
    np.savez_compressed(out / "wall_normals.npz", nw=np.asarray(nw),
                        structure=args.structure.name, buffer=args.buffer, piston=True)
    theta_field = jnp.full((nz, ny, nx), float(args.theta)).at[:, :, 1].set(float(args.theta_face))

    if args.resume is not None:
        ckpt_path = stats.latest_checkpoint(args.resume) if args.resume.is_dir() else args.resume
        if ckpt_path is None:
            sys.exit(f"no checkpoint found under {args.resume}")
        ckpt = np.load(ckpt_path)
        state = color3d.State(jnp.asarray(ckpt["fR"]), jnp.asarray(ckpt["fB"]))
        step0 = int(ckpt["step"])
        if args.steps <= step0:
            sys.exit(f"--steps {args.steps} <= checkpoint step {step0}: nothing to run (raise --steps)")
    else:
        # t=0 reservoir: red fills the entire inlet area (x=1..buffer-1; x=0 is the solid
        # piston face) — init only, never re-pinned; the piston sustains it physically.
        state = porous3d.init_drainage(nz, ny, nx, n_red=args.buffer)
        step0 = 0
    fields = ("step", "saturation", "sat_all", "front", "m_red", "ideal",
              "p_in", "p_mid", "p_end", "dp", "u_max", "u_iface", "u_bulk")
    monitor = stats.RunMonitor(out, title=title,
                               rows=stats.carry_rows(out, step0, fields=fields) if step0 else None)

    # Field I/O: append phi/p/|u|/u to one HDF5 + XDMF sidecar (open in ParaView) every
    # --field-every blocks, so the field evolution is inspectable (e.g. is the front pinned /
    # is there flow when saturation looks frozen?). mask_solid -> rock cells are NaN.
    fw = (
        FieldWriter(
            out / "fields.h5",
            solid_np,
            fields=tuple(v for v in args.field_vars.split(",") if v),
            mask_solid=True,
            attrs={"regime": args.regime, "ca": float(ca), "M": float(M), "u_in": float(u_in),
                   "sigma": float(args.sigma), "theta": float(args.theta), "chi": float(args.chi),
                   "inlet": "piston"},
        )
        if args.field_every
        else None
    )

    def dump_fields(step):
        if fw is not None:
            fw.append(step, state)

    # Multi-GPU: shard state + every 3D field the step closes over along z (flow is x, so
    # the slabs cut across neither inlet nor outlet); Params then closes over the sharded
    # theta field. out_shardings + donate_argnums pin the dispatch loop to one compilation
    # (jax 0.6.2 re-lowering crash otherwise) and halve device memory — see bob.multigpu.
    # Default solver (MRT + akai + CSF Eq.34/35 + recolor_emag + wall_grad="fluid") + the
    # two regime knobs (omega/omega2), drainage wetting (per-cell theta: red-wetting piston
    # face), and mrt_chi. Inlet is the leak-free piston (u_in*A red/step exact, inlet
    # pressure floats and builds against the entry barriers); outlet is the Zou-He pressure
    # outlet with the colour pinned to defending blue (outlet_sa_red=0.0) so the
    # self-referential zero-gradient split cannot accumulate a spurious red pool.
    p = color3d.Params(omega=omega_r, omega2=omega_b, sigma=args.sigma, beta=0.7,
                     theta=theta_field, mrt_chi=chi)
    step_fn = None
    if args.gpus > 1:
        m = multigpu.mesh(args.gpus)
        print(f"sharding along z over {args.gpus} devices: {[str(d) for d in m.devices.reshape(-1)]}")
        # default multi-GPU backend: staged thin-halo shard_map (demo/multi_gpu_halo);
        # statics go in unsharded, and the step mask carries the piston's x=0 seal.
        step_solid = np.asarray(solid).copy()
        step_solid[:, :, 0] = True
        step_fn = multigpu.staged_halo_step(m, p, step_solid, nw=nw)
        state, solid, nw, theta_field = multigpu.shard((state, solid, nw, theta_field), m)

    sharding = jax.tree_util.tree_map(lambda x: x.sharding, state)
    runner = jax.jit(lambda s: porous3d.drain(s, p, solid, 0, u_in, blk, nw=nw,
                                              inlet="piston", outlet_sa_red=0.0, step_fn=step_fn),
                     out_shardings=sharding, donate_argnums=0)
    start = time.perf_counter()

    def record(step):
        rhoR, rhoB = color3d.densities(state)
        rR, rho = np.asarray(rhoR), np.asarray(rhoR + rhoB)
        sat_rock = float(rR[rock].sum() / max(rho[rock].sum(), 1e-9))
        m_red = float(rR[pore_in].sum())
        p_in = float(rho[:, :, 1][pore[:, :, 1]].mean()) / 3.0
        p_mid = float(rho[:, :, nx // 2][pore[:, :, nx // 2]].mean()) / 3.0
        p_end = float(rho[:, :, -2][pore[:, :, -2]].mean()) / 3.0
        umax, u_iface, u_bulk = spurious_metrics(state, sealed_np)
        monitor.log(
            step=step,
            saturation=sat_rock,                       # rock-only drainage curve (excludes the buffers)
            sat_all=float(rR[pore_in].sum() / max(rho[pore_in].sum(), 1e-9)),
            front=porous3d.invasion_front(state, solid),
            m_red=m_red,
            ideal=u_in * a_open * step,
            p_in=p_in, p_mid=p_mid, p_end=p_end, dp=p_in - p_end,
            u_max=umax, u_iface=u_iface, u_bulk=u_bulk,
        )
        try:  # PyVista needs offscreen GL; rendering is best-effort
            viz3d.phase_frame(
                frames_dir / f"frame_{step:06d}.png",
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
        frames = sorted(frames_dir.glob("frame_*.png"), key=lambda q: int(q.stem.split("_")[1]))
        if not frames:
            return
        if len(frames) > 300:  # cap the GIF for watchability; every frame stays on disk
            idx = np.unique(np.linspace(0, len(frames) - 1, 300).round().astype(int))
            frames = [frames[i] for i in idx]
        viz.write_gif(frames, out / f"{name}.gif", args.fps)

    def ledger_curve():
        """Red-mass increase vs the ideal colour-conserving injection u_in*A*t — the
        leak-free reference is the two curves lying on top of each other (ratio = 1)."""
        if not viz.plots_enabled():
            return
        rows = monitor.rows
        steps = np.array([r["step"] for r in rows], float)
        gained = np.array([r["m_red"] - rows[0]["m_red"] for r in rows])
        ideal = np.array([r["ideal"] - rows[0]["ideal"] for r in rows])
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(steps, gained, color="tab:red", lw=2, label="actual  ΔM_red downstream")
        ax[0].plot(steps, ideal, "--", color="tab:green", lw=2, label="ideal  u_in·A·t")
        ax[0].set(xlabel="step", ylabel="red mass increase", title="colour ledger")
        ax[0].legend(fontsize=9)
        ratio = gained / np.where(ideal > 0, ideal, np.nan)
        ax[1].plot(steps, ratio, color="tab:orange", lw=2)
        ax[1].axhline(1.0, color="0.3", ls="--", lw=1.2)
        ax[1].set_ylim(0.5, 1.5)
        ax[1].set(xlabel="step", ylabel="actual / ideal", title="ledger ratio (leak-free = 1.0)")
        for a in ax:
            a.grid(True, alpha=0.3)
        fig.suptitle(f"{title}: injected red vs ideal")
        fig.tight_layout()
        fig.savefig(out / "ledger_curve.svg")
        plt.close(fig)

    def pressure_curve():
        if not viz.plots_enabled():
            return
        rows = monitor.rows
        steps = [r["step"] for r in rows]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for key, c in (("p_in", "tab:green"), ("p_mid", "tab:blue"), ("p_end", "tab:purple")):
            ax[0].plot(steps, [r[key] for r in rows], color=c, lw=1.8, label=key)
        ax[0].set(xlabel="step", ylabel="pressure  cs²·ρ", title="pressure stations")
        ax[1].plot(steps, [r["dp"] for r in rows], color="tab:red", lw=2)
        ax[1].set(xlabel="step", ylabel="p_in − p_end", title="entry pressure buildup (piston's free output)")
        for a in ax:
            a.grid(True, alpha=0.3)
        ax[0].legend(fontsize=8)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out / "pressure_curve.svg")
        plt.close(fig)

    def spurious_curve():
        if not viz.plots_enabled():
            return
        rows = monitor.rows
        steps = [r["step"] for r in rows]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(steps, [r["u_max"] for r in rows], "-o", ms=3, label="max|u| (pore)")
        ax.plot(steps, [r["u_iface"] for r in rows], "-s", ms=3, label="mean|u| interface (spurious-laden)")
        ax.plot(steps, [r["u_bulk"] for r in rows], "-^", ms=3, label="mean|u| bulk (single-phase)")
        ax.axhline(u_in, color="cyan", lw=1.5, ls="--", label=f"u_in={u_in:.2e}")
        ax.set(xlabel="step", ylabel="|u| (lattice units)", title=f"{title} — spurious current vs time")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "spurious_curve.svg")
        plt.close(fig)

    def curves():
        ledger_curve()
        pressure_curve()
        spurious_curve()

    step = step0
    sat = record(step)
    curves()
    dump_fields(step)  # initial field state
    broke, sat_at_breakthrough = None, None
    blocks_done = 0
    while step < args.steps:
        state = runner(state)
        step += blk
        blocks_done += 1
        sat = record(step)
        curves()
        if not np.isfinite(sat):
            print(f"lenormand[{args.regime}]: NON-FINITE saturation at step {step} — diverged "
                  f"(Ca={ca:.1e}, M={M:g}, chi={chi}); stopping early. Watch u_max in "
                  f"metrics.csv for the growth; back off Ca or M.")
            broke = None
            break
        if args.field_every and blocks_done % args.field_every == 0:
            dump_fields(step)
        if broke is None and porous3d.breakthrough(state, solid, x=nx - args.buffer):
            broke, sat_at_breakthrough = step, sat  # invading red reached the outlet
            gif()
            extra_after = max(args.after_breakthrough, int(args.after_breakthrough_x * step))
            if extra_after <= 0:
                break  # stop here: the Lenormand-canonical comparison point
            print(f"breakthrough at step {step} (sat {sat:.3f}); draining on for up to "
                  f"{extra_after} more steps toward the saturation plateau")
            monitor.checkpoint(step, sat, fR=np.asarray(state.fR), fB=np.asarray(state.fB))
        elif broke is not None:
            if step - broke >= extra_after:
                break
            tail = [r["saturation"] for r in monitor.rows[-11:]]
            if args.plateau > 0 and len(tail) == 11 and (tail[-1] - tail[0]) < args.plateau:
                print(f"saturation plateau at step {step}: +{tail[-1] - tail[0]:.4f} over the last "
                      f"10 blocks < --plateau {args.plateau}")
                break
        if blocks_done % 10 == 0:
            gif()
            monitor.checkpoint(step, sat, fR=np.asarray(state.fR), fB=np.asarray(state.fB))
    elapsed = time.perf_counter() - start
    monitor.checkpoint(step, sat, fR=np.asarray(state.fR), fB=np.asarray(state.fB))
    dump_fields(step)  # final field state
    if fw is not None:
        fw.close()
    gif()
    curves()

    morph = front_morphology(state, solid, x_lo, x_hi)
    connected = porous3d.connected_to_inlet(state, solid)
    rows = monitor.rows
    gained = rows[-1]["m_red"] - rows[0]["m_red"]
    ideal = rows[-1]["ideal"] - rows[0]["ideal"]
    # ledger ratio at breakthrough (pre-outflow): the leak-free check is 1.000 there;
    # after breakthrough red exits through the outlet, so the whole-run ratio drops
    # below 1 by the outflow — physics, not a leak.
    ratio_T = None
    if broke is not None:
        pre = [r for r in rows if r["step"] <= broke]
        if len(pre) > 1 and (pre[-1]["ideal"] - pre[0]["ideal"]) > 0:
            ratio_T = (pre[-1]["m_red"] - pre[0]["m_red"]) / (pre[-1]["ideal"] - pre[0]["ideal"])
    (out / "report.md").write_text(
        f"# {title}\n\n"
        f"- Regime **{args.regime}** | Ca (nu0-ref) {ca:.2e} | M = nu_red/nu_blue = {M:g} | mrt_chi {chi}\n"
        f"- omega_red {omega_r:.4f} (nu_red {nu_r:.4f}) | omega_blue "
        f"{omega_b if omega_b is not None else omega_r:.4f} (nu_blue {nu_b:.4f}) | u_in {u_in:.3e} | sigma {args.sigma}"
        f" | precision {'fp32' if FP32 else 'fp64'}\n"
        f"- Structure {args.structure.name} | porosity {phi:.3f} | domain {nz}x{ny}x{nx} "
        f"(piston face x=0, reservoir x=1..{args.buffer - 1}, rock x={x_lo}..{x_hi - 1})\n"
        f"- Piston inlet (u_in*A red/step exact, inlet pressure floats) + blue-pinned pressure outlet "
        f"(outlet_sa_red=0.0) | theta {args.theta:.0f} deg rock / {args.theta_face:.0f} deg piston face\n"
        f"- Steps {rows[-1]['step']} | runtime {elapsed:.1f}s | breakthrough at "
        f"{broke if broke is not None else 'not reached'} (this segment)\n\n"
        f"## Result\n\n"
        f"- Rock saturation {rows[0]['saturation']:.3f} -> {rows[-1]['saturation']:.3f}"
        f"{f' | at breakthrough {sat_at_breakthrough:.3f}' if sat_at_breakthrough is not None else ''}\n"
        f"- Colour ledger: downstream red gained {gained:.2f} vs ideal u_in*A*t {ideal:.2f} "
        f"-> **{gained / max(ideal, 1e-12):.3f}x** whole-run"
        + (f"; **{ratio_T:.3f}x at breakthrough** (the leak-free check)\n" if ratio_T is not None
           else " (leak-free reference = 1.0; no breakthrough this segment)\n")
        + f"- Entry pressure p_in - p_end: {rows[0]['dp']:+.5f} -> {rows[-1]['dp']:+.5f}\n"
        + f"- Inlet-connected red fraction: {connected:.4f}\n"
        + f"- Front morphology: invaded_frac {morph['invaded_frac']:.3f} | tip_mean {morph['tip_mean']:.1f} | "
        + f"tip_rough {morph['tip_rough']:.2f} | fill_behind_tip {morph['fill_behind_tip']:.3f}\n",
        encoding="utf-8",
    )
    print(
        f"lenormand[{args.regime}](piston): Ca {ca:.1e}, M {M:g}, rock sat {rows[0]['saturation']:.3f}->"
        f"{rows[-1]['saturation']:.3f}, ledger {gained / max(ideal, 1e-12):.3f}x, dp {rows[-1]['dp']:+.5f}, "
        f"breakthrough {broke}, tip_rough {morph['tip_rough']:.2f}, fill {morph['fill_behind_tip']:.3f}"
    )
    print(f"  output: {out}")


if __name__ == "__main__":
    main()
