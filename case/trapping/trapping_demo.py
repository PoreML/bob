"""Two-stage piston waterflood to residual oil: equilibrate, then flood until still.

The trapping driver of the dataset campaign: one run folder, two stages, each
with its own frames/GIF/h5/metrics (artifacts are prefixed "stab" / "flood").

  Stage 1 "stab" — equilibrium transition (log every --stab-block = 1000 steps;
    frames/GIF/metrics only. Relaxation fields are not written by default;
    --stab-h5 turns them on).
    The PMI morphological initial condition is a geometric guess, not a
    mechanical state: every cell starts at rho=1, which is uniform but not in
    equilibrium. The stage runs with the piston off (Ca = --stab-ca ~ 0) and
    the open zero-gradient outlet until the system stops moving. What relaxes is
    the pressure field (Young-Laplace: oil above ambient, water below) and,
    because the outlet line is water, the sample also imbibes spontaneously to
    its capillary equilibrium with that line. Converged = |dS_w| per block below
    --stab-tol for --stab-hold consecutive blocks. Reference (64^3 sphere pack):
    S_w 0.4264 -> 0.4399 with increments decaying +1.0e-2, +6e-4, +1.5e-4,
    +1e-5, dp settling at -0.00087.

  Stage 2 "flood" — constant-rate drive (log every --block = 5000 steps,
    fields .h5 included).
    The piston pumps the blue wetting phase at Ca = 1e-5 (exactly u_in*A per
    step, inlet pressure floats) until the oil stops moving or --pv-cap = 1.2
    rock pore volumes are injected, whichever comes first.

    "Stops moving" is a set comparison on the oil body itself, not a velocity
    and not a saturation: the red-majority voxel mask over the rock (buffers
    excluded) is compared with the mask --mob-window = 20000 steps earlier, and
    the run ends ("red_immobile") when fewer than --mob-tol of its voxels
    changed in each of --mob-hold = 3 consecutive, non-overlapping windows —
    60,000 steps of sustained stillness, so one quiet window cannot end a run.
    Saturation is the wrong signal — oil can be produced at constant S_o, and
    can sit still while S_o drifts — and mean|u| is dominated by the model's
    spurious interface currents (~300x u_in). Reference levels (64^3 sphere
    pack): with the piston off the mask still changes by a flat 0.20-0.30% per
    window from diffuse-interface jitter, while an actively displacing flood
    sits at ~3.0%; the 1% default separates the two with ~4x / ~3x margin. The
    jitter floor must be measured from a converged stage-1 state; a
    half-relaxed PMI arrangement is still evolving and inflates it.
    The outlet is the same as in stage 1, and that self-consistency matters:
    stage 1 has already spent the spontaneous imbibition, so the outlet only
    produces here (`produced` stays positive) and dS_w is linear in time, the
    signature of a forced displacement. A flood started directly from the PMI
    state instead reports ~10x more saturation change, nearly all of it
    initial-condition transient + imbibition.

Domain: [piston face x=0 | water buffer | rock | water buffer] along x, sealed
z/y walls, no porous plate. The outlet buffer is kept water-filled by a per-step
red->blue scrub (the produced oil, logged as `oil_out`); the outlet BC itself
(bc.outlet_zero_gradient) never touches colour, it only anchors pressure.

Campaign overrides: the driver's own defaults are beta 0.7 and --mob-tol 0.01.
The dataset campaign runs with `--beta 0.99 --mob-tol 0.003`
(case/run_array.slurm); every run_meta_flood.json there records "moved < 0.3%"
in its finish_detail.

Physics: the case/drainage stack — sigma 0.05, MRT chi 0.8, fp32, MRT + akai +
CSF; nu0 0.04 split geometrically for M = nu_red/nu_blue; --theta is the red
(oil) contact angle, so 120-150 = water-wet rock; the piston face slab x=1 is
theta 180 (pure water).

Colour ledger (blue): piston = dM_blue + produced - oil_out, where `produced` is
metered at the outlet plane every step and `oil_out` is the red->blue scrub.
Reads ~0.98 in fp32 and 1.000 under BOB_FP64=1 — the residual is fp32 mass
drift, not a boundary leak.

Usage (from the repository root):
  .venv/bin/python case/trapping/trapping_demo.py \
      --structure <rock.npy> --out case/trapping/runs/<name> \
      --so 0.6 --theta 140 --m 1
  # stage 2 only, reusing an existing equilibrated_state.npz:
  ... --stage flood
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import jax

# fp32 is the solver default (bandwidth-bound, ~2x steps/s). Precision is decided
# before importing bob; PMI is pure numpy/skimage and does not touch the JAX config.
FP32 = os.environ.get("BOB_FP64") != "1"
jax.config.update("jax_enable_x64", not FP32)

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))  # PMI and pmi_search sit beside this driver
import PMI  # vendored (pure numpy/skimage)
from pmi_search import closest_kernel

from bob import bc, color3d, d3q19, multigpu
from bob.utils import stats, viz, viz3d
from bob.utils.file import FieldWriter, RunMeta

# ---- the standardized setting (the case/drainage stack) --------------------------
SIGMA = 0.05
NU0 = 0.04  # baseline kinematic viscosity, split geometrically for M
CA = 1e-5  # flood capillary number -> u_in = CA * SIGMA / NU0 = 1.25e-5
STAB_CA = 1e-12  # stage 1: piston effectively off
BETA = 0.7
CHI = 0.8
THETA_FACE = 180.0  # piston-face slab x=1: pure water (the blue piston)
PMI_THETA = 35.0  # morphology (water) contact angle for the PMI opening
STAB_BLOCK = 1000  # stage 1 log interval: metrics + frame + fields.h5
BLOCK = 5000  # stage 2 log interval: metrics + frame + fields.h5
PV_CAP = 1.2  # stage 2 hard cap: rock pore volumes injected
MOB_WINDOW = 20000  # stage 2 ends when the oil has not moved over this many steps
# Fraction of the oil body's voxels that may change over MOB_WINDOW and still count as
# immobile. Reference levels on a 64^3 sphere pack, with the floor measured from a
# converged stage-1 state (|dS_w| < 5e-6 held 3 blocks) — a half-relaxed PMI
# arrangement gives a floor that is still decaying, i.e. an over-estimate. With the
# piston off the mask changes by a flat 0.20-0.30% per 20k window (no systematic decay):
# diffuse-interface jitter, not motion. An actively displacing flood at Ca 1e-5 sits
# steady at ~3.0%. 1% is ~4x above the jitter floor and ~3x below the displacing level;
# a more fragmented (higher surface-to-volume) residual oil jitters more. This is the
# driver default; the dataset campaign passes --mob-tol 0.003.
MOB_TOL = 0.01
MOB_HOLD = 3  # consecutive non-overlapping windows that must pass before the run ends
FIELD_VARS = "phi,rho,p,umag,u"
EQ_STATE = "equilibrated_state.npz"  # stage 1 -> stage 2 handoff (protected from ckpt pruning)


def omega_pair(M, nu0=NU0):
    """nu0 split geometrically for M = nu_red/nu_blue (case/drainage convention)."""
    nu_r, nu_b = nu0 * M**0.5, nu0 / M**0.5
    return 1.0 / (3.0 * nu_r + 0.5), 1.0 / (3.0 * nu_b + 0.5), nu_r, nu_b


# ---------------------------------------------------------------- geometry
def build_domain(rock, buf_in=10, buf_out=6):
    """[piston face x=0 + inlet buffer | rock | outlet buffer] along x, solid z/y
    boundary planes (core-holder convention), no plate. x=0 is the piston face,
    sealed inside the step mask so the half-way bounce-back is the piston reflection."""
    rock = np.asarray(rock, bool)
    nz, ny, nxr = rock.shape
    nx = buf_in + nxr + buf_out
    solid = np.zeros((nz, ny, nx), bool)
    solid[:, :, buf_in : buf_in + nxr] = rock
    solid[0, :, :] = solid[-1, :, :] = True
    solid[:, 0, :] = solid[:, -1, :] = True
    return solid, {"inbuf": (0, buf_in), "rock": (buf_in, buf_in + nxr), "outbuf": (buf_in + nxr, nx)}


def init_state(solid, reg, label):
    """u=0 equilibrium IC: both buffers pure blue (water — the injected phase and
    the water-filled effluent line), rock from the PMI label (red where label==1);
    solids blue (inert under bounce-back, masked out of every metric)."""
    nz, ny, nx = solid.shape
    x0, x1 = reg["rock"]
    rhoR = np.zeros((nz, ny, nx))
    rhoR[:, :, x0:x1] = (label == 1).astype(float)
    rhoR[solid] = 0.0
    W = d3q19.W[:, None, None, None]
    return color3d.State(W * jnp.asarray(rhoR)[None], W * jnp.asarray(1.0 - rhoR)[None])


# ---------------------------------------------------------------- PMI init
def pmi_oil_label(rock, so_target, num_workers=8, tol=0.02):
    """Morphological oil placement at oil saturation ``so_target`` (PMI targets the
    wetting saturation, so target_sw = 1 - so). Returns (label, achieved_so).

    PMI's own kernel search is evaluated on a ~10-slice test band, which is biased
    against the full domain, so the search here runs on the full domain from the
    band-best kernel and keeps the closest (pmi_search.closest_kernel). Integer
    kernels quantize the reachable saturations — use the recorded so_init, not the
    target, when analysing runs."""
    target_sw = 1.0 - float(so_target)
    domain = np.ascontiguousarray(np.asarray(rock).astype(np.uint8))  # 1=solid, 0=pore
    nz = domain.shape[0]
    lo, hi = max(nz // 2 - 5, 0), min(nz // 2 + 5, nz)
    best_k, _ = PMI.find_best_kernel_size(domain[lo:hi], target_sw, PMI_THETA, num_workers)

    def eval_k(k):
        sat_k, comb_k = PMI.compute_saturation_parallel(domain, k, PMI_THETA, num_workers)
        print(f"  full-domain kernel {k}: Sw {sat_k:.4f} (S_o {1 - sat_k:.4f})", flush=True)
        return sat_k, comb_k

    k_full, sat_full, comb = closest_kernel(eval_k, best_k, target_sw, tol)
    label = np.asarray(comb).astype(np.uint8)
    assert label.shape == domain.shape
    print(f"  full-domain kernel {k_full} kept", flush=True)
    if abs(sat_full - target_sw) > tol:
        print(f"  WARNING: PMI target Sw={target_sw:g} missed by >{tol}; closest {sat_full:.4f}")
    if not (label == 1).any():
        # a trapping run with nothing to trap still "equilibrates" and ends red_immobile,
        # so it would be counted as a finished run — refuse it here instead
        raise SystemExit(f"PMI placed no oil (closest kernel {k_full}: Sw {sat_full:.4f}) — refusing to flood pure water")
    return label, 1.0 - float(sat_full)


# ---------------------------------------------------------------- diagnostics
def spurious_metrics(state, solid_np):
    """(max|u| pore, mean|u| interface band, mean|u| single-phase bulk)."""
    pore = ~solid_np
    u = color3d.velocity(state)
    umag = np.asarray(jnp.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2))
    rhoR, rhoB = color3d.densities(state)
    rhoN = np.asarray(color3d.color_field(rhoR, rhoB))
    iface = pore & (np.abs(rhoN) < 0.9)
    bulk = pore & (np.abs(rhoN) >= 0.9)
    return (
        float(umag[pore].max()) if pore.any() else 0.0,
        float(umag[iface].mean()) if iface.any() else 0.0,
        float(umag[bulk].mean()) if bulk.any() else 0.0,
    )


# ---------------------------------------------------------------- CLI
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--structure", type=Path, required=True, help="rock bool volume (True=solid)")
    p.add_argument("--out", type=Path, required=True, help="run directory (both stages land here)")
    p.add_argument("--name", type=str, default=None, help="run name (default from the structure stem)")
    p.add_argument("--stage", choices=("both", "stab", "flood"), default="both", help="which stage(s) to run")
    p.add_argument("--so", type=float, default=0.6, help="target initial oil saturation for the PMI init")
    p.add_argument("--theta", type=float, default=140.0, help="rock contact angle (red/oil phase, deg; 120-150 = water-wet)")
    p.add_argument("--m", type=float, default=1.0, help="viscosity ratio M = nu_red/nu_blue")
    p.add_argument("--ca", type=float, default=CA, help="stage-2 capillary number (u_in = ca*sigma/nu0)")
    p.add_argument("--sigma", type=float, default=SIGMA)
    p.add_argument("--beta", type=float, default=BETA, help="recoloring sharpness (dataset campaign: 0.99)")
    p.add_argument("--nu0", type=float, default=NU0)
    p.add_argument("--chi", type=float, default=CHI)
    p.add_argument("--pv-cap", type=float, default=PV_CAP, help="stage 2 hard cap: rock PV injected")
    p.add_argument(
        "--mob-window",
        type=int,
        default=MOB_WINDOW,
        help="stage 2 ends when the oil has not moved over this many steps (default 20000)",
    )
    p.add_argument(
        "--mob-tol",
        type=float,
        default=MOB_TOL,
        help="fraction of its voxels the oil body may change over --mob-window and still count as "
        "immobile (dataset campaign: 0.003). Motion, not saturation: oil can be produced at constant "
        "S_o, and can sit still while S_o drifts",
    )
    p.add_argument(
        "--mob-hold",
        type=int,
        default=MOB_HOLD,
        help="consecutive non-overlapping --mob-window blocks that must all read immobile before the "
        "run ends (default 3 = 60,000 steps of sustained stillness, so one quiet window cannot end it)",
    )
    p.add_argument("--block", type=int, default=BLOCK, help="stage-2 log interval (metrics + frame + fields.h5)")
    p.add_argument("--stab-block", type=int, default=STAB_BLOCK, help="stage-1 log interval")
    p.add_argument("--stab-ca", type=float, default=STAB_CA, help="stage-1 capillary number (~0 = piston off)")
    p.add_argument("--stab-tol", type=float, default=5e-6, help="stage 1 converged when |dS_w| per block is below this")
    p.add_argument("--stab-hold", type=int, default=3, help="consecutive converged blocks required")
    p.add_argument("--stab-steps", type=int, default=200_000, help="stage-1 safety cap")
    p.add_argument("--steps", type=int, default=100_000_000, help="stage-2 safety cap")
    p.add_argument("--buf-in", type=int, default=10)
    p.add_argument("--buf-out", type=int, default=6)
    p.add_argument("--min-blob", type=int, default=8, help="minimum trapped-oil component size (cells)")
    p.add_argument("--pmi-workers", type=int, default=8)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--plots", action="store_true",
                   help="also write the analysis curves (<stage>_curves.svg, sw_<stage>.svg); off by default")
    p.add_argument("--rock-alpha", type=float, default=0.04)
    p.add_argument("--field-vars", type=str, default=FIELD_VARS)
    p.add_argument(
        "--stab-h5",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="also write fields.h5 during stage 1. Off by default: the equilibration is "
        "relaxation, not displacement, so its fields are not worth the volume (~56 MB per "
        "snapshot at 256^3 = 3-11 GB for the stage). Frames/GIF/metrics are unaffected",
    )
    p.add_argument("--resume", action="store_true", help="stage 2: resume from the newest ckpt_step*.npz in --out")
    p.add_argument("--gpus", type=int, default=1, help="shard along z over N devices (bob.multigpu; nz must divide)")
    return p.parse_args()


# ---------------------------------------------------------------- one stage
def run_stage(tag, state, step0, so_init, args, geom, params, meta_extra):
    """Run one stage to its own termination rule; returns (state, step, rows).

    ``tag`` is "stab" or "flood" and prefixes every artifact so both stages can
    share one run folder. Stage 1 stops on |dS_w| convergence, stage 2 on the PV
    cap. The BCs are identical in both — that self-consistency is what makes the
    flood a clean forced displacement (see the module docstring)."""
    out = args.out
    is_stab = tag == "stab"
    blk = args.stab_block if is_stab else args.block
    u_in = (args.stab_ca if is_stab else args.ca) * args.sigma / args.nu0
    solid_np, reg, nw, camera = geom["solid_np"], geom["reg"], geom["nw"], geom["camera"]
    sealed_np, pore, a_open = geom["sealed_np"], geom["pore"], geom["a_open"]
    rock_m, outbuf_m, inbuf_p, led_m = geom["rock_m"], geom["outbuf_m"], geom["inbuf_p"], geom["led_m"]
    xx3 = geom["xx3"]
    x0, x1, x_ob, nx = geom["x0"], geom["x1"], geom["x_ob"], geom["nx"]
    pv_rock = geom["pv_rock"]
    solid, step_fn = geom["solid"], geom["step_fn"]

    frames_dir = out / f"frames_{tag}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    if step0 == 0:
        for stale in frames_dir.glob("frame_*.png"):
            stale.unlink()

    steps_pv = None if is_stab else int(np.ceil(args.pv_cap * pv_rock / (u_in * a_open) / blk)) * blk
    cap = args.stab_steps if is_stab else min(steps_pv, args.steps)
    title = (
        f"{args.name} {tag} ({'equilibrating, piston off' if is_stab else f'flood Ca={args.ca:.0e} to {args.pv_cap:g} PV'}"
        f", theta {args.theta:g}, M {args.m:g})"
    )
    rule = f"converge |dSw| < {args.stab_tol:g} x{args.stab_hold}" if is_stab else f"{args.pv_cap:g} PV = {steps_pv} steps"
    print(f"\n=== stage {tag}: block {blk}, u_in {u_in:.3e}, {rule}, cap {cap} ===", flush=True)

    fields = ("step", "sw", "so_rock", "front", "m_red", "m_blue", "ideal", "p_in", "p_mid", "p_end", "dp",
              "trapped_blobs", "trapped_cells", "trapped_mass", "u_max", "u_iface", "u_bulk",
              "water_out", "oil_out", "produced", "u_red", "red_com", "red_cells", "red_moved")  # fmt: skip
    meta = RunMeta(
        out / f"run_meta_{tag}.json",
        # scalar-theta twin: RunMeta serialises Params to JSON and the solver's theta is a
        # per-cell field here (rock angle + theta 180 on the piston face)
        params=color3d.Params(omega=params.omega, omega2=params.omega2, sigma=args.sigma, beta=args.beta,
                              theta=float(args.theta), mrt_chi=args.chi),
        solid=solid_np,
        geometry_source=str(args.structure),
        resume_step=step0,
        target_steps=cap,
        extra={**meta_extra, "stage": tag, "block": blk, "u_in": float(u_in),
               "protocol": ("equilibrium transition (piston off, open outlet)" if is_stab
                            else f"constant-rate blue piston to {args.pv_cap:g} PV"),
               **({} if is_stab else {"steps_pv": steps_pv})},
    )
    rows = stats.carry_rows(out, step0, fields=fields) if step0 else None
    if rows:
        rows = [{**dict.fromkeys(fields, 0.0), **r} for r in rows]
    monitor = stats.RunMonitor(out, title=title, curve_field="sw", rows=rows, meta=meta)
    write_h5 = (not is_stab) or args.stab_h5
    fw = FieldWriter(
        out / f"{tag}_{args.name}.h5",
        solid_np,
        fields=tuple(v for v in args.field_vars.split(",") if v),
        mask_solid=True,
        attrs={"stage": tag, "so_target": float(args.so), "so_init": float(so_init), "theta": float(args.theta),
               "M": float(args.m), "ca": float(args.stab_ca if is_stab else args.ca), "u_in": float(u_in),
               "sigma": float(args.sigma), "nu0": float(args.nu0), "chi": float(args.chi)},
        meta=meta,
    ) if write_h5 else None

    # ---- the block runner: piston -> zero-gradient outlet -> per-step oil scrub
    step_solid_j = jnp.asarray(solid).at[:, :, 0].set(True)
    red_zone = jnp.asarray(np.ascontiguousarray(outbuf_m[:, :, x_ob:]))[None]

    def _block(s0, produced0, oil0):
        def body(carry, _):
            s, prod, oil = carry
            s = color3d.step(s, params, solid=step_solid_j, nw=nw) if step_fn is None else step_fn(s)
            s = bc.piston_inlet(s, u_in, solid=step_solid_j, sa_red=0.0)
            before = s.fB[:, :, :, -1].sum()
            s = bc.outlet_zero_gradient(s)  # open outflow, pressure anchor, colour untouched
            prod = prod + (before - s.fB[:, :, :, -1].sum())  # blue leaving at the outlet plane
            red = jnp.where(red_zone, s.fR[:, :, :, x_ob:], 0.0)  # produced oil: keep the line water-filled
            s = color3d.State(s.fR.at[:, :, :, x_ob:].add(-red), s.fB.at[:, :, :, x_ob:].add(red))
            return (s, prod, oil + red.sum()), None

        (s, prod, oil), _ = jax.lax.scan(body, (s0, produced0, oil0), None, length=blk)
        return s, prod, oil

    sharding = jax.tree_util.tree_map(lambda x: x.sharding, state)
    runner = jax.jit(_block, out_shardings=(sharding, None, None), donate_argnums=0)

    produced_tot = (monitor.rows[-1].get("produced", 0.0) if monitor.rows else 0.0) or 0.0
    oil_out_tot = (monitor.rows[-1].get("oil_out", 0.0) if monitor.rows else 0.0) or 0.0

    prev_mask: list = []  # last red mask, for the per-block "did it move" diagnostic

    def record(step):
        rhoR, rhoB = color3d.densities(state)
        rR, rB = np.asarray(rhoR), np.asarray(rhoB)
        rho = rR + rB
        sw = float(rB[rock_m].sum() / max(rho[rock_m].sum(), 1e-9))
        frac_r = np.divide(rR, rho, out=np.zeros_like(rho), where=rho > 1e-9)
        red = pore & (frac_r > 0.5)
        lab, n = ndimage.label(red)
        blobs = cells = 0
        mass = 0.0
        if n:
            connected = np.unique(lab[:, :, x_ob:])
            sizes = np.bincount(lab.ravel(), minlength=n + 1)
            keep = np.ones(n + 1, bool)
            keep[0] = False
            keep[connected] = False
            keep &= sizes >= args.min_blob
            blobs, cells, mass = int(keep.sum()), int(sizes[keep].sum()), float(rR[keep[lab]].sum())
        labb, _ = ndimage.label(pore & ~red)
        conn = np.unique(labb[:, :, 1 : reg["inbuf"][1]])
        conn = conn[conn > 0]
        bconn = np.isin(labb, conn) if conn.size else np.zeros_like(red)
        cols = bconn.any(axis=(0, 1))
        umax, u_iface, u_bulk = spurious_metrics(state, sealed_np)
        # Is the oil moving? Mass-weighted mean velocity of the red phase in the rock,
        # and its centre of mass. Both are signed sums, so the model's spurious
        # interface currents (closed loops, and ~300x u_in here) largely cancel —
        # unlike mean|u|, which is swamped by them. Saturation is deliberately not
        # used: oil can be produced at constant S_o, and can sit still while S_o drifts.
        uu = np.asarray(color3d.velocity(state))
        wr = rR[rock_m]
        den = max(float(wr.sum()), 1e-12)
        u_red = float(np.linalg.norm([float((uu[k][rock_m] * wr).sum()) for k in range(3)]) / den)
        red_com = float((xx3[rock_m] * wr).sum() / den)
        red_rock = red & rock_m  # the oil body itself, buffers excluded
        x_mid = (x0 + x1) // 2
        p_in = float(rho[:, :, 1 : reg["inbuf"][1]][inbuf_p].mean()) / 3.0
        p_end = float(rho[:, :, -2][pore[:, :, -2]].mean()) / 3.0
        monitor.log(
            step=step, sw=sw, so_rock=1.0 - sw,
            front=int(np.where(cols)[0].max()) if cols.any() else -1,
            m_red=float(rR[led_m].sum()), m_blue=float(rB[led_m].sum()), ideal=u_in * a_open * step,
            p_in=p_in, p_mid=float(rho[:, :, x_mid][pore[:, :, x_mid]].mean()) / 3.0, p_end=p_end, dp=p_in - p_end,
            trapped_blobs=blobs, trapped_cells=cells, trapped_mass=mass,
            u_max=umax, u_iface=u_iface, u_bulk=u_bulk,
            water_out=float(rB[outbuf_m].sum()), oil_out=oil_out_tot, produced=produced_tot,
            u_red=u_red, red_com=red_com, red_cells=int(red_rock.sum()),
            red_moved=float((red_rock ^ prev_mask[0]).sum() / max(int(red_rock.sum()), 1)) if prev_mask else 0.0,
        )  # fmt: skip
        try:  # PyVista needs offscreen GL; rendering is best-effort
            viz3d.phase_frame(
                frames_dir / f"frame_{step:08d}.png", np.asarray(color3d.color_field(rhoR, rhoB)),
                sealed_np, rock_opacity=args.rock_alpha, camera=camera, zoom=1.0,
            )  # fmt: skip
        except Exception as e:
            if step == step0:
                print(f"  [3D render unavailable: {type(e).__name__}: {e}; metrics/field IO continues]")
        prev_mask[:] = [red_rock]
        return sw, red_rock

    def gif():
        frames = sorted(frames_dir.glob("frame_*.png"))
        if not frames:
            return
        if len(frames) > 300:  # watchability cap; every frame stays on disk
            idx = np.unique(np.linspace(0, len(frames) - 1, 300).round().astype(int))
            frames = [frames[i] for i in idx]
        viz.write_gif(frames, out / f"{tag}_{args.name}.gif", args.fps)

    def curves():
        if not viz.plots_enabled():
            return
        rs = monitor.rows
        st = [r["step"] for r in rs]
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].plot(st, [r["sw"] for r in rs], color="tab:blue", lw=2)
        ax[0].set(xlabel="step", ylabel="rock S_w", title="saturation")
        ax[1].plot(st, [r["dp"] for r in rs], color="tab:red", lw=2)
        ax[1].axhline(0.0, color="0.6", lw=1, ls=":")
        ax[1].set(xlabel="step", ylabel="p_in − p_end", title="drive pressure")
        if not is_stab:
            gained = np.array([(r["m_blue"] + r["produced"] - r["oil_out"])
                               - (rs[0]["m_blue"] + rs[0]["produced"] - rs[0]["oil_out"]) for r in rs])  # fmt: skip
            ideal = np.array([r["ideal"] - rs[0]["ideal"] for r in rs])
            ax[2].plot(st, gained / np.where(ideal > 0, ideal, np.nan), color="tab:orange", lw=2)
            ax[2].axhline(1.0, color="0.3", ls="--", lw=1.2)
            ax[2].set(xlabel="step", ylabel="actual / ideal", ylim=(0.5, 1.5), title="colour ledger (leak-free = 1)")
        else:
            ax[2].plot(st, [r["u_iface"] for r in rs], color="tab:green", lw=2)
            ax[2].set(xlabel="step", ylabel="mean |u| interface", title="relaxation")
        for a in ax:
            a.grid(True, alpha=0.3)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out / f"{tag}_curves.svg")
        plt.close(fig)

    # ---- main loop ------------------------------------------------------------
    step = step0
    sw, mask = record(step)
    curves()
    if fw is not None:
        fw.append(step, state)
    sw_prev, held, blocks = sw, 0, 0
    # (step, red mask) history: stage 2 ends when the oil body is the same set of
    # voxels it was --mob-window steps ago. A boolean set comparison, not a velocity
    # and not a saturation: immune to the model's spurious interface currents, and it
    # cannot be fooled by oil being produced at constant S_o.
    hist = [(step, mask)]
    mob_held, next_check = 0, step + args.mob_window
    finish_type, finish_detail = "cap", f"hit the stage cap ({cap} steps)"
    t0 = time.perf_counter()
    while step < cap:
        state, prod, oil = runner(state, jnp.asarray(0.0), jnp.asarray(0.0))
        produced_tot += float(prod)
        oil_out_tot += float(oil)
        step += blk
        blocks += 1
        sw, mask = record(step)
        hist.append((step, mask))
        hist[:] = [(t, m) for t, m in hist if t >= step - 2 * args.mob_window]  # bound the memory
        curves()
        if fw is not None:
            fw.append(step, state)
        if not np.isfinite(sw):
            finish_type, finish_detail = "diverged", f"non-finite saturation at step {step}"
            print(f"{args.name}: non-finite saturation at step {step} — stopping.")
            break
        if is_stab:
            d = abs(sw - sw_prev)
            held = held + 1 if d < args.stab_tol else 0
            print(f"  [{tag}] step {step}: S_w {sw:.5f} (|dS_w| {d:.2e}, held {held}/{args.stab_hold}) "
                  f"dp {monitor.rows[-1]['dp']:+.6f}", flush=True)  # fmt: skip
            sw_prev = sw
            if held >= args.stab_hold:
                finish_type = "equilibrated"
                finish_detail = f"|dS_w| < {args.stab_tol:g} for {args.stab_hold} blocks at step {step}"
                print(f"{args.name}: equilibrated — {finish_detail}")
                break
        else:
            pv = u_in * a_open * step / pv_rock
            print(f"  [{tag}] step {step}: {pv:.3f} PV, S_w {sw:.5f}, produced {produced_tot:+.1f}, "
                  f"dp {monitor.rows[-1]['dp']:+.6f}", flush=True)
            older = [(t, m) for t, m in hist if t <= step - args.mob_window]
            if older:
                t0m, m0 = older[-1]
                moved = float((mask ^ m0).sum() / max(int(mask.sum()), 1))
                # rolling value every block for monitoring; only the scheduled,
                # non-overlapping checks below decide anything
                print(f"      oil moved {moved * 100:.3f}% of its voxels since step {t0m} "
                      f"(immobile below {args.mob_tol * 100:g}%, held {mob_held}/{args.mob_hold})", flush=True)
                if step >= next_check:
                    mob_held = mob_held + 1 if moved < args.mob_tol else 0
                    next_check = step + args.mob_window
                    if mob_held >= args.mob_hold:
                        finish_type = "red_immobile"
                        finish_detail = (
                            f"the oil body moved < {args.mob_tol * 100:g}% of its voxels in each of "
                            f"{args.mob_hold} consecutive {args.mob_window}-step windows "
                            f"({args.mob_hold * args.mob_window} steps of sustained stillness), "
                            f"last {moved * 100:.3f}% at step {step}"
                        )
                        print(f"{args.name}: oil immobile — {finish_detail}")
                        break
            if step >= steps_pv:
                finish_type = "pv_cap"
                finish_detail = f"{args.pv_cap:g} rock pore volumes injected at step {step}"
                print(f"{args.name}: PV cap — {finish_detail}")
                break
        if blocks % 10 == 0:
            gif()
            monitor.checkpoint(step, sw, fR=np.asarray(state.fR), fB=np.asarray(state.fB),
                               so_init=np.float64(so_init))  # fmt: skip
    elapsed = time.perf_counter() - t0

    monitor.checkpoint(step, sw, fR=np.asarray(state.fR), fB=np.asarray(state.fB), so_init=np.float64(so_init))
    if fw is not None:
        fw.close()
    gif()
    curves()
    meta.finish(step, finish_type=finish_type, finish_detail=finish_detail)
    rs = monitor.rows
    ideal = rs[-1]["ideal"] - rs[0]["ideal"]
    gained = (rs[-1]["m_blue"] + rs[-1]["produced"] - rs[-1]["oil_out"]) - (
        rs[0]["m_blue"] + rs[0]["produced"] - rs[0]["oil_out"]
    )
    (out / f"{tag}_report.md").write_text(
        f"# {title}\n\n"
        f"- Stage **{tag}** | Ca {(args.stab_ca if is_stab else args.ca):.2e} | u_in {u_in:.3e} | "
        f"block {blk} | M {args.m:g} | theta {args.theta:g} (oil) | sigma {args.sigma} | "
        f"{'fp32' if FP32 else 'fp64'}\n"
        f"- Structure {args.structure.name} | domain {'x'.join(map(str, solid_np.shape))} "
        f"(piston x=0, water buffer x=1..{x0 - 1}, rock x={x0}..{x1 - 1}, outlet buffer x={x_ob}..{nx - 1})\n"
        f"- Outlet: bc.outlet_zero_gradient (open, colour untouched) + per-step red->blue scrub in the buffer\n"
        f"- Finished: **{finish_type}** — {finish_detail} | steps {step0} -> {step} | {elapsed / 3600:.2f} h\n\n"
        f"## Result\n\n"
        f"- Rock S_w {rs[0]['sw']:.5f} -> {rs[-1]['sw']:.5f} (S_o {1 - rs[0]['sw']:.5f} -> **{1 - rs[-1]['sw']:.5f}**)\n"
        f"- dp {rs[0]['dp']:+.6f} -> {rs[-1]['dp']:+.6f}\n"
        f"- Trapped oil at end: {rs[-1]['trapped_blobs']:.0f} blobs, {rs[-1]['trapped_cells']:.0f} cells, "
        f"mass {rs[-1]['trapped_mass']:.1f}\n"
        f"- Produced: water {produced_tot:+.1f} (positive = leaving, negative = the outlet feeding the sample), "
        f"oil {oil_out_tot:.1f}\n"
        # only meaningful for the flood: stage 1 injects ~nothing, so ideal ~ 0
        + (f"- Colour ledger (blue): {gained / ideal:.3f}x ideal {ideal:.1f}\n" if not is_stab and ideal > 0 else "")
        + f"- Artifacts: {tag}_{args.name}.gif, frames_{tag}/, metrics_{tag}.csv, "
        f"run_meta_{tag}.json"
        + (f", {tag}_{args.name}.h5 ({args.field_vars})\n" if write_h5 else " (no fields.h5 — see --stab-h5)\n"),
        encoding="utf-8",
    )
    # stage-scoped names so both stages coexist in one folder (RunMonitor writes fixed
    # metrics.csv / sw.svg, and its ckpt pruning would eat the other stage's states).
    # sw.svg is an opt-in curve (--plots), hence the per-name existence check.
    for src, dst in (("metrics.csv", f"metrics_{tag}.csv"), ("sw.svg", f"sw_{tag}.svg")):
        if (out / src).exists():
            (out / src).replace(out / dst)
    for ck in out.glob("ckpt_step*_sat*.npz"):
        ck.replace(out / f"ckpt_{tag}_{ck.name[5:]}")
    print(f"[{tag}] done: S_w {rs[0]['sw']:.5f} -> {rs[-1]['sw']:.5f}, {finish_type}, {elapsed / 60:.1f} min")
    return state, step, sw


def main():
    args = parse_args()
    if args.plots:
        viz.enable_plots(True)
    args.name = args.name or args.structure.stem
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    omega, omega2, nu_r, nu_b = omega_pair(args.m, args.nu0)

    # ---- geometry (shared by both stages) -------------------------------------
    rock = np.load(args.structure)
    solid_np, reg = build_domain(rock, buf_in=args.buf_in, buf_out=args.buf_out)
    solid = jnp.asarray(solid_np)
    nz, ny, nx = solid_np.shape
    x0, x1 = reg["rock"]
    x_ob = reg["outbuf"][0]
    sealed_np = solid_np.copy()
    sealed_np[:, :, 0] = True  # piston face: sealed for the wall normals and every metric
    pore = ~sealed_np
    xx = np.arange(nx)[None, None, :]
    geom = {
        "solid_np": solid_np, "solid": solid, "reg": reg, "nx": nx, "x0": x0, "x1": x1, "x_ob": x_ob,
        "sealed_np": sealed_np, "pore": pore, "a_open": int(pore[:, :, 1].sum()),
        "rock_m": pore & (xx >= x0) & (xx < x1), "outbuf_m": pore & (xx >= x_ob),
        "inbuf_p": pore[:, :, 1 : reg["inbuf"][1]],
        "xx3": np.broadcast_to(xx, solid_np.shape).astype(np.float64),
        "led_m": pore,  # ledger region includes the outlet plane: a non-pinning outlet lets blue accumulate there
        "camera": viz3d.flow_camera(solid_np.shape),
    }
    geom["pv_rock"] = int(geom["rock_m"].sum())
    nw = color3d.wall_normals(jnp.asarray(sealed_np))
    theta_field = jnp.full((nz, ny, nx), float(args.theta)).at[:, :, 1].set(float(THETA_FACE))
    params = color3d.Params(omega=omega, omega2=omega2, sigma=args.sigma, beta=args.beta, theta=theta_field, mrt_chi=args.chi)
    geom["nw"], geom["theta"] = nw, theta_field
    print(f"{args.name}: domain {nz}x{ny}x{nx} | rock pore volume {geom['pv_rock']} | "
          f"nu_red {nu_r:.4f} nu_blue {nu_b:.4f} | {'fp32' if FP32 else 'fp64'}")  # fmt: skip

    geom["step_fn"], gpu_mesh = None, None
    if args.gpus > 1:
        if solid_np.shape[0] % args.gpus:
            sys.exit(f"--gpus {args.gpus}: nz={solid_np.shape[0]} is not divisible (z-sharded halo backend)")
        gpu_mesh = mesh = multigpu.mesh(args.gpus)
        print(f"sharding along z over {args.gpus} devices: {[str(d) for d in mesh.devices.reshape(-1)]}")
        step_solid = np.asarray(solid).copy()
        step_solid[:, :, 0] = True  # statics go in unsharded; the piston seals x=0
        geom["step_fn"] = multigpu.staged_halo_step(mesh, params, step_solid, nw=nw)
        geom["solid"], geom["nw"], geom["theta"] = multigpu.shard((solid, nw, theta_field), mesh)

    meta_extra = {
        "so_target": float(args.so), "theta": float(args.theta), "theta_face": float(THETA_FACE),
        "M": float(args.m), "nu_red": float(nu_r), "nu_blue": float(nu_b), "pv_rock": geom["pv_rock"],
        "inlet": "piston blue (inlet_sa_red=0)", "outlet": "outlet_zero_gradient + per-step red->blue buffer scrub",
        "regions": {k: list(v) for k, v in reg.items()},
    }  # fmt: skip

    eq_path = out / EQ_STATE
    so_init = None

    # ---- stage 1: equilibrium transition --------------------------------------
    if args.stage in ("both", "stab"):
        print(f"\nPMI morphological init: target S_o {args.so:g} (theta_pmi {PMI_THETA:g}) ...", flush=True)
        t_pmi = time.perf_counter()
        label, so_init = pmi_oil_label(rock, args.so, num_workers=args.pmi_workers)
        print(f"  achieved S_o {so_init:.4f} in {time.perf_counter() - t_pmi:.0f}s", flush=True)
        state = init_state(solid_np, reg, label)
        if gpu_mesh is not None:
            state = multigpu.shard(state, gpu_mesh)
        state, step, sw = run_stage("stab", state, 0, so_init, args, geom, params, meta_extra)
        np.savez(eq_path, fR=np.asarray(state.fR), fB=np.asarray(state.fB), step=step,
                 so_init=np.float64(so_init), sw=np.float64(sw))  # fmt: skip
        print(f"equilibrated state -> {eq_path}")

    # ---- stage 2: constant-rate flood -----------------------------------------
    if args.stage in ("both", "flood"):
        step0 = 0
        if args.resume and (ck := stats.latest_checkpoint(out)) is not None:
            d = np.load(ck)
            state = color3d.State(jnp.asarray(d["fR"]), jnp.asarray(d["fB"]))
            step0, so_init = int(d["step"]), float(d["so_init"])
            print(f"\nresuming flood from {ck.name}: step {step0}")
        else:
            if not eq_path.exists():
                sys.exit(f"missing {eq_path} — run --stage stab first (or --stage both)")
            d = np.load(eq_path)
            state = color3d.State(jnp.asarray(d["fR"]), jnp.asarray(d["fB"]))
            so_init = float(d["so_init"])
            print(f"\nflood starts from the equilibrated state ({eq_path.name}, S_w {float(d['sw']):.5f})")
        if gpu_mesh is not None:
            state = multigpu.shard(state, gpu_mesh)
        run_stage("flood", state, step0, so_init, args, geom, params, meta_extra)


if __name__ == "__main__":
    main()
