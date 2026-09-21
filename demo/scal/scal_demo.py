"""SCAL drainage-imbibition on real rock: quasi-static Pc-S_w hysteresis loop.

Protocol:
  * drainage by raising the inlet pressure (zou_he_inlet rho_in ladder, pure red),
    each Pc step equilibrated block-wise; stops when S_w stalls vs rising Pc;
  * then imbibition by lowering the inlet pressure through Pc=0 into negative
    (forced imbibition) until S_w stalls again -> residual oil;
  * a synthetic strongly water-wet porous plate (5x5-voxel pores, 20 long,
    theta 150, entry ~0.035 LU > ladder cap 0.028) seals the outlet buffer
    against red, like the laboratory semi-permeable membrane -- the outlet
    buffer stays water-only, so nothing non-physical flows back during
    imbibition. The pore width must exceed the ~4-5 voxel interface width (a
    2-voxel pore breaches at Pc ~ 0.015 instead of its nominal ~0.1) and the
    pore must be long enough to damp the dissolved-red gradient ahead of the
    scrub/outlet sinks;
  * Ca(t) = mu*u/sigma logged per block (interface, bulk and Darcy-mean flux) --
    the quasi-static evidence.

Fluid/solver settings shared with case/drainage (sigma 0.05, nu 0.04 M=1, theta 135,
MRT chi 0.8, fp32); geometry test/assets/bentheimer128.npy; ladder calibrated by
plate_check/throat_ladder.py. Artifacts: metrics.csv + curves, 3D frames/GIF,
fields.h5+xdmf, status-named checkpoints with the ladder state stored inside
(auto-resume), run_meta.json.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bob import bc, color3d, d3q19, multigpu
from bob.utils import stats, viz, viz3d
from bob.utils.file import FieldWriter, RunMeta

SIGMA, NU, THETA_ROCK, THETA_PLATE, BETA, CHI = 0.05, 0.04, 135.0, 150.0, 0.7, 0.8
CS2 = 1.0 / 3.0


# ---------------------------------------------------------------- geometry


def build_plate(nz, ny, thick=20, hole=5, pitch=8, offset=3):
    """Water-wet porous plate: solid slab pierced by hole x hole square through-pores
    on a pitch lattice starting at ``offset`` (pitch 8 divides 128 -> the y/z-periodic
    tiling stays clean; the offset clears solid boundary walls). Default 5-voxel pores,
    20 long, theta 150: pore width must exceed the ~4-5 vox diffuse-interface width
    (else the meniscus is unresolved — a 2x2-voxel pore breaches at Pc=0.015, not the
    nominal ~0.1) and the pore length must damp the dissolved-red gradient ahead of
    the scrub/outlet sinks. Entry pressure ~4*sigma*|cos(theta)|/hole ~= 0.035 —
    above the ladder cap 0.028."""
    plate = np.ones((nz, ny, thick), bool)
    for z0 in range(offset, nz, pitch):
        for y0 in range(offset, ny, pitch):
            plate[z0 : z0 + hole, y0 : y0 + hole, :] = False
    return plate


def build_domain(rock, buf_in=10, buf_out=6, thick=20, hole=5, pitch=8, offset=3):
    """[inlet buffer | rock | plate (direct contact) | outlet buffer] along x, with
    solid z/y boundary planes over the whole domain (no-slip walls, mask the wrap --
    the same core-holder convention as ``porous3d.with_buffers``).
    Returns (solid, regions) with half-open x-intervals per region."""
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


def theta_field_for(shape, regions, theta_rock=THETA_ROCK, theta_plate=THETA_PLATE):
    """Per-cell contact angle: rock (and inlet buffer, no walls there anyway) at
    theta_rock; plate AND outlet buffer at theta_plate so every wall the plate owns
    -- including its back face -- is strongly water-wet."""
    th = np.full(shape, float(theta_rock))
    th[:, :, regions["plate"][0] :] = float(theta_plate)
    return th


# ---------------------------------------------------------------- ladder controller


class Ladder:
    """Quasi-static SCAL pressure schedule: drainage up, then imbibition down.

    ``advance(sw_eq)`` is called once per equilibrated pressure step with the
    equilibrium water saturation. Stall = |dSw| < stall_ds across stall_n
    consecutive completed steps, counted only after the leg has activated
    (|sw - sw at leg start| >= act_ds) -- below the entry pressure nothing
    moves, and that flatness is not a plateau. Drainage stalls (or hits
    pc_cap) -> reverse; imbibition stalls (or hits the -floor_factor*pc_max
    floor) -> done."""

    def __init__(self, pc0, dpc, pc_cap=None, stall_ds=0.005, stall_n=2, floor_factor=1.0, act_ds=0.02):
        self.leg, self.pc, self.dpc = "drain", float(pc0), float(dpc)
        self.pc_cap = None if pc_cap is None else float(pc_cap)
        self.stall_ds, self.stall_n = float(stall_ds), int(stall_n)
        self.floor_factor, self.act_ds = float(floor_factor), float(act_ds)
        self.sw_prev, self.stall, self.pc_max = None, 0, float(pc0)
        self.sw_leg0 = None  # S_w at the first completed step of the current leg
        self.done = False

    def advance(self, sw_eq):
        sw_eq = float(sw_eq)
        if self.sw_leg0 is None:
            self.sw_leg0 = sw_eq
        active = abs(sw_eq - self.sw_leg0) >= self.act_ds
        if active and self.sw_prev is not None and abs(sw_eq - self.sw_prev) < self.stall_ds:
            self.stall += 1
        else:
            self.stall = 0
        self.sw_prev = sw_eq
        if self.leg == "drain":
            capped = self.pc_cap is not None and self.pc + self.dpc > self.pc_cap
            if self.stall >= self.stall_n or capped:
                self.leg, self.stall, self.sw_leg0 = "imb", 0, None
                self.pc -= self.dpc
            else:
                self.pc += self.dpc
                self.pc_max = max(self.pc_max, self.pc)
        else:
            if self.stall >= self.stall_n or self.pc - self.dpc < -self.floor_factor * self.pc_max:
                self.done = True
            else:
                self.pc -= self.dpc

    def to_dict(self):
        return {
            "leg": self.leg,
            "pc": self.pc,
            "dpc": self.dpc,
            "pc_cap": self.pc_cap,
            "stall_ds": self.stall_ds,
            "stall_n": self.stall_n,
            "floor_factor": self.floor_factor,
            "act_ds": self.act_ds,
            "sw_prev": self.sw_prev,
            "sw_leg0": self.sw_leg0,
            "stall": self.stall,
            "pc_max": self.pc_max,
            "done": self.done,
        }

    @classmethod
    def from_dict(cls, d):
        lad = cls(
            d["pc"],
            d["dpc"],
            pc_cap=d["pc_cap"],
            stall_ds=d["stall_ds"],
            stall_n=d["stall_n"],
            floor_factor=d["floor_factor"],
            act_ds=d["act_ds"],
        )
        lad.leg, lad.sw_prev, lad.stall = d["leg"], d["sw_prev"], int(d["stall"])
        lad.sw_leg0 = d["sw_leg0"]
        lad.pc_max, lad.done = float(d["pc_max"]), bool(d["done"])
        return lad


# ---------------------------------------------------------------- driver


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--structure", type=Path, default=Path("test/assets/bentheimer128.npy"))
    p.add_argument("--out", type=Path, default=Path("demo/scal/bentheimer128"))
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--ladder-json", type=Path, default=Path("demo/scal/plate_check/output/ladder.json"))
    p.add_argument("--pc0", type=float, default=None, help="override ladder.json")
    p.add_argument("--dpc", type=float, default=None)
    p.add_argument("--pc-cap", type=float, default=None)
    p.add_argument("--stall-ds", type=float, default=0.005, help="|dSw| per Pc step below which it counts as stalled")
    p.add_argument("--stall-n", type=int, default=2, help="consecutive stalled steps that end a leg")
    p.add_argument("--block", type=int, default=2000, help="steps per equilibration block")
    p.add_argument("--tol-eq", type=float, default=5e-4, help="|dSw| per block = equilibrated")
    p.add_argument(
        "--tol-ca",
        type=float,
        default=1e-5,
        help="equilibrium additionally requires Ca_flux = mu*|<u_x>_rock|/sigma <= this "
        "(signed Darcy mean: spurious swirls cancel, real through-flow does not; 0 = off)",
    )
    p.add_argument("--max-blocks", type=int, default=150, help="per-Pc-step block cap (logged CAPPED)")
    p.add_argument(
        "--pre-blocks",
        type=int,
        default=0,
        help="block cap per rung while the leg has not yet activated (fine-ladder pressure "
        "buildup: march the ladder fast through the dead zone below entry; 0 = off)",
    )
    p.add_argument("--buf-in", type=int, default=10)
    p.add_argument("--buf-out", type=int, default=6)
    p.add_argument("--plate-thick", type=int, default=20)
    p.add_argument("--hole", type=int, default=5, help="plate pore width (voxels; must exceed the ~4-5 vox interface width)")
    p.add_argument("--pitch", type=int, default=8, help="plate pore pitch (voxels)")
    p.add_argument("--hole-offset", type=int, default=3, help="first pore corner (clears the z/y walls)")
    p.add_argument(
        "--scrub",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="repaint stray red past the plate to blue each block (mass-conserving color swap, "
        "logged as 'scrubbed'): the membrane's suction side -- broken-through NW is produced, "
        "never returned. The plate holds the capillary barrier; this only removes mist.",
    )
    p.add_argument("--frame-blocks", type=int, default=20, help="render a 3D frame every N blocks (plus each equilibrium)")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--plots", action="store_true",
                   help="also write the analysis curves (ca_curve.svg, pc_sw_live.svg, sw.svg); off by default")
    p.add_argument("--rock-alpha", type=float, default=0.04)
    p.add_argument("--resume", type=Path, default=None, help="run dir (newest ckpt) or ckpt .npz")
    p.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="shard the domain along z across the first N devices (bob.multigpu; nz must divide)",
    )
    return p.parse_args()


def init_state_pressure(solid, x_red, rho_in0):
    """Pressure-consistent IC: red at rho_in0 for x < x_red (inlet buffer), blue at 1.0
    beyond, u=0 equilibrium -- mechanical balance with the BCs, no t=0 shock."""
    nz, ny, nx = solid.shape
    xx = jnp.arange(nx)[None, None, :] * jnp.ones((nz, ny, 1))
    rhoR = jnp.where(xx < x_red, rho_in0, 0.0)
    rhoB = jnp.where(xx >= x_red, 1.0, 0.0)
    w = d3q19.W[:, None, None, None]
    return color3d.State(w * rhoR[None], w * rhoB[None])


def main():
    args = parse_args()
    if args.plots:
        viz.enable_plots(True)
    name = args.name or f"scal_{args.structure.stem}"
    out = args.out
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    lad_cfg = json.loads(args.ladder_json.read_text()) if args.ladder_json.exists() else {}
    pc0 = args.pc0 if args.pc0 is not None else lad_cfg["pc0"]
    dpc = args.dpc if args.dpc is not None else lad_cfg["dpc"]
    pc_cap = args.pc_cap if args.pc_cap is not None else lad_cfg.get("pc_cap")

    rock = np.load(args.structure)
    solid_np, reg = build_domain(
        rock, buf_in=args.buf_in, buf_out=args.buf_out, thick=args.plate_thick, hole=args.hole, pitch=args.pitch,
        offset=args.hole_offset,
    )
    solid = jnp.asarray(solid_np)
    pore = ~solid_np
    rock_m = pore.copy()
    rock_m[:, :, : reg["rock"][0]] = False
    rock_m[:, :, reg["rock"][1] :] = False
    outbuf_m = pore.copy()
    outbuf_m[:, :, : reg["outbuf"][0]] = False
    inbuf_p = pore[:, :, 1 : reg["inbuf"][1]]  # pressure probe: inlet buffer sans BC plane
    nw = color3d.wall_normals(solid)
    theta = jnp.asarray(theta_field_for(solid_np.shape, reg))
    camera = viz3d.flow_camera(solid_np.shape)

    blk = int(args.block)

    # ---- init / resume --------------------------------------------------
    if args.resume is not None:
        ckpt_path = stats.latest_checkpoint(args.resume) if args.resume.is_dir() else args.resume
        if ckpt_path is None:
            sys.exit(f"no checkpoint under {args.resume}")
        ck = np.load(ckpt_path)
        state = color3d.State(jnp.asarray(ck["fR"]), jnp.asarray(ck["fB"]))
        ladder = Ladder.from_dict(json.loads(str(ck["ladder"])))
        step0 = int(ck["step"])
        print(f"resumed {ckpt_path.name}: step {step0}, {ladder.leg} pc={ladder.pc:.5f}")
    else:
        ladder = Ladder(pc0, dpc, pc_cap=pc_cap, stall_ds=args.stall_ds, stall_n=args.stall_n)
        state = init_state_pressure(solid_np, reg["inbuf"][1], 1.0 + 3.0 * pc0)
        step0 = 0
        for stale in frames_dir.glob("frame_*.png"):
            stale.unlink()

    # Multi-GPU: shard state + every 3D field the step closes over along z (flow is x,
    # so the slabs cut across neither inlet, outlet nor plate); out_shardings +
    # donate_argnums pin the dispatch loop to one compilation and halve device memory
    # (see the bob.multigpu module docstring).
    omega = 1.0 / (3.0 * NU + 0.5)
    params = color3d.Params(omega=omega, omega2=omega, sigma=SIGMA, beta=BETA, theta=theta, mrt_chi=CHI)
    mesh = step_fn = None
    if args.gpus > 1:
        mesh = multigpu.mesh(args.gpus)
        print(f"sharding along z over {args.gpus} devices: {[str(d) for d in mesh.devices.reshape(-1)]}")
        # default multi-GPU backend: staged thin-halo shard_map; statics (incl. the
        # per-cell theta field inside params) go in unsharded — it pre-pads them.
        step_fn = multigpu.staged_halo_step(mesh, params, solid_np, nw=nw)
        state, solid, nw, theta = multigpu.shard((state, solid, nw, theta), mesh)

    def _run_block(state, rho_in):
        def body(s, _):
            s = step_fn(s) if step_fn is not None else color3d.step(s, params, solid, nw=nw)
            s = bc.zou_he_inlet(s, solid=solid, rho_in=rho_in, sa_red=1.0)
            s = bc.zou_he_outlet(s, solid=solid, rho_out=1.0, sa_red=0.0)
            return s, None

        s, _ = jax.lax.scan(body, state, None, length=blk)
        return s

    sharding = jax.tree_util.tree_map(lambda x: x.sharding, state)
    run_block = jax.jit(_run_block, out_shardings=sharding, donate_argnums=0)

    fields = (
        "step", "leg", "pc", "pc_meas", "sw", "ca_iface", "ca_bulk", "ca_flux",
        "plate_leak", "scrubbed", "u_max", "m_red", "m_blue",
    )  # fmt: skip
    meta = RunMeta(
        out / "run_meta.json",
        params=params._replace(theta=THETA_ROCK),  # scalar theta: the per-cell field is not JSON-able
        solid=solid_np,
        geometry_source=str(args.structure),
        resume_step=step0,
        extra={
            "protocol": "SCAL pressure ladder",
            "pc0": pc0,
            "dpc": dpc,
            "pc_cap": pc_cap,
            "theta_plate": THETA_PLATE,
            "plate": f"{args.hole}x{args.hole} pores pitch {args.pitch} thick {args.plate_thick} offset {args.hole_offset}",
        },
    )
    rows = stats.carry_rows(out, step0, fields=fields) if step0 else None
    if rows:  # backfill fields added since the carried segment (write_csv headers from row 0)
        rows = [{**dict.fromkeys(fields, 0.0), **r} for r in rows]
    monitor = stats.RunMonitor(out, title=name, curve_field="sw", rows=rows, meta=meta)
    fw = FieldWriter(
        out / "fields.h5",
        solid_np,
        fields=("phi", "p", "umag"),
        mask_solid=True,
        attrs={
            "sigma": SIGMA,
            "nu": NU,
            "theta": THETA_ROCK,
            "theta_plate": THETA_PLATE,
            "protocol": "scal-ladder",
            "inlet": "zou_he_pressure",
        },
        meta=meta,
    )

    pc_steps_path = out / "pc_steps.csv"
    pc_rows = []
    if args.resume is not None and pc_steps_path.exists():
        with open(pc_steps_path) as fh:
            pc_rows = [r for r in csv.DictReader(fh) if int(float(r["step_end"])) <= step0]

    x_ob = reg["outbuf"][0]
    scrub_tot = (monitor.rows[-1].get("scrubbed", 0.0) if monitor.rows else 0.0) or 0.0

    def scrub(state):
        """Membrane suction side: swap any red past the plate into blue (mass- and
        momentum-conserving per population). Returns (state, mass swapped)."""
        nonlocal scrub_tot
        red = float(np.asarray(state.fR[:, :, :, x_ob:]).sum())
        if red > 1e-12:
            fB = state.fB.at[:, :, :, x_ob:].add(state.fR[:, :, :, x_ob:])
            fR = state.fR.at[:, :, :, x_ob:].set(0.0)
            state = color3d.State(fR, fB)
            scrub_tot += red
        return state

    def record(step, ladder):
        rhoR, rhoB = color3d.densities(state)
        rR, rB = np.asarray(rhoR), np.asarray(rhoB)
        rho = rR + rB
        sw = float(rB[rock_m].sum() / max(rho[rock_m].sum(), 1e-9))
        p_in = float(rho[:, :, 1 : reg["inbuf"][1]][inbuf_p].mean()) / 3.0
        p_out = float(rho[outbuf_m].mean()) / 3.0
        u = np.asarray(color3d.velocity(state))
        umag = np.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
        phi_n = np.divide(rR - rB, rho, out=np.zeros_like(rho), where=rho > 1e-9)
        iface = rock_m & (np.abs(phi_n) < 0.9)
        bulk = rock_m & (np.abs(phi_n) >= 0.9)
        u_iface = float(umag[iface].mean()) if iface.any() else 0.0
        u_bulk = float(umag[bulk].mean()) if bulk.any() else 0.0
        ca_flux = NU * abs(float(u[0][rock_m].mean())) / SIGMA  # signed Darcy mean: the net through-flow Ca
        monitor.log(
            step=step,
            leg=0 if ladder.leg == "drain" else 1,  # numeric: carry_rows floats every field on resume
            pc=ladder.pc,
            pc_meas=p_in - p_out,
            sw=sw,
            ca_iface=NU * u_iface / SIGMA,
            ca_bulk=NU * u_bulk / SIGMA,
            ca_flux=ca_flux,
            plate_leak=float(rR[outbuf_m].sum()),
            scrubbed=scrub_tot,
            u_max=float(umag[pore].max()),
            m_red=float(rR[pore].sum()),
            m_blue=float(rB[pore].sum()),
        )
        return sw, p_in - p_out, ca_flux

    def frame(step):
        try:
            rhoR, rhoB = color3d.densities(state)
            viz3d.phase_frame(
                frames_dir / f"frame_{step:08d}.png",
                np.asarray(color3d.color_field(rhoR, rhoB)),
                solid_np,
                rock_opacity=args.rock_alpha,
                camera=camera,
                zoom=1.0,
            )
        except Exception as e:
            if step == step0:
                print(f"  [3D render unavailable: {type(e).__name__}: {e}]")

    def ca_curve():
        rs = monitor.rows
        if not viz.plots_enabled() or len(rs) < 2:
            return
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.semilogy([r["step"] for r in rs], [max(r["ca_iface"], 1e-12) for r in rs], lw=1.2, label="Ca interface")
        ax.semilogy([r["step"] for r in rs], [max(r["ca_bulk"], 1e-12) for r in rs], lw=1.2, label="Ca bulk")
        ax.semilogy(
            [r["step"] for r in rs],
            [max(r.get("ca_flux", 0.0), 1e-12) for r in rs],
            lw=1.4,
            color="tab:red",
            label="Ca flux (Darcy)",
        )
        if args.tol_ca > 0:
            ax.axhline(args.tol_ca, color="0.4", ls="--", lw=1.0, label=f"gate {args.tol_ca:.0e}")
        for pr in pc_rows:
            ax.axvline(int(float(pr["step_end"])), color="0.85", lw=0.6, zorder=0)
        ax.set(xlabel="step", ylabel="Ca = mu*u/sigma", title=f"{name}: capillary number along time")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "ca_curve.svg")
        plt.close(fig)

    def live_curve(sw_now, pc_meas_now):
        """Live Pc-S_w loop: completed rungs per leg + the current moving point."""
        if not viz.plots_enabled():
            return
        fig, ax = plt.subplots(figsize=(7, 5))
        for leg_name, colr in (("drain", "#440154"), ("imb", "#21918c")):
            pts = [(float(r["sw_eq"]), float(r["pc_meas"])) for r in pc_rows if str(r["leg"]) == leg_name]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] * 100 for p in pts], "-o", color=colr, ms=5, lw=1.6, label=leg_name)
        ax.plot([sw_now], [pc_meas_now * 100], "*", color="tab:orange", ms=14,
                label=f"now ({ladder.leg} pc={ladder.pc:.4f})")  # fmt: skip
        ax.axhline(0.0, color="0.8", lw=0.8, zorder=0)
        ax.set(xlabel="water saturation S_w", ylabel="P_c (x1e-2 l.u.)", title=f"{name}: live P_c-S_w", xlim=(0, 1))
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / "pc_sw_live.svg")
        plt.close(fig)

    def save_pc_rows():
        if pc_rows:
            viz.write_csv(pc_steps_path, pc_rows)

    def checkpoint(step, sw):
        monitor.checkpoint(
            step, sw, fR=np.asarray(state.fR), fB=np.asarray(state.fB), ladder=np.asarray(json.dumps(ladder.to_dict()))
        )

    # ---- ladder loop -----------------------------------------------------
    step = step0
    if step0 == 0:
        sw, _, _ = record(step, ladder)
        frame(step)
        fw.append(step, state)
    t0 = time.perf_counter()
    while not ladder.done:
        rho_in = jnp.asarray(1.0 + 3.0 * ladder.pc)  # jnp scalar: one compilation for all Pc
        if mesh is not None:
            # commit the scalar to the mesh (replicated): an uncommitted scalar arg into a
            # jit with out_shardings crashes GPU dispatch with 'UnspecifiedValue' has no
            # attribute 'addressable_devices_indices_map' (openxla/xla#24400 workaround)
            rho_in = multigpu.shard(rho_in, mesh)
        sw_prev_block, nblk, equilibrated = None, 0, False
        while True:
            nblk += 1
            state = run_block(state, rho_in)
            step += blk
            sw, pc_meas, ca_flux = record(step, ladder)  # plate_leak logged pre-scrub: the per-block arrival
            if args.scrub:
                state = scrub(state)
            if not np.isfinite(sw):
                sys.exit(f"NON-FINITE sw at step {step} (pc={ladder.pc:.5f}) -- diverged")
            live_curve(sw, pc_meas)
            if nblk % args.frame_blocks == 0:
                frame(step)
            if nblk % 25 == 0:
                checkpoint(step, sw)
                ca_curve()
            # fine-ladder buildup: while the leg hasn't activated (below entry / pre-snap-off
            # dead zone) hold each rung only pre_blocks blocks and march the pressure on
            active = ladder.sw_leg0 is not None and abs(sw - ladder.sw_leg0) >= ladder.act_ds
            cap = args.max_blocks if (active or args.pre_blocks <= 0) else min(args.pre_blocks, args.max_blocks)
            if nblk >= cap:
                break
            eq_sw = sw_prev_block is not None and abs(sw - sw_prev_block) < args.tol_eq
            eq_ca = args.tol_ca <= 0 or ca_flux <= args.tol_ca
            equilibrated = eq_sw and eq_ca
            if equilibrated:
                break
            sw_prev_block = sw
        capped = not equilibrated
        pc_rows.append(
            {
                "leg": ladder.leg,
                "pc": ladder.pc,
                "rho_in": 1.0 + 3.0 * ladder.pc,
                "sw_eq": sw,
                "pc_meas": pc_meas,
                "blocks": nblk,
                "capped": int(capped),
                "step_end": step,
            }
        )
        save_pc_rows()
        rate = (step - step0) / max(time.perf_counter() - t0, 1e-9)
        print(
            f"[{name}] {ladder.leg} pc={ladder.pc:.5f}: S_w={sw:.4f} ({nblk} blocks"
            f"{', CAPPED' if capped else ''})  {rate:.0f} steps/s",
            flush=True,
        )
        frame(step)
        fw.append(step, state)
        checkpoint(step, sw)
        ca_curve()
        ladder.advance(sw)

    # ---- wrap up ----------------------------------------------------------
    frames = sorted(frames_dir.glob("frame_*.png"))
    if len(frames) > 300:
        idx = np.unique(np.linspace(0, len(frames) - 1, 300).round().astype(int))
        frames = [frames[i] for i in idx]
    if frames:
        viz.write_gif(frames, out / f"{name}.gif", args.fps)
    fw.close()
    meta.finish(step)
    drain_rows = [r for r in pc_rows if str(r["leg"]) == "drain"]
    imb_rows = [r for r in pc_rows if str(r["leg"]) == "imb"]
    swi = min(float(r["sw_eq"]) for r in drain_rows) if drain_rows else float("nan")
    sor = (1.0 - max(float(r["sw_eq"]) for r in imb_rows)) if imb_rows else float("nan")
    leak_max = max(r["plate_leak"] for r in monitor.rows)
    (out / "report.md").write_text(
        f"""# {name} -- SCAL pressure-ladder drainage + imbibition

- Geometry: `{args.structure}` + porous plate ({args.hole}x{args.hole} pores, pitch {args.pitch}, thick {args.plate_thick}, theta {THETA_PLATE:.0f} deg)
- Ladder: pc0={pc0}, dpc={dpc}, cap={pc_cap}; stall |dSw|<{args.stall_ds} x{args.stall_n}
- Drainage: {len(drain_rows)} steps -> S_wi = {swi:.4f}
- Imbibition: {len(imb_rows)} steps -> S_or = {sor:.4f}
- Max plate leak (red mass past plate, per block): {leak_max:.3e}  (sealed plate: ~0)
- Total scrubbed red (membrane production side): {scrub_tot:.3e}
- Steps {step}, {(time.perf_counter() - t0) / 3600:.2f} h this segment
- Data: pc_steps.csv (the Pc-S_w loop), metrics.csv; analysis curves with --plots, publication figs via publication_plot.py
"""
    )
    print(f"done: {len(pc_rows)} pressure steps, S_wi={swi:.4f}, S_or={sor:.4f}, plate_leak_max={leak_max:.2e}")


if __name__ == "__main__":
    main()
