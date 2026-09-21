"""Multi-GPU strong-scaling benchmark for the 3D drainage stack.

Runs the same jitted ``porous3d.drain`` (default MRT + CSF + akai wetting,
rate-controlled velocity inlet) on 1/2/4/... devices via ``bob.multigpu`` —
the state is sharded along z (plain GSPMD) and XLA turns the streaming rolls
into halo exchanges; no solver code changes. Each invocation benchmarks one
device count and appends a row to ``output/scaling.csv``:

    size, ngpu, steps, seconds, mlups, saturation

``mlups`` is million lattice-site updates per second over the whole domain
(solids included — they are computed too); ``saturation`` is the end-state red
pore fraction, which must match across device counts at the same size/steps
(the physics-parity check, enforced by ``--report``). ``--report`` aggregates
the CSV into ``report.md``; add ``--plots`` for the ``speedup.svg`` curve.

Usage:
    uv run --extra cuda python multi_gpu_scaling_demo.py --gpus 1 --size 256
    uv run --extra cuda python multi_gpu_scaling_demo.py --gpus 2 --size 256
    uv run --extra cuda python multi_gpu_scaling_demo.py --gpus 4 --size 256
    uv run python multi_gpu_scaling_demo.py --report
    # CPU smoke (4 emulated devices):
    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
        uv run python multi_gpu_scaling_demo.py --gpus 4 --size 64 --steps 20 --block 10
"""

from __future__ import annotations

import argparse
import csv
import time
from functools import partial
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from bob import color3d, multigpu, porous3d
from bob.utils import viz

ASSET = Path(__file__).resolve().parents[2] / "test" / "assets" / "sphere256.npy"
OUT = Path(__file__).resolve().parent / "output"
FIELDS = ("size", "ngpu", "steps", "seconds", "mlups", "saturation")


def build_domain(size):
    """Drainage domain from the committed geometry volume: the canonical 64^3
    crop, a centered size^3 slice, or the full 256^3, each with x-buffers."""
    if size == 64:
        crop = porous3d.sphere64(ASSET)
    else:
        vol = porous3d.load_structure(ASSET)
        lo = (256 - size) // 2
        crop = vol[lo : lo + size, lo : lo + size, lo : lo + size]
    solid = porous3d.with_buffers(crop, buffer=10)
    nz, ny, nx = solid.shape
    state = porous3d.init_drainage(nz, ny, nx, n_red=4)
    return state, solid


def bench(args):
    m = multigpu.mesh(args.gpus, platform=args.platform or None)
    print(f"mesh: {[str(d) for d in m.devices.reshape(-1)]}")
    state, solid = build_domain(args.size)
    nz, ny, nx = solid.shape
    if nz % args.gpus:
        raise SystemExit(f"nz={nz} not divisible by {args.gpus} devices")
    params = color3d.Params(omega=args.omega, sigma=args.sigma, beta=0.7, theta=args.theta)
    nw = color3d.wall_normals(solid)
    state, solid, nw = multigpu.shard((state, solid, nw), m)
    mu = (1.0 / args.omega - 0.5) / 3.0  # dynamic viscosity (rho=1)
    u_in = args.ca * args.sigma / mu

    if args.steps % args.block:
        raise SystemExit(f"--steps {args.steps} must be a multiple of --block {args.block}")

    # Fixed scan length -> one compilation, reused by warmup and every timed block.
    # out_shardings + donation pin the output to the input's sharding and layout, so
    # feeding the state back in is a cache hit — without this, XLA may hand back a
    # differently-laid-out output at large sizes and jax 0.6.2's re-lowering path
    # crashes (UnspecifiedValue TypeError); donation also halves device memory.
    sharding = jax.tree_util.tree_map(lambda x: x.sharding, state)

    @partial(jax.jit, out_shardings=sharding, donate_argnums=0)
    def run_block(s):
        return porous3d.drain(s, params, solid, 0, u_in, args.block, nw=nw, inlet="velocity", outlet_sa_red=0.0)

    state = jax.block_until_ready(run_block(state))  # compile + settle (untimed)
    t0 = time.perf_counter()
    for _ in range(args.steps // args.block):
        state = run_block(state)
    state = jax.block_until_ready(state)
    dt = time.perf_counter() - t0

    mlups = nz * ny * nx * args.steps / dt / 1e6
    sat = porous3d.saturation(state, solid)
    print(
        f"size {args.size}^3 (domain {nz}x{ny}x{nx}), {args.gpus} device(s): "
        f"{args.steps} steps in {dt:.2f} s = {mlups:.1f} MLUPS, saturation {sat:.6f}"
    )

    OUT.mkdir(exist_ok=True)
    path = OUT / args.csv
    fresh = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if fresh:
            w.writeheader()
        w.writerow(
            {
                "size": args.size,
                "ngpu": args.gpus,
                "steps": args.steps,
                "seconds": f"{dt:.3f}",
                "mlups": f"{mlups:.2f}",
                "saturation": f"{sat:.12f}",
            }
        )
    print(f"appended -> {path}")


def report(args):
    """Aggregate the CSV into report.md (per size: speedup vs 1-GPU, parallel
    efficiency, and the cross-device saturation parity check), plus speedup.svg
    with --plots. A non-default --csv writes <stem>_report.md /
    <stem>_speedup.svg instead, keeping the XLA-flag variants apart from the
    main sweep."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = viz.plots_enabled()
    stem = Path(args.csv).stem
    md, svg = ("report.md", "speedup.svg") if args.csv == "scaling.csv" else (f"{stem}_report.md", f"{stem}_speedup.svg")
    rows = list(csv.DictReader((OUT / args.csv).open()))
    sizes = sorted({int(r["size"]) for r in rows})
    lines = [
        "# Multi-GPU strong scaling — 3D drainage (`porous3d.drain`, default MRT+CSF+akai stack)",
        "",
        "| size | GPUs | steps | seconds | MLUPS | speedup | efficiency | saturation |",
        "|-----:|-----:|------:|--------:|------:|--------:|-----------:|-----------:|",
    ]
    fig, ax = plt.subplots(figsize=(5, 4)) if plots else (None, None)
    for size in sizes:
        grp = sorted((r for r in rows if int(r["size"]) == size), key=lambda r: int(r["ngpu"]))
        base = next((float(r["mlups"]) for r in grp if int(r["ngpu"]) == 1), None)
        sats = [float(r["saturation"]) for r in grp]
        for r in grp:
            sp = float(r["mlups"]) / base if base else float("nan")
            eff = sp / int(r["ngpu"]) if base else float("nan")
            lines.append(
                f"| {size}^3 | {r['ngpu']} | {r['steps']} | {r['seconds']} | {r['mlups']} "
                f"| {sp:.2f}x | {eff:.0%} | {r['saturation']} |"
            )
        drift = max(sats) - min(sats)
        ok = "PASS" if drift < 1e-9 else "FAIL"
        lines.append(f"\n- {size}^3 saturation parity across device counts: max drift {drift:.2e} ({ok})\n")
        if plots:
            ax.plot(
                [int(r["ngpu"]) for r in grp], [float(r["mlups"]) / base if base else 0 for r in grp], marker="o",
                label=f"{size}^3",
            )
    if plots:
        ngpus = sorted({int(r["ngpu"]) for r in rows})
        ax.plot(ngpus, ngpus, "k--", lw=0.8, label="ideal")
        ax.set_xlabel("GPUs")
        ax.set_ylabel("speedup vs 1 GPU")
        ax.set_xticks(ngpus)
        ax.legend()
        ax.set_title("bob 3D drainage strong scaling")
        fig.tight_layout()
        fig.savefig(OUT / svg)
        plt.close(fig)
    (OUT / md).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT / md}" + (f" and {OUT / svg}" if plots else ""))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpus", type=int, default=1, help="device count (first N visible devices)")
    p.add_argument("--size", type=int, default=256, choices=(64, 128, 256), help="crop edge from sphere256")
    p.add_argument("--steps", type=int, default=200, help="timed drainage steps (multiple of --block)")
    p.add_argument("--block", type=int, default=50, help="scan length per jitted block; warmup = one untimed block")
    p.add_argument("--ca", type=float, default=1e-4, help="capillary number -> u_in = ca*sigma/mu")
    p.add_argument("--omega", type=float, default=1.5)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--theta", type=float, default=135.0)
    p.add_argument("--platform", default="", help="force a JAX platform, e.g. 'gpu' (default: auto)")
    p.add_argument("--csv", default="scaling.csv", help="CSV filename under output/ (the XLA-flag variants use their own)")
    p.add_argument("--report", action="store_true", help="aggregate output/<csv> into a report + speedup plot")
    p.add_argument("--plots", action="store_true", help="also write the analysis curve (speedup.svg); off by default")
    args = p.parse_args()
    if args.plots:
        viz.enable_plots(True)
    if args.report:
        report(args)
    else:
        bench(args)


if __name__ == "__main__":
    main()
