"""Multi-GPU scaling of the 3D drainage stack: halo-exchange backends vs GSPMD.

Benchmarks the same jitted ``porous3d.drain`` (default MRT + CSF + akai wetting,
rate-controlled velocity inlet) under the three z-sharding backends of ``bob.multigpu``:

* ``--mode gspmd``  — automatic partitioning: XLA's GSPMD partitioner turns every
  z-crossing ``jnp.roll`` into its own one-plane CollectivePermute (~100+ small
  messages per step for the default CSF stack).
* ``--mode halo``   — ``multigpu.halo_step``: the unmodified ``color3d.step`` runs
  inside ``shard_map`` on a z-slab padded with ghost planes; one packed ghost-plane
  exchange per color per step (2 ``ppermute`` pairs), at the price of
  ``required_halo`` (default 5) redundant ghost planes recomputed per slab side.
* ``--mode staged`` — ``multigpu.staged_halo_step``: per-stage thin halos — phi
  exchanged at depth ``required_phi_halo`` (4) before collision and the post-collision
  populations at depth 1 before streaming, so collision never runs on ghost planes.

Precision defaults to the production float32 (plain ``import bob``); ``--fp64``
opts into float64 for comparability with the
``demo/multi_gpu_scaling`` table. Each invocation appends one row to
``output/halo_scaling.csv``:

    size, mode, precision, halo, ngpu, steps, seconds, mlups, saturation

``--report`` aggregates per (size, precision): speedup/efficiency vs the
same-mode 1-GPU baseline, the halo/gspmd throughput ratio, and the physics
checks — saturation drift across device counts within a mode, and across the
two modes at equal device count (fp64: < 1e-9; fp32: < 1e-4, ULP-level
divergence at the contact line between different XLA programs is expected there).

Usage:
    uv run --extra cuda python multi_gpu_halo_demo.py --gpus 8 --size 256 --mode halo
    uv run --extra cuda python multi_gpu_halo_demo.py --gpus 8 --size 256 --mode gspmd
    uv run python multi_gpu_halo_demo.py --report
    # CPU smoke (4 emulated devices):
    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
        uv run python multi_gpu_halo_demo.py --gpus 4 --size 64 --steps 20 --block 10 --mode halo
"""

from __future__ import annotations

import argparse
import csv
import time
from functools import partial
from pathlib import Path

ASSET = Path(__file__).resolve().parents[2] / "test" / "assets" / "sphere256.npy"
OUT = Path(__file__).resolve().parent / "output"
FIELDS = ("size", "mode", "precision", "halo", "ngpu", "steps", "seconds", "mlups", "saturation")


def bench(args):
    import jax

    if args.fp64:  # must land before bob imports (lattice constants build at import)
        jax.config.update("jax_enable_x64", True)
    from bob import color3d, multigpu, porous3d

    precision = "fp64" if jax.config.jax_enable_x64 else "fp32"
    m = multigpu.mesh(args.gpus, platform=args.platform or None)
    print(f"mesh: {[str(d) for d in m.devices.reshape(-1)]}  precision: {precision}")

    if args.size == 64:
        crop = porous3d.sphere64(ASSET)
    else:
        vol = porous3d.load_structure(ASSET)
        lo = (256 - args.size) // 2
        crop = vol[lo : lo + args.size, lo : lo + args.size, lo : lo + args.size]
    solid = porous3d.with_buffers(crop, buffer=10)
    nz, ny, nx = solid.shape
    state = porous3d.init_drainage(nz, ny, nx, n_red=4)
    if nz % args.gpus:
        raise SystemExit(f"nz={nz} not divisible by {args.gpus} devices")

    params = color3d.Params(omega=args.omega, sigma=args.sigma, beta=0.7, theta=args.theta)
    nw = color3d.wall_normals(solid)
    step_fn, halo_used = None, 0
    if args.mode == "halo":
        halo_used = args.halo or multigpu.required_halo(params)
        step_fn = multigpu.halo_step(m, params, solid, nw=nw, halo=args.halo or None)
    elif args.mode == "staged":
        halo_used = args.halo or multigpu.required_phi_halo(params)
        step_fn = multigpu.staged_halo_step(m, params, solid, nw=nw, phi_halo=args.halo or None)
    state, solid, nw = multigpu.shard((state, solid, nw), m)
    mu = (1.0 / args.omega - 0.5) / 3.0
    u_in = args.ca * args.sigma / mu

    if args.steps % args.block:
        raise SystemExit(f"--steps {args.steps} must be a multiple of --block {args.block}")

    # One fixed-length compilation; out_shardings + donation keep the dispatch
    # loop a cache hit (jax 0.6.2 re-lowering crash otherwise) and halve memory.
    sharding = jax.tree_util.tree_map(lambda x: x.sharding, state)

    @partial(jax.jit, out_shardings=sharding, donate_argnums=0)
    def run_block(s):
        return porous3d.drain(
            s, params, solid, 0, u_in, args.block, nw=nw, inlet="velocity", outlet_sa_red=0.0, step_fn=step_fn
        )

    state = jax.block_until_ready(run_block(state))  # compile + settle (untimed)
    t0 = time.perf_counter()
    for _ in range(args.steps // args.block):
        state = run_block(state)
    state = jax.block_until_ready(state)
    dt = time.perf_counter() - t0

    mlups = nz * ny * nx * args.steps / dt / 1e6
    sat = porous3d.saturation(state, solid)
    print(
        f"size {args.size}^3 ({nz}x{ny}x{nx}), {args.gpus} device(s), mode={args.mode}"
        f"{f' (halo {halo_used})' if halo_used else ''}: {args.steps} steps in {dt:.2f} s "
        f"= {mlups:.1f} MLUPS, saturation {sat:.6f}"
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
                "mode": args.mode,
                "precision": precision,
                "halo": halo_used,
                "ngpu": args.gpus,
                "steps": args.steps,
                "seconds": f"{dt:.3f}",
                "mlups": f"{mlups:.2f}",
                "saturation": f"{sat:.12f}",
            }
        )
    print(f"appended -> {path}")


def report(args):
    """Aggregate the CSV per (size, precision): both modes side by side, speedup /
    efficiency vs the same-mode 1-GPU base, halo/gspmd ratio, and the saturation
    parity checks (across device counts within a mode; across modes at equal
    count). The speedup curve is opt-in (--plots); the table and report.md are not."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from bob.utils import viz  # deferred like the bob imports in bench(): precision is set at import

    if args.plots:
        viz.enable_plots(True)
    plots = viz.plots_enabled()
    stem = Path(args.csv).stem
    md, svg = (
        (f"{stem}_report.md" if stem != "halo_scaling" else "report.md"),
        (f"{stem}_speedup.svg" if stem != "halo_scaling" else "speedup.svg"),
    )
    rows = list(csv.DictReader((OUT / args.csv).open()))
    groups = sorted({(int(r["size"]), r["precision"]) for r in rows})
    lines = [
        "# Halo-step vs GSPMD — 3D drainage strong scaling (`porous3d.drain`, default MRT+CSF+akai stack)",
        "",
        "| size | prec | mode | GPUs | steps | seconds | MLUPS | speedup | efficiency | vs gspmd | saturation |",
        "|-----:|:----:|:----:|-----:|------:|--------:|------:|--------:|-----------:|---------:|-----------:|",
    ]
    fig, axes_ = plt.subplots(1, len(groups), figsize=(4.6 * len(groups), 4), squeeze=False) if plots else (None, None)
    for gi, (size, prec) in enumerate(groups):
        ax = axes_[0][gi] if plots else None
        grp = [r for r in rows if int(r["size"]) == size and r["precision"] == prec]
        gspmd = {int(r["ngpu"]): r for r in grp if r["mode"] == "gspmd"}
        for mode in ("gspmd", "halo", "staged"):
            sub = sorted((r for r in grp if r["mode"] == mode), key=lambda r: int(r["ngpu"]))
            if not sub:
                continue
            base = next((float(r["mlups"]) for r in sub if int(r["ngpu"]) == 1), None)
            for r in sub:
                n = int(r["ngpu"])
                sp = float(r["mlups"]) / base if base else float("nan")
                eff = sp / n if base else float("nan")
                ratio = float(r["mlups"]) / float(gspmd[n]["mlups"]) if mode != "gspmd" and n in gspmd else None
                lines.append(
                    f"| {size}^3 | {prec} | {mode} | {n} | {r['steps']} | {r['seconds']} | {r['mlups']} "
                    f"| {sp:.2f}x | {eff:.0%} | {f'{ratio:.2f}x' if ratio else '—'} | {r['saturation']} |"
                )
            sats = [float(r["saturation"]) for r in sub]
            drift = max(sats) - min(sats)
            tol = 1e-9 if prec == "fp64" else 1e-4
            ok = "PASS" if drift < tol else "FAIL"
            lines.append(f"\n- {size}^3 {prec} {mode}: saturation drift across device counts {drift:.2e} ({ok}, tol {tol:g})\n")
            if plots:
                ax.plot(
                    [int(r["ngpu"]) for r in sub], [float(r["mlups"]) / base if base else 0 for r in sub], marker="o",
                    label=mode,
                )
        cross = [
            abs(float(r["saturation"]) - float(gspmd[int(r["ngpu"])]["saturation"]))
            for r in grp
            if r["mode"] == "halo" and int(r["ngpu"]) in gspmd
        ]
        if cross:
            tol = 1e-9 if prec == "fp64" else 1e-4
            ok = "PASS" if max(cross) < tol else "FAIL"
            lines.append(f"- {size}^3 {prec}: halo vs gspmd saturation gap (same GPU count) max {max(cross):.2e} ({ok})\n")
        if plots:
            ns = sorted({int(r["ngpu"]) for r in grp})
            ax.plot(ns, ns, "k--", lw=0.8, label="ideal")
            ax.set_xlabel("GPUs")
            ax.set_ylabel("speedup vs same-mode 1 GPU")
            ax.set_xticks(ns)
            ax.set_title(f"{size}^3 {prec}")
            ax.legend()
    if plots:
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
    p.add_argument("--mode", choices=("gspmd", "halo", "staged"), default="halo", help="sharding backend to benchmark")
    p.add_argument("--halo", type=int, default=0, help="override ghost depth (0 = required_halo / required_phi_halo; experiments only)")
    p.add_argument("--fp64", action="store_true", help="float64 (default: production float32)")
    p.add_argument("--steps", type=int, default=200, help="timed drainage steps (multiple of --block)")
    p.add_argument("--block", type=int, default=50, help="scan length per jitted block; warmup = one untimed block")
    p.add_argument("--ca", type=float, default=1e-4, help="capillary number -> u_in = ca*sigma/mu")
    p.add_argument("--omega", type=float, default=1.5)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--theta", type=float, default=135.0)
    p.add_argument("--platform", default="", help="force a JAX platform, e.g. 'gpu' (default: auto)")
    p.add_argument("--csv", default="halo_scaling.csv", help="CSV filename under output/")
    p.add_argument("--report", action="store_true", help="aggregate output/<csv> into a report + speedup plot")
    p.add_argument("--plots", action="store_true", help="also write the analysis curve (speedup.svg); off by default")
    args = p.parse_args()
    if args.report:
        report(args)
    else:
        bench(args)


if __name__ == "__main__":
    main()
