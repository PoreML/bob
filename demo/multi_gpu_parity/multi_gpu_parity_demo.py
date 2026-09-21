"""Multi-GPU parity demonstration: sharded runs reproduce the single-device physics.

Four scenarios re-run the signature physics of existing demos at 1/2/N devices
via ``bob.multigpu`` (plain GSPMD sharding), each sharded along an adversarial
axis so the slabs cut through walls, inlets, or pressure BCs (the flexible-axis
stress test, as opposed to the standard z-sharded benchmark configuration of
``demo/multi_gpu_scaling``):

  wetting  : sessile droplet between sealed z-walls (demo/wetting pattern),
             shard axis="z" — the wall planes live inside the first/last slab.
  washburn : Zou-He capillary-rise tube along x (demo/washburn pattern),
             shard axis="x" — both pressure-BC planes sit on the sharded axis.
  drainage : sphere-pack drainage, rate-controlled velocity inlet + pressure
             outlet, shard axis="y".
  pc       : capillary-pressure square tube, fixed-pressure layer slabs at
             both x ends (demo/capillary_pressure pattern), shard axis="x".

``--run`` (GPU node; no display stack needed) executes every scenario at each
``--gpus`` count, saves per-block phase-field snapshots (float64 .npz) and the
scenario metric (contact angle / imbibed length / saturation / S_w), then
writes ``output/parity.csv``: per (scenario, ngpu, step) the max |phi_gN -
phi_g1| over the whole domain and the metric difference. ``--render``
(workstation) turns the snapshots into, per scenario: side-by-side PyVista
renders (1 GPU | N GPUs | midplane |delta phi| heatmap) per block -> frames/
+ an animated GIF, a final still, a metric-overlay plot (one curve per device
count — they must coincide), a log-scale phi-drift curve, and report.md.

PASS criterion — metric agreement, not raw-field bits: over a few dozen steps
the sharded state matches single-device to float64 roundoff (pinned at 1e-12
by test/test_multigpu.py on every axis). A partitioned compilation may differ
by ~1 ULP per step (GSPMD changes XLA's fusion/reassociation) and
moving-interface dynamics amplify such seeds exponentially at the contact
line — measured under CPU emulation: localized under the droplet rim, not at
slab boundaries, max |dphi| 5e-16 -> 2.5e-6 over 100 steps while the contact
angle agrees to 6 digits (the decorrelation any MPI decomposition shows). On
an 8x H100 node all four scenarios are bit-identical (|dphi| = 0) across
1/2/8 devices over the full 3000-12000 step runs.

Usage:
    uv run --extra cuda python multi_gpu_parity_demo.py --run --gpus 1 2 8
    uv run python multi_gpu_parity_demo.py --render --gpus 1 2 8
    # CPU smoke (4 emulated devices, tiny step counts):
    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
        uv run python multi_gpu_parity_demo.py --run --gpus 1 4 --smoke
"""

from __future__ import annotations

import argparse
import csv
import time
from functools import partial
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from bob import bc, color3d, d3q19, misc, multigpu, porous3d
from bob.utils import viz

ASSET = Path(__file__).resolve().parents[2] / "test" / "assets" / "sphere256.npy"
OUT = Path(__file__).resolve().parent / "output"
RUNS = OUT / "runs"
PARITY_FIELDS = ["scenario", "axis", "ngpu", "step", "max_abs_dphi", "metric", "metric_ref", "metric_absdiff"]


# ----------------------------------------------------------------------- scenarios
class Scenario:
    """name, shard axis, block/nblocks, geometry + params + one-step body, metric."""

    def __init__(self, name, axis, block, nblocks, slice_axis, camera):
        self.name, self.axis, self.block, self.nblocks = name, axis, block, nblocks
        self.slice_axis, self.camera = slice_axis, camera  # midplane axis for the diff panel; render camera


def scen_wetting(smoke=False):
    sc = Scenario("wetting", "z", 20 if smoke else 200, 5 if smoke else 20, 1, "xz")
    nz, n, R = 40, 64, 12
    solid = np.zeros((nz, n, n), bool)
    solid[0] = solid[-1] = True
    params = color3d.Params(omega=1.0, sigma=0.02, beta=0.95, theta=60.0)

    def build():
        zz, yy, xx = jnp.meshgrid(jnp.arange(nz), jnp.arange(n), jnp.arange(n), indexing="ij")
        red = ((xx - (n - 1) / 2.0) ** 2 + (yy - (n - 1) / 2.0) ** 2 + (zz - 1.0) ** 2 <= R**2) & (zz >= 1)
        W = d3q19.W[:, None, None, None]
        state = color3d.State(W * jnp.where(red, 1.0, 0.0)[None], W * jnp.where(red, 0.0, 1.0)[None])
        return state, jnp.asarray(solid)

    def make_body(solid_j, nw):
        return lambda s: color3d.step(s, params, solid=solid_j, nw=nw)

    sc.solid, sc.build, sc.make_body = solid, build, make_body
    sc.needs_nw = True
    sc.metric_name, sc.metric_tol = "contact_angle_deg", 0.2
    sc.metric = lambda state: float(misc.contact_angle(state, wall_plane=0))
    return sc


def scen_washburn(smoke=False):
    sc = Scenario("washburn", "x", 60 if smoke else 600, 5 if smoke else 20, 0, "vertical")
    nz, ny, nx, R, res = 12, 12, 104, 5, 6
    yc, zc = (ny - 1) / 2.0, (nz - 1) / 2.0
    zz2, yy2 = np.indices((nz, ny))
    tube = ((yy2 - yc) ** 2 + (zz2 - zc) ** 2) >= R**2
    solid = np.zeros((nz, ny, nx), bool)
    solid[:, :, res : nx - res] = tube[:, :, None]
    params = color3d.Params(omega=1.0, sigma=1.0 / 45, beta=0.95, theta=30.0, omega2=1.0 / 0.6)

    def build():
        return porous3d.init_drainage(nz, ny, nx, n_red=res), jnp.asarray(solid)

    def make_body(solid_j, nw):
        def body(s):
            s = color3d.step(s, params, solid=solid_j, nw=nw)
            s = bc.zou_he_inlet(s, solid=solid_j, rho_in=1.0, sa_red=1.0)
            s = bc.zou_he_outlet(s, solid=solid_j, rho_out=1.0)
            return s

        return body

    sc.solid, sc.build, sc.make_body = solid, build, make_body
    sc.needs_nw = True
    sc.metric_name, sc.metric_tol = "imbibed_length_lu", 0.5
    sc.metric = lambda state: float(porous3d.filled_length(state, solid, x0=res))
    return sc


def scen_drainage(smoke=False):
    sc = Scenario("drainage", "y", 30 if smoke else 150, 5 if smoke else 20, 0, "flow")
    crop = porous3d.sphere64(ASSET) if ASSET.exists() else np.zeros((16, 16, 16), bool)
    solid = porous3d.with_buffers(crop, buffer=10)
    omega, sigma, ca = 1.5, 0.05, 1e-4
    params = color3d.Params(omega=omega, sigma=sigma, beta=0.7, theta=135.0)
    u_in = ca * sigma / ((1.0 / omega - 0.5) / 3.0)

    def build():
        nz, ny, nx = solid.shape
        return porous3d.init_drainage(nz, ny, nx, n_red=4), jnp.asarray(solid)

    def make_body(solid_j, nw):
        def body(s):
            s = color3d.step(s, params, solid=solid_j, nw=nw)
            s = bc.eqm_velocity_inlet(s, u_in, solid=solid_j, sa_red=1.0)
            s = bc.zou_he_outlet(s, solid=solid_j, rho_out=1.0, sa_red=0.0)
            return s

        return body

    sc.solid, sc.build, sc.make_body = solid, build, make_body
    sc.needs_nw = True
    sc.metric_name, sc.metric_tol = "red_saturation", 1e-3
    sc.metric = lambda state: float(porous3d.saturation(state, solid))
    return sc


def scen_pc(smoke=False):
    sc = Scenario("pc", "x", 30 if smoke else 300, 5 if smoke else 20, 0, "flow")
    n, a, duct, buf, layers = 24, 10, 40, 4, 2
    lo = (n - a) // 2
    cross = np.ones((n, n), bool)
    cross[lo : lo + a, lo : lo + a] = False
    nx = duct + 2 * buf
    solid = np.broadcast_to(cross[:, :, None], (n, n, nx)).copy()
    solid[:, :, :buf] = False
    solid[:, :, nx - buf :] = False
    rho_in, rho_out = 1.0, 0.97  # dP = 0.01 cs^-2 > square-throat entry (~0.006)
    params = color3d.Params(omega=1.0, sigma=1.0 / 45, beta=0.95, theta=135.0)
    W = d3q19.W[:, None, None, None]
    xidx = jnp.arange(nx)
    duct_pore = ~solid & (np.arange(nx) >= buf) & (np.arange(nx) < nx - buf)

    def build():
        xx = np.broadcast_to(np.arange(nx), solid.shape)
        rhoR = jnp.asarray(np.where(xx < buf, rho_in, 0.0))
        rhoB = jnp.asarray(np.where(xx >= buf, rho_out, 0.0))
        state = color3d.State(W * rhoR[None], W * rhoB[None])
        return state, jnp.asarray(solid)

    def make_body(solid_j, nw):
        def body(s):
            s = color3d.step(s, params, solid=solid_j, nw=nw)
            # fixed-pressure layer slabs, where-form (demo/capillary_pressure BCs)
            s = color3d.State(jnp.where(xidx < layers, W * rho_in, s.fR), jnp.where(xidx < layers, 0.0, s.fB))
            s = color3d.State(jnp.where(xidx >= nx - layers, 0.0, s.fR), jnp.where(xidx >= nx - layers, W * rho_out, s.fB))
            return s

        return body

    def s_w(state):
        rR, rB = (np.asarray(z) for z in color3d.densities(state))
        phi = (rR - rB) / np.maximum(rR + rB, 1e-12)
        return float(((phi < 0.0) & duct_pore).sum() / duct_pore.sum())

    sc.solid, sc.build, sc.make_body = solid, build, make_body
    sc.needs_nw = True
    sc.metric_name, sc.metric_tol = "wetting_saturation_duct", 1e-3
    sc.metric = s_w
    return sc


SCENARIOS = {"wetting": scen_wetting, "washburn": scen_washburn, "drainage": scen_drainage, "pc": scen_pc}


# ----------------------------------------------------------------------- run phase
def phase_field(state):
    return np.asarray(color3d.color_field(*color3d.densities(state)))


def run_case(sc, ndev, platform):
    """One scenario at one device count: sharded along sc.axis, per-block snapshots."""
    m = multigpu.mesh(ndev, platform=platform or None)
    state, solid_j = sc.build()
    nw = jnp.asarray(color3d.wall_normals(np.asarray(solid_j))) if sc.needs_nw else None
    state, solid_j, nw = multigpu.shard((state, solid_j, nw), m, axis=sc.axis)
    body = sc.make_body(solid_j, nw)
    sharding = jax.tree_util.tree_map(lambda x: x.sharding, state)

    @partial(jax.jit, out_shardings=sharding, donate_argnums=0)
    def run_block(s):
        return jax.lax.scan(lambda a, _: (body(a), None), s, None, length=sc.block)[0]

    t0 = time.perf_counter()
    snaps, metrics = [], []
    for _ in range(sc.nblocks):
        state = run_block(state)
        snaps.append(phase_field(state))
        metrics.append(sc.metric(state))
    dt = time.perf_counter() - t0

    RUNS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RUNS / f"{sc.name}_g{ndev}.npz", snaps=np.stack(snaps))
    steps = [(k + 1) * sc.block for k in range(sc.nblocks)]
    with (RUNS / f"{sc.name}_g{ndev}_metrics.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", sc.metric_name])
        w.writerows(zip(steps, metrics))
    print(f"  {sc.name}: {ndev} device(s), axis={sc.axis}, {sc.nblocks * sc.block} steps in {dt:.1f}s, "
          f"final {sc.metric_name}={metrics[-1]:.6g}")
    return steps


def run(args):
    devs = jax.devices(args.platform or None)
    print(f"devices: {[str(d) for d in devs]}")
    parity_rows = []
    for name in args.scenarios:
        sc = SCENARIOS[name](smoke=args.smoke)
        ext = sc.solid.shape[multigpu.AXES[sc.axis]]
        counts = [g for g in args.gpus if ext % g == 0 and g <= len(devs)]
        if counts != list(args.gpus):
            print(f"  {name}: using counts {counts} ({sc.axis} extent {ext})")
        steps = None
        for g in counts:
            steps = run_case(sc, g, args.platform)
        base = np.load(RUNS / f"{name}_g{counts[0]}.npz")["snaps"]
        base_metrics = _load_metrics(name, counts[0], sc.metric_name)
        for g in counts[1:]:
            snaps = np.load(RUNS / f"{name}_g{g}.npz")["snaps"]
            metrics = _load_metrics(name, g, sc.metric_name)
            for k, step in enumerate(steps):
                parity_rows.append({"scenario": name, "axis": sc.axis, "ngpu": g, "step": step,
                                    "max_abs_dphi": f"{np.abs(snaps[k] - base[k]).max():.3e}",
                                    "metric": f"{metrics[k]:.6g}", "metric_ref": f"{base_metrics[k]:.6g}",
                                    "metric_absdiff": f"{abs(metrics[k] - base_metrics[k]):.3e}"})
    OUT.mkdir(exist_ok=True)
    with (OUT / "parity.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PARITY_FIELDS)
        w.writeheader()
        w.writerows(parity_rows)
    worst_phi = max((float(r["max_abs_dphi"]) for r in parity_rows), default=0.0)
    worst_metric = max((float(r["metric_absdiff"]) for r in parity_rows), default=0.0)
    print(f"wrote {OUT / 'parity.csv'}; worst |dphi| = {worst_phi:.3e} (ULP-seeded interface decorrelation, "
          f"informational), worst |dmetric| = {worst_metric:.3e} (the PASS criterion, per-scenario tol)")


def _load_metrics(name, g, metric_name):
    rows = list(csv.DictReader((RUNS / f"{name}_g{g}_metrics.csv").open()))
    return [float(r[metric_name]) for r in rows]


# ----------------------------------------------------------------------- render phase
def _vertical_camera(shape):
    nz, ny, nx = shape
    c = (nx / 2.0, ny / 2.0, nz / 2.0)
    return [(c[0], c[1] - 2.0 * nx, c[2]), c, (1.0, 0.0, 0.0)]


def render(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from bob.utils import viz3d

    counts = list(args.gpus)
    for name in args.scenarios:
        sc = SCENARIOS[name](smoke=args.smoke)
        have = [g for g in counts if (RUNS / f"{name}_g{g}.npz").exists()]
        if len(have) < 2:
            print(f"  {name}: need >= 2 device counts on disk, have {have} — skipped")
            continue
        g1, gN = have[0], have[-1]
        base = np.load(RUNS / f"{name}_g{g1}.npz")["snaps"]
        snaps = np.load(RUNS / f"{name}_g{gN}.npz")["snaps"]
        cam = {"xz": "xz", "vertical": _vertical_camera(sc.solid.shape), "flow": viz3d.flow_camera(sc.solid.shape)}[sc.camera]
        rdir, fdir = OUT / "_render" / name, OUT / "frames" / name
        rdir.mkdir(parents=True, exist_ok=True)
        fdir.mkdir(parents=True, exist_ok=True)
        for stale in list(rdir.glob("*.png")) + list(fdir.glob("*.png")):
            stale.unlink()

        mid = sc.solid.shape[sc.slice_axis] // 2
        frame_paths = []
        for k in range(base.shape[0]):
            pa, pb = rdir / f"g{g1}_k{k:03d}.png", rdir / f"g{gN}_k{k:03d}.png"
            viz3d.phase_frame(pa, base[k], sc.solid, camera=cam, smooth=False, shade=True, zoom=1.4, rock_opacity=0.12)
            viz3d.phase_frame(pb, snaps[k], sc.solid, camera=cam, smooth=False, shade=True, zoom=1.4, rock_opacity=0.12)
            diff = np.abs(np.take(snaps[k] - base[k], mid, axis=sc.slice_axis))
            fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
            for ax, img, title in zip(axes[:2], (pa, pb), (f"{g1} device(s)", f"{gN} devices (axis={sc.axis})")):
                ax.imshow(plt.imread(img))
                ax.set_axis_off()
                ax.set_title(title, fontsize=11)
            im = axes[2].imshow(diff, origin="lower", cmap="magma", vmin=0.0, vmax=max(diff.max(), 1e-15))
            axes[2].set_title(f"|Δφ| midplane   max={np.abs(snaps[k] - base[k]).max():.1e}", fontsize=11)
            fig.colorbar(im, ax=axes[2], fraction=0.046)
            step = (k + 1) * sc.block
            fig.suptitle(f"{name} — single vs sharded (step {step})", fontsize=13)
            fig.tight_layout()
            fp = fdir / f"frame_{k:03d}.png"
            fig.savefig(fp, dpi=args.frame_dpi)
            plt.close(fig)
            frame_paths.append(fp)
        viz.write_gif(frame_paths, OUT / f"{name}_parity.gif", args.fps)
        import shutil

        shutil.copy(frame_paths[-1], OUT / f"{name}_final.png")

        if viz.plots_enabled():
            # metric overlay: every device count must trace the same curve
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for g in have:
                rows = list(csv.DictReader((RUNS / f"{name}_g{g}_metrics.csv").open()))
                ax.plot([int(r["step"]) for r in rows], [float(r[sc.metric_name]) for r in rows],
                        marker="o", ms=3.5, lw=1.0, label=f"{g} device(s)")
            ax.set_xlabel("timestep")
            ax.set_ylabel(sc.metric_name)
            ax.set_title(f"{name}: metric vs device count (curves must coincide)")
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(OUT / f"{name}_metric.png", dpi=200)
            plt.close(fig)

            # phi-drift growth curve (log y): ULP-seeded interface decorrelation, not a slab error
            prows = [r for r in csv.DictReader((OUT / "parity.csv").open()) if r["scenario"] == name]
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for g in sorted({r["ngpu"] for r in prows}, key=int):
                sub = [r for r in prows if r["ngpu"] == g]
                ax.semilogy([int(r["step"]) for r in sub], [max(float(r["max_abs_dphi"]), 1e-17) for r in sub],
                            marker="o", ms=3.5, lw=1.0, label=f"{g} devices vs 1")
            ax.set_xlabel("timestep")
            ax.set_ylabel("max |Δφ| (whole domain)")
            ax.set_title(f"{name}: drift from float64 roundoff (interface decorrelation)")
            ax.grid(alpha=0.3, which="both")
            ax.legend()
            fig.tight_layout()
            fig.savefig(OUT / f"{name}_drift.png", dpi=200)
            plt.close(fig)
        print(f"  {name}: {OUT / f'{name}_parity.gif'}")

    _write_report(args)


def _write_report(args):
    rows = list(csv.DictReader((OUT / "parity.csv").open())) if (OUT / "parity.csv").exists() else []
    lines = [
        "# Multi-GPU parity — sharded runs vs the single-device physics",
        "",
        "Each scenario runs the same jitted solver at 1/2/N devices, sharded along an",
        "adversarial axis (walls / Zou-He BC planes / pressure slabs on the sliced",
        "axis). PASS = the physics observable agrees with the single-device run",
        "within the scenario tolerance; max |Δφ| (whole domain) is reported alongside.",
        "In the 8x H100 reference run every scenario is bit-identical (|Δφ| = 0) across",
        "1/2/8 devices. Under CPU emulation the partitioned compilation differs by ~1 ULP/step",
        "(different fusion order) and the moving contact line amplifies that seed",
        "(localized at the interface, not at slab boundaries — the decorrelation any",
        "MPI decomposition shows), which is why the criterion is the observable, with",
        "the strict 1e-12 short-horizon parity pinned by test/test_multigpu.py on",
        "every axis.",
        "",
        "| scenario | shard axis | GPUs | final step | metric (1 GPU) | metric (N GPU) | |Δmetric| | tol | final max |Δφ| | status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name in args.scenarios:
        sc = SCENARIOS[name](smoke=args.smoke)
        grp = [r for r in rows if r["scenario"] == name]
        for g in sorted({r["ngpu"] for r in grp}, key=int):
            sub = [r for r in grp if r["ngpu"] == g]
            worst_m = max(float(r["metric_absdiff"]) for r in sub)
            final = sub[-1]
            ok = "PASS" if worst_m <= sc.metric_tol else "FAIL"
            lines.append(f"| {name} ({sc.metric_name}) | {final['axis']} | {g} | {final['step']} | {final['metric_ref']} "
                         f"| {final['metric']} | {worst_m:.2e} | {sc.metric_tol:g} | {final['max_abs_dphi']} | {ok} |")
    lines += [
        "",
        "Artifacts per scenario: `<name>_parity.gif` (1-device | N-device | |Δφ| midplane),",
        "`<name>_final.png`, `frames/<name>/`. Raw data in `runs/`. With `--plots` also",
        "`<name>_metric.png` (metric curves per device count) and `<name>_drift.png` (log-scale |Δφ| growth).",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'report.md'}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="store_true", help="execute the scenarios (GPU node)")
    p.add_argument("--render", action="store_true", help="render GIFs/frames/report from saved runs")
    p.add_argument("--gpus", type=int, nargs="+", default=[1, 2, 8], help="device counts (first is the reference)")
    p.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    p.add_argument("--smoke", action="store_true", help="tiny step counts (CPU emulation smoke test)")
    p.add_argument("--platform", default="", help="force a JAX platform, e.g. 'cpu'")
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--frame-dpi", type=int, default=130)
    p.add_argument("--plots", action="store_true",
                   help="also write the analysis curves (<name>_metric.png, <name>_drift.png); off by default")
    args = p.parse_args()
    if args.plots:
        viz.enable_plots(True)
    if not args.run and not args.render:
        raise SystemExit("pass --run and/or --render")
    if args.run:
        run(args)
    if args.render:
        render(args)


if __name__ == "__main__":
    main()
