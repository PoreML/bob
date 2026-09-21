"""Batch helper for the flip-chip solder-ball underfill case (32 x 1 GPU).

A frozen seeded draw, geometry staging from the flipchip geometry dataset, and
a status sweep. `underfill_demo.py` stays the pure single-run driver:

    .venv/bin/python case/underfill/campaign.py prepare      # stage geometries into run folders
    .venv/bin/python case/underfill/campaign.py status       # queue + per-run progress
    .venv/bin/python case/underfill/campaign.py cancel       # scancel this batch's jobs
    .venv/bin/python case/underfill/campaign.py verify-draw  # frozen list == seeded draw?

The draw (32 runs, 2 rounds): each round is one full factorial of viscosity
ratio M in {5, 10, 20, 30} x contact angle theta in {30, 40, 50, 60} deg
(every combination twice overall), paired with geometry samples drawn without
replacement across rounds — sample indices 0-31 of the geometry dataset's
`data/flipchip/samples/` volumes are each used exactly once. The draw (`random.Random(20260808)`) is frozen in
ASSIGNMENTS — rerunning it must never reshuffle a batch with data on disk
(`verify-draw` checks).

This draw is not the one behind the published underfill runs, and `submit` is
disabled; see README.md ("Published runs") for the run table of the dataset.

Run folders are self-describing:

    case/underfill/runs/fc<sample>_M<M>_th<theta>/
    ├── <name>.npy / <name>_geometry.json     # staged dataset volume (sha256-verified)
    ├── uf_<name>.h5/.xdmf                    # phi, rho, p, umag, u every 1000 steps
    ├── frames/ + uf_<name>.gif               # top-view melt front + void overlay
    ├── frames3d/ + uf3d_<name>.gif           # PyVista 3D melt front
    └── metrics.csv, fill.svg, run_meta.json, report.md, ckpt_step*.npz
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

CASE = Path(__file__).resolve().parent
REPO = CASE.parent.parent
RUNS = CASE / "runs"
SLURM_SCRIPT = CASE / "run_underfill.slurm"
# geometry dataset tree: $GEOMETRY_DATA, else the sibling checkout <repo>/../geometry/data.
DATASET = Path(os.environ.get("GEOMETRY_DATA") or REPO.parent / "geometry" / "data") / "flipchip" / "samples"
MS = (5, 10, 20, 30)
THETAS = (30, 40, 50, 60)
N_DATASET = 32
SEED = 20260808


@dataclass(frozen=True)
class Run:
    sample: int  # flipchip dataset sample index
    m: int  # viscosity ratio M = nu_red/nu_blue
    theta: int  # encapsulant contact angle (deg, all walls)
    idx: int  # draw index

    @property
    def name(self) -> str:
        return f"fc{self.sample:04d}_M{self.m:02d}_th{self.theta:02d}"

    @property
    def job(self) -> str:
        return f"uf_{self.name}"

    @property
    def run_dir(self) -> Path:
        return RUNS / self.name

    @property
    def structure(self) -> Path:
        return self.run_dir / f"{self.name}.npy"


def draw_assignments(rounds: int = 2) -> list[Run]:
    """The seeded draw that produced ASSIGNMENTS (kept for provenance/extension).

    Each round is one full M x theta factorial (16 combos, freshly shuffled) on
    geometries sampled without replacement across all rounds — round 2 extends
    the same RNG stream, so the first 16 are identical to a one-round draw and
    rounds 1+2 together use dataset samples 0-31 exactly once each."""
    rng = random.Random(SEED)
    out: list[Run] = []
    remaining = list(range(N_DATASET))
    for _ in range(rounds):
        combos = [(m, th) for m in MS for th in THETAS]  # full factorial, 16 combos
        rng.shuffle(combos)
        samples = rng.sample(remaining, len(combos))
        remaining = [s for s in remaining if s not in samples]
        out += [Run(s, m, th, len(out) + j) for j, ((m, th), s) in enumerate(zip(combos, samples))]
    return out


# Frozen — equals draw_assignments(); verified by `campaign.py verify-draw`.
ASSIGNMENTS = [
    Run(5, 20, 60, 0),
    Run(26, 10, 50, 1),
    Run(29, 10, 30, 2),
    Run(15, 20, 30, 3),
    Run(11, 30, 60, 4),
    Run(1, 10, 60, 5),
    Run(21, 30, 40, 6),
    Run(13, 5, 60, 7),
    Run(23, 5, 30, 8),
    Run(14, 5, 50, 9),
    Run(12, 20, 40, 10),
    Run(20, 20, 50, 11),
    Run(9, 30, 50, 12),
    Run(4, 10, 40, 13),
    Run(24, 30, 30, 14),
    Run(3, 5, 40, 15),
    # round 2: second factorial pass, remaining samples
    Run(28, 10, 50, 16),
    Run(10, 30, 50, 17),
    Run(6, 20, 40, 18),
    Run(19, 20, 60, 19),
    Run(2, 5, 50, 20),
    Run(31, 30, 30, 21),
    Run(8, 10, 40, 22),
    Run(30, 10, 30, 23),
    Run(25, 30, 40, 24),
    Run(22, 5, 30, 25),
    Run(0, 10, 60, 26),
    Run(27, 20, 50, 27),
    Run(16, 20, 30, 28),
    Run(17, 30, 60, 29),
    Run(7, 5, 40, 30),
    Run(18, 5, 60, 31),
]


def cmd_prepare(_args):
    """Stage each assignment's dataset geometry into its run folder (idempotent;
    the staged .npy is sha256-verified against the dataset JSON)."""
    for run in ASSIGNMENTS:
        stem = f"flipchip_{run.sample:04d}"
        info = json.loads((DATASET / f"{stem}.json").read_text())
        run.run_dir.mkdir(parents=True, exist_ok=True)
        if not run.structure.exists():
            shutil.copyfile(DATASET / f"{stem}.npy", run.structure)
            shutil.copyfile(DATASET / f"{stem}.json", run.run_dir / f"{run.name}_geometry.json")
        sha = hashlib.sha256(__import__("numpy").load(run.structure).tobytes()).hexdigest()
        if sha != info["sha256"]:
            sys.exit(f"{run.name}: staged sha {sha[:12]} != dataset {info['sha256'][:12]}")
        print(
            f"{run.name:22s} D {info['D']:2d}  gap {info['gap']:2d}  pitch {info['pitch_realized']:.2f}D  "
            f"{info['n_balls']:3d} balls  M {run.m:2d}  theta {run.theta:2d}  shape {'x'.join(map(str, info['shape']))}"
        )


def finish_type(run: Run) -> str | None:
    meta_path = run.run_dir / "run_meta.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text()).get("extra", {}).get("finish_type")


def queued_jobs() -> dict[str, tuple[str, str]]:
    """{job_name: (job_id, state)} for this user's queue."""
    user = os.environ.get("USER") or getpass.getuser()
    out = subprocess.run(["squeue", "-u", user, "-h", "-o", "%i %j %T"], capture_output=True, text=True, check=True).stdout
    jobs = {}
    for line in out.strip().splitlines():
        jid, jname, state = line.split(None, 2)
        jobs[jname] = (jid, state)
    return jobs


def last_fill(run: Run) -> float | None:
    csv = run.run_dir / "metrics.csv"
    if not csv.exists():
        return None
    head, *rows = csv.read_text().strip().splitlines()
    if not rows:
        return None
    return float(dict(zip(head.split(","), rows[-1].split(","))).get("fill", 0))


def cmd_submit(args):
    """sbatch every assignment not already queued/finished (32 x 1 GPU, direct).

    With --steps N, runs that hit the previous step cap without filling the
    array (fill < 0.999) are resubmitted to resume toward the raised cap."""
    # Disabled: ASSIGNMENTS is not the draw behind the published underfill runs
    # (README.md, "Published runs"), so a batch submitted from it would create runs
    # outside that table.
    sys.exit(
        "case/underfill/campaign.py submit is disabled: ASSIGNMENTS is not the draw behind the\n"
        "published underfill runs (see case/underfill/README.md). Submit single runs with\n"
        "case/underfill/run_underfill.slurm; prepare/status/cancel/verify-draw here still work."
    )
    queued = queued_jobs()
    for run in ASSIGNMENTS:
        if run.job in queued:
            print(f"skip {run.job}: already queued as {queued[run.job][0]} ({queued[run.job][1]})")
            continue
        ft = finish_type(run)
        extend = args.steps and ft == "step_cap" and (last_fill(run) or 0.0) < 0.999
        if ft and not args.fresh and not extend:
            print(f"skip {run.job}: finished ({ft}, fill {last_fill(run)}) — --fresh to redo, --steps to extend")
            continue
        if not run.structure.exists():
            sys.exit(f"missing geometry {run.structure} — run `campaign.py prepare` first")
        cmd = [
            "sbatch",
            f"--job-name={run.job}",
            f"--output={run.run_dir}/slurm-%j.log",
            f"--error={run.run_dir}/slurm-%j.log",
        ]
        if args.time_limit:
            cmd.append(f"--time={args.time_limit}")
        cmd.append(str(SLURM_SCRIPT))
        env = {
            "NAME": run.name,
            "SAMPLE": str(run.sample),
            "M": str(run.m),
            "THETA": str(run.theta),
        }
        if args.steps:
            env["STEPS"] = str(args.steps)
        if args.fresh:
            env["FRESH"] = "1"
        r = subprocess.run(cmd, cwd=REPO, env={**os.environ, **env}, capture_output=True, text=True, check=False)
        out = (r.stdout + r.stderr).strip()
        if r.returncode:
            print(f"{run.job}: {out}")
            sys.exit(r.returncode)
        print(f"{run.job}: job {out.split()[-1]}")


def cmd_cancel(_args):
    queued = queued_jobs()
    ids = [jid for run in ASSIGNMENTS for jname, (jid, _) in queued.items() if jname == run.job]
    if not ids:
        print("no campaign jobs in the queue")
        return
    subprocess.run(["scancel", *ids], check=True)
    print(f"cancelled {len(ids)} jobs: {' '.join(ids)}")


def cmd_status(_args):
    """Queue state + last metrics row per run (step, fill, voids)."""
    queued = queued_jobs()
    for run in ASSIGNMENTS:
        jid, state = queued.get(run.job, ("-", finish_type(run) or "not queued"))
        line = f"{run.name:22s} [{jid:>8s} {state:12s}]"
        csv = run.run_dir / "metrics.csv"
        if csv.exists():
            head, *rows = csv.read_text().strip().splitlines()
            if rows:
                cols = dict(zip(head.split(","), rows[-1].split(",")))
                line += f" step {cols['step']:>8s}  fill {float(cols.get('fill', 0)):.3f}  voids {cols.get('voids', '?')}"
        meta_path = run.run_dir / "run_meta.json"
        if meta_path.exists():
            pr = json.loads(meta_path.read_text()).get("progress", {})
            if pr.get("steps_per_s"):
                line += f"  {pr['steps_per_s']:.0f} steps/s"
            if pr.get("eta"):
                line += f"  eta {pr['eta']}"
        print(line)


def cmd_clean(args):
    """Remove run folders under runs/ that are not current assignments
    (--all additionally wipes the current ones for a fresh campaign)."""
    keep = {run.run_dir for run in ASSIGNMENTS}
    victims = [d for d in sorted(RUNS.iterdir()) if d.is_dir() and d not in keep] if RUNS.exists() else []
    if args.all:
        victims += [d for d in sorted(keep) if d.exists()]
    if not victims:
        print("nothing to clean")
        return
    for d in victims:
        print(f"rm -rf {d}")
        if not args.dry_run:
            shutil.rmtree(d)


def cmd_verify_draw(_args):
    drawn = draw_assignments()
    ok = drawn == ASSIGNMENTS
    print("ASSIGNMENTS == draw_assignments():", ok)
    if not ok:
        for a, b in zip(ASSIGNMENTS, drawn):
            if a != b:
                print(f"  frozen {a}  !=  drawn {b}")
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare", help="stage dataset geometries into run folders")
    ps = sub.add_parser("submit", help="sbatch the batch")
    ps.add_argument("--fresh", action="store_true", help="resubmit finished runs from scratch")
    ps.add_argument("--steps", type=int, default=None, help="raise the step cap; resumes unfilled step_cap runs")
    ps.add_argument("--time-limit", type=str, default=None, help="override the sbatch --time limit")
    sub.add_parser("status", help="queue + per-run progress")
    sub.add_parser("cancel", help="scancel the campaign's jobs")
    pc = sub.add_parser("clean", help="remove non-assignment run folders")
    pc.add_argument("--all", action="store_true")
    pc.add_argument("--dry-run", action="store_true")
    sub.add_parser("verify-draw", help="frozen ASSIGNMENTS == seeded draw?")
    args = p.parse_args()
    {
        "prepare": cmd_prepare,
        "submit": cmd_submit,
        "status": cmd_status,
        "cancel": cmd_cancel,
        "clean": cmd_clean,
        "verify-draw": cmd_verify_draw,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
