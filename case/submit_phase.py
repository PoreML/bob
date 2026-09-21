"""Launch the dataset campaign as throttled SLURM job arrays.

Reads the frozen draw (campaign_draw.py, the same one behind README.md), takes
every run that has not finished, splits it by GPU class, and submits one job
array per class with a concurrency throttle:

    small-domain runs (384)   --gres=gpu:1  --array=0-N%20  --time 12:00:00  ->  20 GPUs
    large-domain runs  (48)   --gres=gpu:4  --array=0-N%3   --time 48:00:00  ->  12 GPUs
                                                                     total     32 GPUs

The ``%N`` on ``--array`` is SLURM's concurrency throttle, not a task count: all
N tasks are queued at once, but at most %N of them run simultaneously. That is
what fixes the GPU footprint — %20 tasks x 1 GPU + %3 tasks x 4 GPUs = 32 GPUs
held no matter how many runs remain.

Arrays rather than dependency chains: an array task becomes eligible the moment
a slot frees, so the campaign refills itself and holds a fixed footprint. A
``--dependency=afterany`` chain makes each job wait for a named predecessor,
cannot backfill into capacity that opens early, and stalls behind one slow run.

Run from the repository root:

    .venv/bin/python case/submit_phase.py                    # submit everything unfinished
    .venv/bin/python case/submit_phase.py --dry-run          # show what would go
    .venv/bin/python case/submit_phase.py --case drainage    # one case only
    .venv/bin/python case/submit_phase.py --small-slots 12 --large-slots 1   # smaller footprint
    .venv/bin/python case/submit_phase.py --phase extra_beta_1g --tag _x1 --small-slots 8
                        # the expansion phase alone, while the base arrays are still queued

The schedule covers both frozen draws (campaign_draw.draw_all): the 432-run
base phase and the 96 x 1-GPU extra_beta_1g expansion. A bare run submits
everything unfinished across both; --phase narrows it — needed while another
phase's arrays are live, because an unfinished-but-queued run would otherwise be
double-submitted.

Re-running is the resume path: finished runs are skipped, so after a walltime
expiry (the array script asks for a short --time on purpose, because the
drivers checkpoint) just submit again and the unfinished tasks continue.

Every submission writes its own manifest, _manifests/beta<tag>_<stamp>_<n>gpu.tsv,
and nothing here ever writes to a path that already exists. run_array.slurm reads
the manifest when each task starts, not at submit time, so a queued array is bound
to a file, not to a list of runs: rewriting that file would make every task that
has not started yet read a different line. Manifests are therefore immutable once
written. Two rules enforce this, both tested in test/test_submit_phase.py: a dry
run writes nothing, and a manifest is written once and never touched again.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

CASE = Path(__file__).resolve().parent

# campaign_draw is loaded by file path under a unique module name: case/ is a
# folder of scripts, not a package, so the import must not depend on sys.path.
_spec = importlib.util.spec_from_file_location("case_campaign_draw", CASE / "campaign_draw.py")
campaign_draw = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = campaign_draw
_spec.loader.exec_module(campaign_draw)
draw = campaign_draw.draw
draw_all = campaign_draw.draw_all
PHASES = ("beta", campaign_draw.EXTRA_PHASE)

MANIFESTS = CASE / "_manifests"
LOGS = CASE / "_logs"
SMALL_SLOTS = 20  # concurrent 1-GPU runs -> 20 GPUs
LARGE_SLOTS = 3  # concurrent 4-GPU runs -> 12 GPUs; 20 + 12 = 32 GPUs held
# Walltime is set per GPU class: 1-GPU runs finish inside one 12 h segment, while 4-GPU
# (large-domain) runs can take longer than a day and would otherwise need several
# resubmissions each. Short requests backfill better — the scheduler can only start a
# low-priority job early if it fits a gap without delaying higher-priority work — so the
# bulk of the campaign keeps the short one, and only the 48 large runs (3 concurrent) ask
# for the long one. Measured run times are listed in README.md.
TIME_SMALL = "12:00:00"
TIME_LARGE = "48:00:00"


def finished(run):
    """True if this run already ended (any finish_type) — those are skipped."""
    meta = run["dir"] / run["name"] / "run_meta.json"
    if not meta.exists():  # trapping records per stage; the flood is the deliverable
        meta = run["dir"] / run["name"] / "run_meta_flood.json"
    if not meta.exists():
        return False
    try:
        return bool(json.loads(meta.read_text()).get("extra", {}).get("finish_type"))
    except Exception:
        return False


def todo_runs(phases, *, cases=None, only=None, phase=None, fresh=False):
    """Flatten {phase: campaign} into the (case, run) submit list, filters applied.

    ``phase`` limits to those phases (e.g. ["extra_beta_1g"] to launch the
    expansion while the base arrays are still queued); default is every phase —
    the resume-everything semantics of a bare re-run.
    """
    todo = []
    for ph, campaign in phases.items():
        if phase and ph not in phase:
            continue
        for case, fams in campaign.items():
            if cases and case not in cases:
                continue
            for doms in fams.values():
                for runs in doms.values():
                    for r in runs:
                        if only and r["name"] not in only:
                            continue
                        if fresh or not finished(r):
                            todo.append((case, r))
    return todo


def manifest_path(tag, stamp, gpus, manifests=MANIFESTS):
    """One manifest per submission, never reused: run_array.slurm reads it when each task
    starts, so a queued array must be the only writer its file ever sees. The stamp makes a
    resubmit (or a dry run) physically unable to touch an earlier array's file."""
    return manifests / f"beta{tag}_{stamp}_{gpus}gpu.tsv"


def manifest_text(batch):
    """TSV rows in run_array.slurm's column order: case, name, gpus, out, geometry, M, theta."""
    return "".join(
        "\t".join(
            [c, r["name"], str(r["gpus"]), str(r["dir"] / r["name"]), str(r["src"]), f"{r['m']:g}", str(r["theta"])]
        )
        + "\n"
        for c, r in batch
    )


class Plan(NamedTuple):
    gpus: int
    slots: int
    walltime: str
    batch: list
    tsv: Path
    cmd: list


def plan(todo, *, tag, stamp, small_slots, large_slots, time_small, time_large, qos, manifests=MANIFESTS, logs=LOGS):
    """Split the todo list by GPU class into one job array each. Pure: nothing is written here."""
    plans = []
    for gpus, slots, walltime in ((1, small_slots, time_small), (4, large_slots, time_large)):
        batch = [(c, r) for c, r in todo if r["gpus"] == gpus]
        if not batch:
            continue
        tsv = manifest_path(tag, stamp, gpus, manifests)
        cmd = [
            "sbatch",
            f"--array=0-{len(batch) - 1}%{slots}",
            f"--gres=gpu:{gpus}",
            f"--qos={qos}",
            f"--time={walltime}",
            f"--job-name=beta{tag}_{gpus}g",
            f"--output={logs}/beta{tag}_{gpus}gpu_%A_%a.log",
            str(CASE / "run_array.slurm"),
        ]
        plans.append(Plan(gpus, slots, walltime, batch, tsv, cmd))
    return plans


def launch(plans, *, dry_run, run=subprocess.run):
    """Print each array; unless dry_run, write its manifest and submit it.

    The manifest is written only on the real path and only to its own stamped file, so a
    dry run cannot change what a queued array will read."""
    for p in plans:
        print(
            f"\n{len(p.batch):3d} runs x {p.gpus} GPU  ->  {p.slots} concurrent = {p.slots * p.gpus} GPUs held, "
            f"walltime {p.walltime}"
        )
        print(f"    MANIFEST={p.tsv} {' '.join(p.cmd)}")
        for c, r in p.batch[:3]:
            print(f"      e.g. {c}/{r['name']}")
        if len(p.batch) > 3:
            print(f"      ... and {len(p.batch) - 3} more")
        if dry_run:
            print("    dry run: manifest not written, nothing submitted")
            continue
        p.tsv.parent.mkdir(exist_ok=True)
        p.tsv.write_text(manifest_text(p.batch))
        res = run(p.cmd, cwd=CASE.parent, env={**os.environ, "MANIFEST": str(p.tsv)}, capture_output=True, text=True, check=False)
        out = (res.stdout + res.stderr).strip()
        if res.returncode:
            sys.exit(f"sbatch failed: {out}")
        print(f"    -> {out}  (manifest {p.tsv.name})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", action="append", help="limit to these cases (repeatable)")
    ap.add_argument("--only", action="append", help="limit to these run names (repeatable)")
    ap.add_argument(
        "--phase",
        action="append",
        choices=list(PHASES),
        help="limit to these phases (repeatable). Default: all — use --phase extra_beta_1g to "
        "launch the expansion while the base arrays are still queued, or the base runs would be "
        "double-submitted.",
    )
    ap.add_argument(
        "--tag",
        default="",
        help="optional suffix for the job name, log files and manifest (e.g. _b for a second "
        "batch, so it reads apart from the original array in squeue). Manifests are stamped "
        "per submission, so a tag is a label, not a safety device.",
    )
    ap.add_argument("--small-slots", type=int, default=SMALL_SLOTS, help="concurrent 1-GPU tasks")
    ap.add_argument("--large-slots", type=int, default=LARGE_SLOTS, help="concurrent 4-GPU tasks")
    ap.add_argument("--time-small", default=TIME_SMALL, help="walltime for 1-GPU tasks")
    ap.add_argument("--time-large", default=TIME_LARGE, help="walltime for 4-GPU tasks")
    ap.add_argument("--qos", default="normal", help="SLURM QOS (site-specific); a low-priority QOS keeps the campaign out of other users' way")
    ap.add_argument("--fresh", action="store_true", help="include runs that already finished")
    ap.add_argument("--dry-run", action="store_true", help="print the sbatch lines; write nothing, submit nothing")
    args = ap.parse_args()

    todo = todo_runs(draw_all(), cases=args.case, only=args.only, phase=args.phase, fresh=args.fresh)
    if not todo:
        print("nothing to submit (all finished? try --fresh)")
        return

    plans = plan(
        todo,
        tag=args.tag,
        stamp=datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
        small_slots=args.small_slots,
        large_slots=args.large_slots,
        time_small=args.time_small,
        time_large=args.time_large,
        qos=args.qos,
    )
    if not args.dry_run:
        LOGS.mkdir(exist_ok=True)
    launch(plans, dry_run=args.dry_run)

    held = sum(min(p.slots, len(p.batch)) * p.gpus for p in plans)
    parts = " + ".join(f"{min(p.slots, len(p.batch))}x{p.gpus}" for p in plans)
    print(f"\nfootprint while running: <= {held} GPUs ({parts})")
    if not args.dry_run:
        print("re-run this command after a walltime expiry: finished runs are skipped, the rest resume")


if __name__ == "__main__":
    main()
