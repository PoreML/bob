"""Run-progress monitor for porous-media drainage simulations.

Long drainage runs can be interrupted (out-of-memory kills, filesystem faults,
exhausted step budgets), so every production porous run should be observable
while it runs and resumable afterwards. ``RunMonitor`` standardises that
contract for all drainage demos:

- ``log(**row)`` appends one metrics row (e.g. step/saturation/front) and
  immediately rewrites ``metrics.csv`` and the saturation-curve SVG, so the
  on-disk artifacts always reflect the run's progress — call it at the same
  cadence as visualisation.
- ``checkpoint(step, saturation, **arrays)`` saves the solver state to a
  *status-named* file, ``ckpt_step{step:08d}_sat{saturation:.3f}.npz``, so a
  directory listing reads as a run history and the resume point is
  unambiguous. Uncompressed on purpose: multi-GB float64 populations gain
  almost nothing from zlib but cost minutes. Old checkpoints are pruned,
  keeping the newest ``keep``.
- ``latest_checkpoint(out_dir)`` finds the newest checkpoint for ``--resume``;
  ``carry_rows(out_dir, step0)`` reloads the earlier segments' metrics so the
  CSV/curve span the whole multi-segment run.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

from bob.utils import viz

_CKPT = re.compile(r"ckpt_step(\d+)_sat([0-9.]+)\.npz$")


def latest_checkpoint(out_dir):
    """Newest ``ckpt_step*_sat*.npz`` in ``out_dir`` (by step), or None."""
    found = [(int(m.group(1)), p) for p in Path(out_dir).glob("ckpt_step*_sat*.npz") if (m := _CKPT.search(p.name))]
    return max(found)[1] if found else None


def carry_rows(out_dir, step0, fields=("step", "saturation", "front")):
    """Rows from an earlier segment's metrics.csv with step < ``step0``, typed
    int for 'step'/'front' and float otherwise — resume bookkeeping."""
    path = Path(out_dir) / "metrics.csv"
    if not path.exists():
        return []
    with open(path) as fh:
        return [
            {k: (int(float(r[k])) if k in ("step", "front") else float(r[k])) for k in fields if k in r}
            for r in csv.DictReader(fh)
            if int(float(r["step"])) < step0
        ]


class RunMonitor:
    """Incremental metrics + status-named checkpoints for one run directory."""

    def __init__(self, out_dir, title="drainage", curve_field="saturation", rows=None, meta=None):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.title = title
        self.curve_field = curve_field
        self.rows = list(rows or [])  # carried from earlier segments on resume
        self.meta = meta  # optional bob.utils.file.RunMeta: log() drives its heartbeat

    def log(self, **row):
        """Record one row and refresh metrics.csv + the progress curve on disk."""
        self.rows.append(row)
        viz.write_csv(self.out / "metrics.csv", self.rows)
        steps = [r["step"] for r in self.rows]
        vals = [r[self.curve_field] for r in self.rows]
        viz.line_plot(
            self.out / f"{self.curve_field}.svg",
            [(steps, vals, f"red {self.curve_field}", {"color": "#d62728", "lw": 2})],
            self.title,
            "time step",
            f"red {self.curve_field} of pore space",
        )
        if self.meta is not None and "step" in row:
            self.meta.update(row["step"])

    def checkpoint(self, step, saturation, keep=2, **arrays):
        """Save ``arrays`` (+ step) to a status-named npz; prune old checkpoints.

        Uncompressed npz: multi-GB float64 LBM populations barely compress but
        zlib costs minutes at 256^3. ``keep`` newest checkpoints survive, so an
        interrupted write can never leave the run without a good resume point.
        """
        path = self.out / f"ckpt_step{step:08d}_sat{saturation:.3f}.npz"
        np.savez(path, step=step, **arrays)
        kept = sorted(
            (p for p in self.out.glob("ckpt_step*_sat*.npz") if _CKPT.search(p.name)),
            key=lambda p: int(_CKPT.search(p.name).group(1)),
        )
        for stale in kept[:-keep]:
            stale.unlink()
        return path
