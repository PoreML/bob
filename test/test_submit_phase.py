"""Tests for case/submit_phase.py — the manifest a queued job array reads must be immutable.

A queued job array reads its manifest at start time (run_array.slurm reads one line
when each task starts), so manifests must be immutable once written: rewriting one
while an array is queued would repoint every task that has not started yet. A dry
run therefore writes nothing, and every submission writes its own stamped file.
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "case"))
import submit_phase


class FakeSbatch:
    """Injectable `subprocess.run`: records every call, submits nothing."""

    def __init__(self):
        self.calls = []

    def __call__(self, cmd, **kw):
        self.calls.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, 0, stdout=f"Submitted batch job {1000 + len(self.calls)}\n", stderr="")


def run(name, gpus, tmp_path):
    return {
        "name": name,
        "gpus": gpus,
        "dir": tmp_path / "runs",
        "src": tmp_path / f"{name}.npy",
        "m": 0.1,
        "theta": 120,
    }


def plans(tmp_path, stamp="20260821-101500", tag=""):
    todo = [("drainage", run(f"small_{i}", 1, tmp_path)) for i in range(3)] + [("trapping", run("big_0", 4, tmp_path))]
    return submit_phase.plan(
        todo,
        tag=tag,
        stamp=stamp,
        small_slots=20,
        large_slots=3,
        time_small="12:00:00",
        time_large="48:00:00",
        qos="normal",
        manifests=tmp_path / "_manifests",
        logs=tmp_path / "_logs",
    )


def test_dry_run_writes_nothing_and_submits_nothing(tmp_path):
    manifests = tmp_path / "_manifests"
    manifests.mkdir()
    live = manifests / "beta_20260818-230310_1gpu.tsv"  # an array is still reading this one
    live.write_text("drainage\tqueued_run\t1\t/out\t/geom\t0.1\t120\n")
    sbatch = FakeSbatch()

    submit_phase.launch(plans(tmp_path), dry_run=True, run=sbatch)

    assert sorted(manifests.iterdir()) == [live]
    assert live.read_text() == "drainage\tqueued_run\t1\t/out\t/geom\t0.1\t120\n"
    assert sbatch.calls == []


def test_submission_writes_a_stamped_manifest_and_hands_it_to_sbatch(tmp_path):
    sbatch = FakeSbatch()

    submit_phase.launch(plans(tmp_path, stamp="20260821-101500", tag="_b"), dry_run=False, run=sbatch)

    small = tmp_path / "_manifests" / "beta_b_20260821-101500_1gpu.tsv"
    large = tmp_path / "_manifests" / "beta_b_20260821-101500_4gpu.tsv"
    assert small.exists() and large.exists()
    rows = [line.split("\t") for line in small.read_text().splitlines()]
    assert [r[1] for r in rows] == ["small_0", "small_1", "small_2"]
    assert all(len(r) == 7 for r in rows)
    assert [kw["env"]["MANIFEST"] for _, kw in sbatch.calls] == [str(small), str(large)]
    small_cmd, large_cmd = (cmd for cmd, _ in sbatch.calls)
    assert "--array=0-2%20" in small_cmd and "--gres=gpu:1" in small_cmd and "--time=12:00:00" in small_cmd
    assert "--array=0-0%3" in large_cmd and "--gres=gpu:4" in large_cmd and "--time=48:00:00" in large_cmd
    assert "--job-name=beta_b_1g" in small_cmd


def test_resubmitting_the_same_phase_never_overwrites_an_earlier_manifest(tmp_path):
    sbatch = FakeSbatch()
    submit_phase.launch(plans(tmp_path, stamp="20260818-230310"), dry_run=False, run=sbatch)
    first = tmp_path / "_manifests" / "beta_20260818-230310_1gpu.tsv"
    before = first.read_text()

    submit_phase.launch(plans(tmp_path, stamp="20260821-101500"), dry_run=False, run=sbatch)

    assert first.read_text() == before
    assert len(list((tmp_path / "_manifests").glob("beta_*_1gpu.tsv"))) == 2
