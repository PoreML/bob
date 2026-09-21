"""Tests for bob.utils.file.RunMeta — live run-metadata sidecar."""

import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

from bob import color3d
from bob.utils import file as bobfile

T0 = datetime(2026, 7, 27, 12, 0, 0, tzinfo=timezone.utc)


class Clock:
    """Injectable monotonic clock: starts at 0, advance by hand."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class Now:
    """Injectable wall clock pinned to a fixed datetime."""

    def __init__(self):
        self.dt = T0

    def __call__(self):
        return self.dt


def make_meta(tmp_path, **kw):
    kw.setdefault("params", color3d.Params(omega=1.2, sigma=0.05, beta=0.7, theta=140.0))
    kw.setdefault("target_steps", 1000)
    return bobfile.RunMeta(tmp_path / "run_meta.json", clock=kw.pop("clock", Clock()), now=kw.pop("now", Now()), **kw)


def read(meta):
    return json.loads(meta.path.read_text())


def test_sections_and_solver_fields(tmp_path):
    solid = np.zeros((4, 5, 6), bool)
    solid[0] = True  # one z-plane solid: solid_fraction = 1/4
    meta = make_meta(tmp_path, solid=solid, geometry_source="assets/x.npy", extra={"u_in": np.float32(0.01)}, notes="hi")
    doc = read(meta)
    assert set(doc) == {"run", "solver", "geometry", "environment", "progress", "extra"}
    # solver: every Params field lands, tuples become lists, precision/lattice/commit added
    p = color3d.Params(omega=1.2, sigma=0.05, beta=0.7, theta=140.0)
    for k, v in p._asdict().items():
        assert doc["solver"][k] == (list(v) if isinstance(v, tuple) else v)
    assert doc["solver"]["precision"] in ("float32", "float64")
    assert doc["solver"]["lattice"] == "d3q19"  # inferred from 3D mask
    assert "bob_commit" in doc["solver"]
    # run section
    assert doc["run"]["status"] == "running"
    assert doc["run"]["start_time"] == T0.isoformat(timespec="seconds")
    assert doc["run"]["end_time"] is None
    assert doc["run"]["resume_step"] == 0
    assert doc["run"]["notes"] == "hi"
    # environment + extra (numpy scalar must serialize)
    assert doc["environment"]["n_devices"] >= 1
    assert doc["environment"]["backend"] in ("cpu", "gpu", "tpu")
    assert np.isclose(doc["extra"]["u_in"], 0.01)


def test_geometry_stats(tmp_path):
    solid = np.zeros((4, 5, 6), bool)
    solid[0] = True
    meta = make_meta(tmp_path, solid=solid, geometry_source="assets/x.npy")
    g = read(meta)["geometry"]
    assert g["shape"] == [4, 5, 6]
    assert np.isclose(g["solid_fraction"], 0.25) and np.isclose(g["porosity"], 0.75)
    assert g["source"] == "assets/x.npy"
    assert g["sha256"] == hashlib.sha256(np.ascontiguousarray(solid).tobytes()).hexdigest()
    # no mask -> whole section null, lattice unknown
    meta2 = bobfile.RunMeta(tmp_path / "m2.json", clock=Clock(), now=Now())
    assert read(meta2)["geometry"] is None
    assert read(meta2)["solver"]["lattice"] is None


def test_progress_math_and_eta(tmp_path):
    clock, now = Clock(), Now()
    meta = make_meta(tmp_path, clock=clock, now=now)
    pr = read(meta)["progress"]  # written at construction, zero elapsed
    assert pr["step"] == 0 and pr["target_steps"] == 1000
    assert pr["steps_per_s"] is None and pr["eta"] is None
    clock.t = 3600.0  # 1 h wall
    meta.update(500)
    pr = read(meta)["progress"]
    n_dev = read(meta)["environment"]["n_devices"]
    assert pr["step"] == 500 and np.isclose(pr["fraction"], 0.5)
    assert np.isclose(pr["steps_per_s"], 500 / 3600)
    assert np.isclose(pr["elapsed_hours"], 1.0)
    assert np.isclose(pr["device_hours"], 1.0 * n_dev)
    # 500 steps left at 500/h -> ETA one hour from 'now'
    assert pr["eta"] == (now.dt + timedelta(hours=1)).isoformat(timespec="seconds")
    assert pr["updated_at"] == T0.isoformat(timespec="seconds")


def test_finish_and_atomicity(tmp_path):
    clock = Clock()
    meta = make_meta(tmp_path, clock=clock)
    clock.t = 60.0
    meta.finish(1000)
    doc = read(meta)
    assert doc["run"]["status"] == "finished"
    assert doc["run"]["end_time"] == T0.isoformat(timespec="seconds")
    assert doc["progress"]["step"] == 1000 and doc["progress"]["eta"] is None
    assert not list(meta.path.parent.glob("*.tmp"))  # atomic replace leaves no residue


def test_resume_accumulates_hours_and_sets_resume_step(tmp_path):
    clock, now = Clock(), Now()
    meta = make_meta(tmp_path, clock=clock, now=now)
    clock.t = 7200.0  # segment 1: 2 h
    meta.update(400)
    n_dev = read(meta)["environment"]["n_devices"]
    # segment 2: fresh RunMeta on the same path (fresh clock), resuming at 400
    clock2 = Clock()
    meta2 = make_meta(tmp_path, clock=clock2, now=Now(), resume_step=400)
    clock2.t = 3600.0  # +1 h
    meta2.update(700)
    pr = read(meta2)["progress"]
    assert np.isclose(pr["elapsed_hours"], 3.0)  # 2 h carried + 1 h live
    assert np.isclose(pr["device_hours"], 3.0 * n_dev)
    assert np.isclose(pr["steps_per_s"], 300 / 3600)  # rate is per-segment
    assert read(meta2)["run"]["resume_step"] == 400


def finished_segment(tmp_path, **finish_extra):
    """Segment 1: a run that completed at step 400 with the given finish extras."""
    clock = Clock()
    meta = make_meta(tmp_path, clock=clock, extra={"steps_pv": 1000})
    clock.t = 7200.0
    meta.finish(400, **finish_extra)
    return meta


def test_fresh_run_has_no_resume_history(tmp_path):
    meta = make_meta(tmp_path)
    doc = read(meta)
    assert "resumes" not in doc["run"] and "resumed_time" not in doc["run"]


def test_resume_preserves_original_start_time_and_sets_running(tmp_path):
    finished_segment(tmp_path, finish_type="dp_cap")
    later = Now()
    later.dt = T0 + timedelta(days=1)
    meta2 = make_meta(tmp_path, now=later, resume_step=400)
    doc = read(meta2)
    assert doc["run"]["start_time"] == T0.isoformat(timespec="seconds")  # original, not the resume's
    assert doc["run"]["resumed_time"] == later.dt.isoformat(timespec="seconds")
    assert doc["run"]["status"] == "running" and doc["run"]["end_time"] is None


def test_resume_records_history_entry_with_prev_finish(tmp_path):
    finished_segment(tmp_path, finish_type="dp_cap", finish_detail="dp 0.0173 at step 400")
    later = Now()
    later.dt = T0 + timedelta(days=1)
    meta2 = make_meta(tmp_path, now=later, resume_step=400)
    (entry,) = read(meta2)["run"]["resumes"]
    assert entry["at"] == later.dt.isoformat(timespec="seconds")
    assert entry["resume_step"] == 400
    assert entry["argv"] == list(sys.argv)
    assert entry["prev_status"] == "finished"
    assert entry["prev_finish_type"] == "dp_cap"
    assert entry["prev_finish_detail"] == "dp 0.0173 at step 400"
    assert entry["prev_step"] == 400
    assert entry["prev_argv"] == list(sys.argv)


def test_resume_merges_extra_without_resurrecting_finish(tmp_path):
    finished_segment(tmp_path, finish_type="bt_cap", breakthrough_step=300)
    meta2 = make_meta(tmp_path, resume_step=400,
                      extra={"protocol": "looser cap", "breakthrough_step": None})
    ex = read(meta2)["extra"]
    assert ex["protocol"] == "looser cap"  # new non-None value wins
    assert ex["breakthrough_step"] == 300  # new None does not clobber carried value
    assert ex["steps_pv"] == 1000  # untouched prior key carries forward
    assert "finish_type" not in ex and "finish_detail" not in ex  # history, not live state


def test_second_resume_appends_to_history(tmp_path):
    finished_segment(tmp_path, finish_type="dp_cap")
    make_meta(tmp_path, resume_step=400).finish(700, finish_type="pv_cap")
    meta3 = make_meta(tmp_path, resume_step=700)
    entries = read(meta3)["run"]["resumes"]
    assert [e["prev_finish_type"] for e in entries] == ["dp_cap", "pv_cap"]
    assert [e["resume_step"] for e in entries] == [400, 700]


def test_resume_with_corrupt_prior_file_starts_fresh(tmp_path):
    (tmp_path / "run_meta.json").write_text("{not json")
    clock = Clock()
    meta = make_meta(tmp_path, clock=clock)
    clock.t = 3600.0
    meta.update(100)
    assert np.isclose(read(meta)["progress"]["elapsed_hours"], 1.0)


def test_static_json_excludes_progress(tmp_path):
    meta = make_meta(tmp_path)
    static = json.loads(meta.static_json())
    assert "progress" not in static and "solver" in static and "run" in static


def tiny_state(nz=4, ny=5, nx=6):
    import jax.numpy as jnp

    from bob import color3d

    base = jnp.asarray(np.linspace(0.02, 0.2, 19 * nz * ny * nx).reshape(19, nz, ny, nx))
    return color3d.State(base, base[::-1] * 0.5)


def test_runmonitor_meta_hook(tmp_path):
    from bob.utils import stats

    clock = Clock()
    meta = make_meta(tmp_path, clock=clock)
    mon = stats.RunMonitor(tmp_path, meta=meta)
    clock.t = 5.0
    mon.log(step=50, saturation=0.1)
    assert read(meta)["progress"]["step"] == 50


def test_fieldwriter_meta_hook_updates_and_mirrors(tmp_path):
    import h5py

    clock = Clock()
    meta = make_meta(tmp_path, clock=clock)
    solid = np.zeros((4, 5, 6), bool)
    with bobfile.FieldWriter(tmp_path / "run.h5", solid, fields=("phi",), meta=meta) as w:
        clock.t = 10.0
        w.append(100, tiny_state())
    assert read(meta)["progress"]["step"] == 100  # append drove the heartbeat
    with h5py.File(tmp_path / "run.h5", "r") as f:
        mirrored = json.loads(f.attrs["run_meta"])
        assert mirrored["solver"]["theta"] == 140.0 and "progress" not in mirrored
