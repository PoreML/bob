"""Tests for the dataset-campaign machinery in case/ (documented in case/README.md).

The draw is a frozen schedule feeding a 432-run GPU campaign plus a 96-run
expansion, so what is asserted here is what the README states: reproducibility,
the run count and GPU split, the throat screen (r_c > 5 floor never waived,
margin >= 1.3 or flagged), the even source split, balanced parameters,
collision-free b99 run names, and the manifest-immutability rules of
submit_phase.py.

case/ is a folder of scripts, not a package, so its modules are loaded by file
path under unique names.
"""

import hashlib
import importlib.util
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

CASE = Path(__file__).resolve().parents[1] / "case"


def _load(name, fname):
    key = f"case_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, CASE / fname)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


cd = _load("campaign_draw", "campaign_draw.py")
sp = _load("submit_phase", "submit_phase.py")

pytestmark = pytest.mark.skipif(not cd.DATA.exists(), reason="geometry dataset not on this machine")


@pytest.fixture(scope="module")
def campaign():
    return cd.draw()


def _runs(campaign):
    return [
        (case, fam, dom, r)
        for case, fams in campaign.items()
        for fam, doms in fams.items()
        for dom, runs in doms.items()
        for r in runs
    ]


def test_draw_is_deterministic(campaign):
    again = cd.draw()
    key = [(r["name"], r["src"], r["flags"]) for *_x, r in _runs(campaign)]
    assert key == [(r["name"], r["src"], r["flags"]) for *_x, r in _runs(again)]


def test_432_runs_384_small_48_large(campaign):
    runs = _runs(campaign)
    assert len(runs) == 432
    by_gpus = Counter(r["gpus"] for *_x, r in runs)
    assert by_gpus == {1: 384, 4: 48}


def test_names_unique_tagged_and_self_describing(campaign):
    names = [r["name"] for *_x, r in _runs(campaign)]
    assert len(names) == len(set(names))
    for case, _fam, dom, r in _runs(campaign):
        assert r["name"].endswith("_b99")
        assert f"_{dom}_" in r["name"]
        # every GDL CT name embeds its dataset: the three gdl_ct* pools share stems
        if case == "GDL" and r["dataset"].startswith("gdl_ct"):
            assert r["name"].startswith(f"gdl_{r['dataset']}_")
        assert str(r["dir"]).startswith(str(CASE))


def test_throat_floor_never_waived_and_margin_flagged(campaign):
    throats = cd.load_throats()
    for case, _fam, dom, r in _runs(campaign):
        if case == "GDL":
            assert r["r_c"] is None
            continue
        rc = throats[r["dataset"], dom, r["stem"]]
        assert rc == r["r_c"] and rc > cd.RC_FLOOR
        margin = rc / cd.r_cap(r["theta"])
        assert abs(margin - r["margin"]) < 1e-9
        if margin < cd.MARGIN_MIN:
            assert "margin" in r["flags"]


def test_geometry_reuse_only_when_flagged_and_never_exact_triple(campaign):
    for fams in campaign.values():
        for doms in fams.values():
            for runs in doms.values():
                triples = [(r["dataset"], r["stem"], r["m"], r["theta"]) for r in runs]
                assert len(triples) == len(set(triples))
                seen = Counter((r["dataset"], r["stem"]) for r in runs)
                for r in runs:
                    if seen[r["dataset"], r["stem"]] > 1:
                        flagged = [
                            x for x in runs
                            if (x["dataset"], x["stem"]) == (r["dataset"], r["stem"]) and "reused" in x["flags"]
                        ]
                        assert flagged, f"{r['name']}: geometry repeated without a reused flag"


def test_even_source_split(campaign):
    for case, spec in cd.CASES.items():
        for fam, datasets in spec["families"].items():
            for dom, (_sub, n, _g, _th) in spec["domains"].items():
                got = Counter(r["dataset"] for r in campaign[case][fam][dom])
                assert got == Counter(cd.quotas(datasets, n))


def test_balanced_params_per_set(campaign):
    for case, spec in cd.CASES.items():
        for fam in spec["families"]:
            for dom, (_sub, n, _g, theta_menu) in spec["domains"].items():
                pairs = [(r["m"], r["theta"]) for r in campaign[case][fam][dom]]
                cells = Counter(pairs)
                per_cell = n // (len(spec["m"]) * len(theta_menu))
                assert set(cells) == {(m, th) for m in spec["m"] for th in theta_menu}
                assert max(cells.values()) - min(cells.values()) <= 1
                assert min(cells.values()) >= per_cell


# ---------------------------------------------------------------- extra_beta_1g
#
# The expansion phase: 16 more 1-GPU runs per (case, family), drawn under the
# same protocol (even source split, throat screen, percolation QC, balanced
# parameters) from a separate seed, with the base draw's per-set state carried
# over — a rock the base already ran is only repeated at a different (M, theta)
# and flagged `reused`.

# sha256 over the base draw's (case, fam, dom, name, src, flags) rows, with src taken
# relative to the data root so the pin holds wherever the dataset is checked out: the
# expansion draw must not move the frozen 432-run schedule by a single pick.
BASE_DIGEST = "b6fcf0f81c3f9567c95582bca05574668d5b60b6674019b1ba6a13287953d7f8"


@pytest.fixture(scope="module")
def extra(campaign):
    return cd.draw_extra(campaign)


def test_base_draw_pinned_bit_for_bit(campaign):
    rows = [(case, fam, dom, r["name"], Path(r["src"]).relative_to(cd.DATA).as_posix(), r["flags"])
            for case, fam, dom, r in _runs(campaign)]
    assert hashlib.sha256(repr(rows).encode()).hexdigest() == BASE_DIGEST


def test_extra_draw_is_deterministic(campaign, extra):
    again = cd.draw_extra(campaign)
    key = [(r["name"], r["src"], r["flags"]) for *_x, r in _runs(extra)]
    assert key == [(r["name"], r["src"], r["flags"]) for *_x, r in _runs(again)]


def test_extra_96_runs_16_per_set_all_single_gpu(extra):
    runs = _runs(extra)
    assert len(runs) == 96
    assert all(r["gpus"] == 1 for *_x, r in runs)
    assert len({(case, fam, dom) for case, fam, dom, _r in runs}) == 6
    for case, fams in extra.items():
        for doms in fams.values():
            for dom, rs in doms.items():
                assert len(rs) == 16
                assert cd.CASES[case]["domains"][dom][2] == 1  # small-domain sets only


def test_extra_every_param_cell_exactly_once_per_set(extra):
    for case, fams in extra.items():
        for doms in fams.values():
            for dom, runs in doms.items():
                theta_menu = cd.CASES[case]["domains"][dom][3]
                cells = Counter((r["m"], r["theta"]) for r in runs)
                assert cells == {(m, th): 1 for m in cd.CASES[case]["m"] for th in theta_menu}


def test_extra_even_source_split(extra):
    for case, fams in extra.items():
        for fam, doms in fams.items():
            datasets = cd.CASES[case]["families"][fam]
            for runs in doms.values():
                assert Counter(r["dataset"] for r in runs) == Counter(cd.quotas(datasets, cd.EXTRA_N))


def test_extra_names_unique_and_no_triple_repeats_across_phases(campaign, extra):
    names = [r["name"] for *_x, r in _runs(campaign)] + [r["name"] for *_x, r in _runs(extra)]
    assert len(names) == len(set(names))
    for case, fams in extra.items():
        for fam, doms in fams.items():
            for dom, runs in doms.items():
                combined = campaign[case][fam][dom] + runs
                triples = [(r["dataset"], r["stem"], r["m"], r["theta"]) for r in combined]
                assert len(triples) == len(set(triples))


def test_extra_throat_floor_never_waived_and_margin_flagged(extra):
    throats = cd.load_throats()
    for case, _fam, dom, r in _runs(extra):
        if case == "GDL":
            assert r["r_c"] is None
            continue
        rc = throats[r["dataset"], dom, r["stem"]]
        assert rc == r["r_c"] and rc > cd.RC_FLOOR
        margin = rc / cd.r_cap(r["theta"])
        assert abs(margin - r["margin"]) < 1e-9
        if margin < cd.MARGIN_MIN:
            assert "margin" in r["flags"]


def test_extra_unflagged_picks_are_rocks_the_base_never_ran(campaign, extra):
    for case, fams in extra.items():
        for fam, doms in fams.items():
            for dom, runs in doms.items():
                base_stems = {(r["dataset"], r["stem"]) for r in campaign[case][fam][dom]}
                for r in runs:
                    if not r["flags"]:
                        assert (r["dataset"], r["stem"]) not in base_stems, r["name"]


def test_extra_gdl_generated_all_reused_and_spread(campaign, extra):
    # the base draw consumed all 64 percolating fiber samples, so every
    # expansion run reuses one — least-used-first spreads over 16 distinct fibers
    runs = extra["GDL"]["generated"]["128x128x64"]
    assert all("reused" in r["flags"] for r in runs)
    base_stems = {r["stem"] for r in campaign["GDL"]["generated"]["128x128x64"]}
    assert {r["stem"] for r in runs} <= base_stems
    assert len({r["stem"] for r in runs}) == 16


def test_draw_all_keys_both_phases(campaign):
    phases = cd.draw_all()
    assert list(phases) == ["beta", "extra_beta_1g"]
    assert len(_runs(phases["beta"])) == 432
    assert len(_runs(phases["extra_beta_1g"])) == 96


def test_tables_render_extra_manifest_section(campaign, extra):
    text = cd.tables(campaign, extra)
    assert "## Manifest — the frozen 432-run draw" in text
    assert "extra_beta_1g" in text
    assert "**96**" in text


# ---------------------------------------------------------------- submit_phase


class FakeSbatch:
    def __init__(self):
        self.calls = []

    def __call__(self, cmd, **kw):
        self.calls.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, 0, stdout=f"Submitted batch job {1000 + len(self.calls)}\n", stderr="")


def _fake_run(name, gpus, tmp_path):
    return {
        "name": name,
        "gpus": gpus,
        "dir": tmp_path / "runs",
        "src": tmp_path / f"{name}.npy",
        "m": 0.1,
        "theta": 120,
    }


def _plans(tmp_path, stamp="20260825-101500", tag=""):
    todo = [("drainage", _fake_run(f"small_{i}_b99", 1, tmp_path)) for i in range(3)] + [
        ("trapping", _fake_run("big_0_b99", 4, tmp_path))
    ]
    return sp.plan(
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


def test_plan_targets_case_dir_and_stamped_manifests(tmp_path):
    plans = _plans(tmp_path)
    assert [p.gpus for p in plans] == [1, 4]
    for p in plans:
        assert p.cmd[-1] == str(CASE / "run_array.slurm")
        assert p.tsv.name.startswith("beta_20260825-101500") and p.tsv.name.endswith(f"{p.gpus}gpu.tsv")
    # geometry column is the src path
    line = sp.manifest_text(plans[0].batch).splitlines()[0].split("\t")
    assert line[4] == str(tmp_path / "small_0_b99.npy")


def test_dry_run_writes_nothing_and_submits_nothing(tmp_path):
    fake = FakeSbatch()
    plans = _plans(tmp_path)
    sp.launch(plans, dry_run=True, run=fake)
    assert fake.calls == []
    assert not (tmp_path / "_manifests").exists()


def test_todo_runs_phase_filter(tmp_path):
    phases = {
        "beta": {"drainage": {"generated": {"128": [_fake_run("base_run_b99", 1, tmp_path)]}}},
        "extra_beta_1g": {"drainage": {"generated": {"128": [_fake_run("extra_run_b99", 1, tmp_path)]}}},
    }
    assert [r["name"] for _c, r in sp.todo_runs(phases)] == ["base_run_b99", "extra_run_b99"]
    assert [r["name"] for _c, r in sp.todo_runs(phases, phase=["extra_beta_1g"])] == ["extra_run_b99"]


def test_todo_runs_case_and_only_filters_survive(tmp_path):
    phases = {
        "beta": {
            "drainage": {"generated": {"128": [_fake_run("d_b99", 1, tmp_path)]}},
            "trapping": {"generated": {"128": [_fake_run("t_b99", 1, tmp_path)]}},
        },
    }
    assert [r["name"] for _c, r in sp.todo_runs(phases, cases=["trapping"])] == ["t_b99"]
    assert [r["name"] for _c, r in sp.todo_runs(phases, only=["d_b99"])] == ["d_b99"]


def test_launch_writes_each_manifest_once_to_its_own_stamped_file(tmp_path):
    fake = FakeSbatch()
    plans = _plans(tmp_path)
    sp.launch(plans, dry_run=False, run=fake)
    written = sorted(p.name for p in (tmp_path / "_manifests").iterdir())
    assert written == ["beta_20260825-101500_1gpu.tsv", "beta_20260825-101500_4gpu.tsv"]
    assert len(fake.calls) == 2
    # a later submission (new stamp) cannot touch these files
    later = _plans(tmp_path, stamp="20260825-111500")
    assert {p.tsv for p in later}.isdisjoint({p.tsv for p in plans})
