"""The frozen, seeded draws behind the dataset campaigns (manifest in case/README.md).

Two seeded draws fix every run: the 432-run base phase (``SEED``) and the
96-run ``extra_beta_1g`` expansion (``EXTRA_SEED``) — 16 more 1-GPU runs per
(case, family) under the identical protocol, drawn strictly after replaying
the base draw so the base schedule cannot move (sha256-pinned in
test/test_campaign.py). Running this script reproduces both manifests exactly
(``random.Random(seed)``, pools sorted before sampling) and refreshes the
per-run Status column by reading each run's ``run_meta.json`` off disk. Only
the part of README.md below the marker line is generated; everything above it
is hand-written documentation.

    .venv/bin/python case/campaign_draw.py            # refresh the README manifest
    .venv/bin/python case/campaign_draw.py --check    # verify the draw is unchanged

Geometries are read from the geometry dataset tree: ``$GEOMETRY_DATA`` if
set, otherwise the sibling checkout ``<repo>/../geometry/data``.

Draw rules:
  * base phase: 432 runs, 384 x 1 GPU + 48 x 4 GPU; the extra_beta_1g
    expansion adds 96 x 1 GPU under the same rules, preferring geometries the
    base never ran and falling back to flagged reuse where a pool runs dry
    (the base draw uses all 64 fiber samples, so the GDL generated expansion
    set is all reuse);
  * every family draws evenly across its sources (22/21/21 into 64 runs,
    3/3/2 into 8 — sorted source order takes the remainder);
  * (M, theta) assigned for balanced coverage — every combination on the menu
    appears the same number of times, +/-1;
  * drainage/trapping geometries are throat-filtered against
    geometry_throats.csv: the pool is cut to r_c > 5 lattice units (the
    narrowest controlling throat is resolved by the diffuse interface — never
    waived), and a rock is eligible at contact angle theta only with dp-cap
    margin r_c / r_cap(theta) >= 1.3. Geometries are assigned
    hardest-theta-first so the theta-150 runs get first pick of the wide
    rocks. Two fallbacks, both flagged in the manifest: when a source has no
    unused eligible rock left, an already-used one is reused at a different
    (M, theta) — run names embed the (geometry, M, theta) triple, so only an
    exact repeat would collide (castlegate/128 has 17 rocks above the floor
    for a 21-run quota) — and when not even a reuse clears the margin, the
    largest remaining margin is taken;
  * GDL slabs are not in the throat table (the scan targets porous cubes) and
    keep percolation QC only;
  * run names carry a ``_b99`` tag (recoloring sharpness beta = 0.99), and
    every GDL CT name embeds its dataset (gdl_ct / gdl_ct_20 / gdl_ct_40 share
    gdl_XXXX stems).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import re
import sys
from collections import Counter
from math import cos, radians
from pathlib import Path

CASE = Path(__file__).resolve().parent
# geometry dataset tree: $GEOMETRY_DATA, else the sibling checkout <repo>/../geometry/data.
DATA = Path(os.environ.get("GEOMETRY_DATA") or CASE.parent.parent / "geometry" / "data")
SEED = 20260825
# extra_beta_1g — the 96-run expansion phase (README section of the same name):
# 16 more 1-GPU runs per (case, family), drawn under the same protocol from a
# separate seed, with the base draw's per-set state carried over. The base
# stream is never touched, so the frozen 432-run schedule cannot move.
EXTRA_SEED = 20260830
EXTRA_N = 16
EXTRA_PHASE = "extra_beta_1g"
THROATS = CASE / "geometry_throats.csv"

# README.md is spliced at this line: hand-written documentation above, generated
# manifest below. The marker must survive every rewrite.
MARKER = "<!-- campaign_draw.py manifest — everything below is generated; do not edit by hand -->"

# Throat screen: r_cap(theta) is the widest throat the drainage dp cap forbids,
# with the cap at 0.5x the porous-plate entry pressure (plate pores r = 2.5,
# theta_plate 150) — the same constants as drainage/drainage_demo.py.
SIGMA = 0.05
DP_STOP = 0.5 * 2.0 * SIGMA * abs(cos(radians(150.0))) / 2.5
RC_FLOOR = 5.0  # lattice units; below this the controlling throat is unresolved
MARGIN_MIN = 1.3


def r_cap(theta):
    """Widest throat the drainage dp cap forbids at this contact angle (voxels)."""
    return 2.0 * SIGMA * abs(cos(radians(float(theta)))) / DP_STOP


# Viscosity ratio menus. M = mu_nwp / mu_wp — the non-wetting phase over the
# wetting one. Densities are equal, so M = nu_red/nu_blue, exactly the --m the
# drivers take (red is the non-wetting fluid in every case; theta 120-150 is
# the red contact angle).
DRAINAGE_M = [0.05, 0.1, 0.2, 1]
GDL_M = [1, 5, 10, 20]
POROUS_THETA = [120, 130, 140, 150]
# The 8-run large-domain sets cannot cover a 4x4 grid; they take the ends of
# the contact-angle range, so 4 M x 2 theta = 8 and every combination is
# exercised exactly once per set.
POROUS_THETA_LARGE = [120, 150]

# case -> run-name prefix, families (label -> dataset dirs, drawn evenly),
#         domains (label -> (subdir, N, GPUs, theta menu)), M menu,
#         throat: whether the geometry_throats.csv screen applies.
CASES = {
    "drainage": {
        "prefix": "drain",
        "families": {"generated": ["blob", "poly", "sphere"], "ct": ["bentheimer", "buffberea", "castlegate"]},
        "domains": {"128": ("128", 64, 1, POROUS_THETA), "256": ("256", 8, 4, POROUS_THETA_LARGE)},
        "m": DRAINAGE_M,
        "throat": True,
    },
    "trapping": {
        "prefix": "trap",
        "families": {"generated": ["blob", "poly", "sphere"], "ct": ["bentheimer", "buffberea", "castlegate"]},
        "domains": {"128": ("128", 64, 1, POROUS_THETA), "256": ("256", 8, 4, POROUS_THETA_LARGE)},
        "m": DRAINAGE_M,
        "throat": True,
    },
    "GDL": {
        "prefix": "gdl",
        "families": {"generated": ["fiber"], "ct": ["gdl_ct", "gdl_ct_20", "gdl_ct_40"]},
        "domains": {
            "128x128x64": ("128x128x64", 64, 1, POROUS_THETA),
            "256x256x128": ("256x256x128", 8, 4, POROUS_THETA_LARGE),
        },
        "m": GDL_M,
        "throat": False,
    },
}

# Termination rules, quoted from each driver (finish_type lands in run_meta.json).
TERMINATION = {
    "drainage": (
        "`pv_cap` 0.75 PV, or `dp_cap` when dp reaches 0.5x the porous-plate entry pressure — beyond that "
        "the plate itself risks draining and it stops being a valid porous-plate experiment"
    ),
    "GDL": (
        "`bt_cap` 1.2x the breakthrough step, or `pv_cap` 0.5 PV, whichever comes first; breakthrough is "
        "recorded as `breakthrough_step` / `breakthrough_sat`"
    ),
    "trapping": (
        "two stages. Stage 1 `equilibrated` when |dS_w| per 1000-step block < 5e-6 held 3 blocks. Stage 2 "
        "`red_immobile` at `--mob-tol 0.003` — the oil body moved less than 0.3% of its voxels in each of 3 "
        "consecutive, non-overlapping 20,000-step windows — or `pv_cap` at 1.2 PV"
    ),
}


def pool(dataset, subdir):
    """Sorted sample stems for one dataset/domain (sorted => the draw is stable)."""
    d = DATA / dataset / subdir
    return sorted(p.stem for p in d.glob("*.npy"))


def load_throats():
    """{(dataset, domain, stem): r_c} from the local throat-scan table."""
    out = {}
    with THROATS.open() as f:
        for row in csv.DictReader(f):
            if row.get("error"):
                continue
            out[row["dataset"], row["domain"], row["name"]] = float(row["r_c"])
    return out


def balanced_params(m_menu, theta_menu, n, rng):
    """n (M, theta) pairs covering the menu as evenly as it divides.

    Every combination appears floor(n/K) times, and the n mod K leftovers go to
    distinct combinations — so no corner of the parameter space is over- or
    under-sampled by more than one run. i.i.d. draws do not give this: 64 runs
    over 16 combinations come out lumpy (some 8x, some 1x), which biases
    anything trained on the result.

    Built one run at a time, always taking a least-used cell of the grid and
    breaking that tie on the least-used M and theta marginals, so the leftovers
    also land evenly on the single-parameter histograms. The greedy can paint
    itself into a corner, so a draw that misses the target is redrawn; the
    retry is part of the seeded sequence, so the result is exactly
    reproducible.
    """
    combos = [(m, th) for m in m_menu for th in theta_menu]
    for _ in range(200):
        cell = dict.fromkeys(combos, 0)
        m_seen = dict.fromkeys(m_menu, 0)
        th_seen = dict.fromkeys(theta_menu, 0)
        picks = []
        for _i in range(n):
            least = min(cell.values())
            free = [c for c in combos if cell[c] == least]
            best = min(m_seen[m] + th_seen[th] for m, th in free)
            m, th = rng.choice([c for c in free if m_seen[c[0]] + th_seen[c[1]] == best])
            picks.append((m, th))
            cell[m, th] += 1
            m_seen[m] += 1
            th_seen[th] += 1
        if _flat(picks, combos, m_menu, theta_menu):
            return picks
    sys.exit(f"balanced_params: no flat assignment for n={n} over {len(combos)} cells after 200 tries")


def _flat(seq, combos, m_menu, theta_menu):
    """True if seq spreads over the cells and over each parameter to within one run."""
    cells = Counter(seq)
    for counts in (
        [cells[c] for c in combos],
        [Counter(x[0] for x in seq)[v] for v in m_menu],
        [Counter(x[1] for x in seq)[v] for v in theta_menu],
    ):
        if max(counts) - min(counts) > 1:
            return False
    return True


def _geometry_qc():
    """case/geometry_qc.py, loaded by file path under a unique module name.

    case/ is a folder of scripts, not a package; loading by path keeps the
    import independent of sys.path and of the caller's working directory.
    """
    spec = importlib.util.spec_from_file_location("case_geometry_qc", CASE / "geometry_qc.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def percolating(candidates, subdir):
    """Drop geometries whose pore space does not connect inlet to outlet.

    Applied to the candidate pool before sampling, so a non-percolating cube is
    never part of the draw. Such a cube produces nothing: oil fills what it can
    reach, the displaced water has nowhere to go, and the run drives the
    pressure into the dp cap at a few percent of the step budget.

    Fails closed — a missing cache entry is computed (~0.1 s per cube, once),
    never skipped.
    """
    qc = _geometry_qc()
    if not qc.CACHE.exists():
        print(f"{qc.CACHE.name} missing — building geometry QC cache (~1 min, once)", file=sys.stderr)
    cache = json.loads(qc.CACHE.read_text()) if qc.CACHE.exists() else {}

    keep = []
    for ds, stem in candidates:
        if qc.ok(DATA / ds / subdir / f"{stem}.npy", cache, persist=True):
            keep.append((ds, stem))
        else:
            print(f"  QC: {ds}/{subdir}/{stem} does not percolate — excluded from the pool", file=sys.stderr)
    return keep


def quotas(sources, n):
    """Even split of n runs across the sorted sources; the remainder goes to
    the first source(s) in sorted order (22/21/21 into 64, 3/3/2 into 8)."""
    srcs = sorted(sources)
    q, r = divmod(n, len(srcs))
    return {ds: q + (i < r) for i, ds in enumerate(srcs)}


def _pick_geometry(ds, m, th, unused, floor_pool, uses, triples, rc_of, rng):
    """One throat-screened geometry for a (source, M, theta) slot.

    Preference order (each tier deterministic given the rng stream):
      1. an unused rock with margin >= MARGIN_MIN — random among them;
      2. reuse a wide rock at a different (M, theta) — least-used first,
         flagged ``reused`` (run names embed the triple, so only an exact
         repeat would collide);
      3. the largest remaining margin, unused before used — flagged
         ``margin x.xx``. The r_c > RC_FLOOR floor was applied to the pool
         and is never waived.
    """
    need = MARGIN_MIN * r_cap(th)
    flags = []
    elig_unused = [s for s in unused if rc_of(s) >= need]
    if elig_unused:
        stem = rng.choice(elig_unused)
    else:
        elig_used = [s for s in floor_pool if uses[s] and rc_of(s) >= need and (s, m, th) not in triples]
        if elig_used:
            low = min(uses[s] for s in elig_used)
            stem = rng.choice([s for s in elig_used if uses[s] == low])
            flags.append("reused")
        elif unused:
            stem = max(unused, key=lambda s: (rc_of(s), s))
            flags.append(f"margin {rc_of(stem) / r_cap(th):.2f}")
        else:
            rest = [s for s in floor_pool if (s, m, th) not in triples]
            if not rest:
                sys.exit(f"{ds}: floor pool exhausted even for reuse at (M={m:g}, theta={th})")
            low = min(uses[s] for s in rest)
            stem = max((s for s in rest if uses[s] == low), key=lambda s: (rc_of(s), s))
            flags.append("reused")
            if rc_of(stem) < need:
                flags.append(f"margin {rc_of(stem) / r_cap(th):.2f}")
    return stem, flags


def draw_set(case, spec, fam, datasets, dom, subdir, n, gpus, theta_menu, throats, rng, prior=None):
    """One (case, family, domain) set: n runs, even source split, throat screen.

    ``prior`` carries an earlier phase's runs for the same set into this draw:
    their rocks start as used (so an unused rock is always preferred) and their
    (geometry, M, theta) triples are off-limits (run names embed the triple, so
    an exact repeat would collide on disk). This is how extra_beta_1g continues
    the base draw without touching it.
    """
    quota = quotas(datasets, n)
    params = balanced_params(spec["m"], theta_menu, n, rng)
    slots = [ds for ds in sorted(datasets) for _ in range(quota[ds])]
    rng.shuffle(slots)

    floor_pool, unused, uses = {}, {}, {}
    for ds in sorted(datasets):
        stems = [s for _, s in percolating([(ds, s) for s in pool(ds, subdir)], subdir)]
        if spec["throat"]:
            missing = [s for s in stems if (ds, dom, s) not in throats]
            if missing:
                sys.exit(f"{case}/{fam}/{dom}: {ds} has {len(missing)} stems missing from {THROATS.name}")
            stems = [s for s in stems if throats[ds, dom, s] > RC_FLOOR]
        if not stems:
            sys.exit(f"{case}/{fam}/{dom}: {ds} pool is empty after QC/throat filtering")
        floor_pool[ds] = stems
        unused[ds] = list(stems)
        uses[ds] = Counter()

    triples = set()
    for r in prior or ():
        ds = r["dataset"]
        if r["stem"] in unused[ds]:
            unused[ds].remove(r["stem"])
        uses[ds][r["stem"]] += 1
        triples.add((r["stem"], r["m"], r["theta"]))

    # hardest theta first: the theta-150 runs need the widest rocks, so they
    # pick before a theta-120 run (eligible for anything) can take one.
    order = sorted(range(n), key=lambda i: (-params[i][1], i))
    picked = [None] * n
    for i in order:
        m, th = params[i]
        ds = slots[i]
        if spec["throat"]:
            rc_of = lambda s, ds=ds: throats[ds, dom, s]
            stem, flags = _pick_geometry(ds, m, th, unused[ds], floor_pool[ds], uses[ds], triples, rc_of, rng)
            rc, margin = rc_of(stem), rc_of(stem) / r_cap(th)
        elif unused[ds]:
            stem = rng.choice(unused[ds])
            rc = margin = None
            flags = []
        else:
            # No throat screen (GDL) and the pool ran dry — the expansion case
            # (the base GDL generated draw uses all 64 fiber samples).
            # Same reuse tier as _pick_geometry: least-used first, at a triple
            # not yet run, flagged ``reused``.
            elig = [s for s in floor_pool[ds] if (s, m, th) not in triples]
            if not elig:
                sys.exit(f"{case}/{fam}/{dom}: {ds} pool exhausted even for reuse at (M={m:g}, theta={th})")
            low = min(uses[ds][s] for s in elig)
            stem = rng.choice([s for s in elig if uses[ds][s] == low])
            rc = margin = None
            flags = ["reused"]
        if stem in unused[ds]:
            unused[ds].remove(stem)
        uses[ds][stem] += 1
        triples.add((stem, m, th))
        # gdl_ct / gdl_ct_20 / gdl_ct_40 share gdl_XXXX stems: every GDL CT
        # name embeds its dataset. Rock/generated stems already carry identity.
        name_stem = f"{ds}_{stem.removeprefix('gdl_')}" if ds.startswith("gdl_ct") else stem
        picked[i] = {
            "i": i + 1,
            "dataset": ds,
            "stem": stem,
            "m": m,
            "theta": th,
            # the _b99 tag records the recoloring sharpness (beta = 0.99) in the run name
            "name": f"{spec['prefix']}_{name_stem}_{dom}_M{m:g}_th{th}_b99",
            "src": str(DATA / ds / subdir / f"{stem}.npy"),
            "gpus": gpus,
            "dir": CASE / case / "runs" / ds / dom,
            "r_c": rc,
            "margin": margin,
            "flags": ", ".join(flags),
        }
    return picked


def draw():
    """The whole campaign: {case: {family: {domain: [run, ...]}}}. Deterministic."""
    rng = random.Random(SEED)
    throats = load_throats()
    out = {}
    for case, spec in CASES.items():
        out[case] = {}
        for fam, datasets in spec["families"].items():
            out[case][fam] = {}
            for dom, (subdir, n, gpus, theta_menu) in spec["domains"].items():
                out[case][fam][dom] = draw_set(case, spec, fam, datasets, dom, subdir, n, gpus, theta_menu, throats, rng)
    names = [r["name"] for c in out.values() for f in c.values() for rs in f.values() for r in rs]
    if len(names) != len(set(names)):
        dup = sorted({n for n in names if names.count(n) > 1})
        sys.exit(f"draw produced duplicate run names, artifacts would collide: {dup}")
    return out


def draw_extra(campaign=None):
    """The extra_beta_1g expansion: EXTRA_N more 1-GPU runs per (case, family).

    Same protocol as the base draw — even source split, percolation QC, throat
    screen, balanced (M, theta) with every cell of the 4x4 grid covered exactly
    once per set (joint base+expansion coverage: 5x per cell) — from a
    separate seed (EXTRA_SEED), so the frozen base schedule cannot move. Each
    set continues where the base left off via ``prior=``: unused rocks first,
    reuse (different triple, flagged) only where a pool ran dry.
    """
    if campaign is None:
        campaign = draw()
    rng = random.Random(EXTRA_SEED)
    throats = load_throats()
    out = {}
    for case, spec in CASES.items():
        out[case] = {}
        for fam, datasets in spec["families"].items():
            out[case][fam] = {}
            for dom, (subdir, _n, gpus, theta_menu) in spec["domains"].items():
                if gpus != 1:  # _1g: the expansion is small-domain / 1-GPU only
                    continue
                out[case][fam][dom] = draw_set(
                    case, spec, fam, datasets, dom, subdir, EXTRA_N, gpus, theta_menu, throats, rng,
                    prior=campaign[case][fam][dom],
                )
    names = [r["name"] for c in (campaign, out) for cs in c.values() for f in cs.values() for rs in f.values() for r in rs]
    if len(names) != len(set(names)):
        dup = sorted({n for n in names if names.count(n) > 1})
        sys.exit(f"extra draw collides with existing run names: {dup}")
    return out


def draw_all():
    """Both frozen phases, in schedule order: {'beta': ..., 'extra_beta_1g': ...}."""
    base = draw()
    return {"beta": base, EXTRA_PHASE: draw_extra(base)}


def status_of(run):
    """Progress read back from disk: the run's finish_type, or 'running'/'—'.

    trapping does not write run_meta.json — it writes one per stage, and the
    flood is the deliverable, so fall back to those before giving up.
    """
    d = run["dir"] / run["name"]
    for fn in ("run_meta.json", "run_meta_flood.json", "run_meta_stab.json"):
        meta = d / fn
        if not meta.exists():
            continue
        try:
            doc = json.loads(meta.read_text())
        except Exception:
            return "?"
        ft = doc.get("extra", {}).get("finish_type")
        if ft:
            return "stage 1 done" if fn.endswith("_stab.json") else ft
        return doc.get("run", {}).get("status", "running")
    return "—"


IN_FLIGHT = ("—", "?", "running", "stage 1 done")


def tables(campaign, extra):
    """The generated README section: draw summary + per-case manifests, both phases."""
    L = []
    L.append("## Manifest — the frozen 432-run draw\n")
    L.append(f"Seed {SEED}. Regenerate/refresh with `.venv/bin/python case/campaign_draw.py`; "
             "`--check` verifies the draw is unchanged. `r_c` is the critical throat radius from "
             "`geometry_throats.csv` (lattice units, pool floor r_c > 5), `margin` is r_c / r_cap(theta) "
             "against the drainage dp cap (eligibility >= 1.3). Flags: `reused` = geometry repeated at a "
             "different (M, theta) because its source ran out of unused wide rocks; `margin x.xx` = best "
             "remaining rock fell short of 1.3 (the floor is never waived).\n")

    L.append("| Case | Family | Sources (even split) | Domain | GPUs | Runs | flagged |")
    L.append("|---|---|---|---|---:|---:|---:|")
    total = flagged_total = 0
    for case, spec in CASES.items():
        for fam, datasets in spec["families"].items():
            for dom, (_sub, n, gpus, _th) in spec["domains"].items():
                runs = campaign[case][fam][dom]
                nf = sum(1 for r in runs if r["flags"])
                total += n
                flagged_total += nf
                L.append(
                    f"| {case} | {fam} | {', '.join(sorted(datasets))} | {dom} | {gpus} | {n} | {nf or '—'} |"
                )
    L.append(f"| **total** | | | | | **{total}** | **{flagged_total}** |")
    L.append("")

    L.append("\n## Termination\n")
    L.append("What ends a run, per case — the `finish_type` recorded in its `run_meta.json`:\n")
    L.append("| Case | Stops on |")
    L.append("|---|---|")
    for case, rule in TERMINATION.items():
        L.append(f"| `{case}` | {rule} |")
    L.append("\nEvery case also records `diverged` on a non-finite saturation, and `step_cap` (trapping: `cap`) "
             "if it reaches its safety step cap first.\n")

    for case in CASES:
        rows = [
            (fam, dom, r)
            for fam, doms in campaign[case].items()
            for dom, runs in doms.items()
            for r in runs
        ]
        states = [status_of(r) for _f, _d, r in rows]
        done = sum(1 for s in states if s not in IN_FLIGHT)
        going = sum(1 for s in states if s in ("running", "stage 1 done"))
        L.append(f"\n## {case} — {len(rows)} runs\n")
        L.append(f"**{len(rows)} runs** — {done} finished, {going} running.\n")
        L.append("| # | Family | Domain | GPUs | Run | Geometry | r_c | margin | M | theta | Flags | Status |")
        L.append("|---:|---|---|---:|---|---|---:|---:|---:|---:|---|---|")
        for k, (fam, dom, r) in enumerate(rows, 1):
            rc = f"{r['r_c']:.2f}" if r["r_c"] is not None else "—"
            mg = f"{r['margin']:.2f}" if r["margin"] is not None else "—"
            L.append(
                f"| {k} | {fam} | {dom} | {r['gpus']} | `{r['name']}` | "
                f"`{r['dataset']}/{dom}/{r['stem']}.npy` | "
                f"{rc} | {mg} | {r['m']:g} | {r['theta']} | {r['flags'] or ''} | {status_of(r)} |"
            )

    # ------------------------------------------------ extra_beta_1g expansion
    L.append(f"\n## Manifest — {EXTRA_PHASE}, the frozen 96-run expansion\n")
    L.append(f"Seed {EXTRA_SEED}, drawn after (and never touching) the base draw above: {EXTRA_N} more "
             "1-GPU runs per (case, family) under the same protocol — even source split, percolation QC, "
             "throat screen, and every (M, theta) cell of the 4x4 grid exactly once per set (joint "
             "base+expansion coverage: 5x per cell). A rock the base already ran is only repeated at a "
             "different (M, theta), flagged `reused` — the GDL generated set is all reuse, because the "
             "base draw uses all 64 fiber samples.\n")

    L.append("| Case | Family | Sources (even split) | Domain | GPUs | Runs | flagged |")
    L.append("|---|---|---|---|---:|---:|---:|")
    total = flagged_total = 0
    for case, spec in CASES.items():
        for fam, datasets in spec["families"].items():
            for dom, runs in extra[case][fam].items():
                nf = sum(1 for r in runs if r["flags"])
                total += len(runs)
                flagged_total += nf
                L.append(
                    f"| {case} | {fam} | {', '.join(sorted(datasets))} | {dom} | 1 | {len(runs)} | {nf or '—'} |"
                )
    L.append(f"| **total** | | | | | **{total}** | **{flagged_total}** |")
    L.append("")

    for case in CASES:
        rows = [
            (fam, dom, r)
            for fam, doms in extra[case].items()
            for dom, runs in doms.items()
            for r in runs
        ]
        states = [status_of(r) for _f, _d, r in rows]
        done = sum(1 for s in states if s not in IN_FLIGHT)
        going = sum(1 for s in states if s in ("running", "stage 1 done"))
        L.append(f"\n## {EXTRA_PHASE} — {case} — {len(rows)} runs\n")
        L.append(f"**{len(rows)} runs** — {done} finished, {going} running.\n")
        L.append("| # | Family | Domain | GPUs | Run | Geometry | r_c | margin | M | theta | Flags | Status |")
        L.append("|---:|---|---|---:|---|---|---:|---:|---:|---:|---|---|")
        for k, (fam, dom, r) in enumerate(rows, 1):
            rc = f"{r['r_c']:.2f}" if r["r_c"] is not None else "—"
            mg = f"{r['margin']:.2f}" if r["margin"] is not None else "—"
            L.append(
                f"| {k} | {fam} | {dom} | {r['gpus']} | `{r['name']}` | "
                f"`{r['dataset']}/{dom}/{r['stem']}.npy` | "
                f"{rc} | {mg} | {r['m']:g} | {r['theta']} | {r['flags'] or ''} | {status_of(r)} |"
            )
    return "\n".join(L)


def spliced_readme(generated):
    """Hand-written README documentation + marker + the generated manifest."""
    path = CASE / "README.md"
    text = path.read_text() if path.exists() else ""
    hand = text.split(MARKER)[0].rstrip("\n")
    return f"{hand}\n\n{MARKER}\n\n{generated}\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify README.md still matches the draw (exit 1 if not)")
    args = ap.parse_args()
    campaign = draw()
    extra = draw_extra(campaign)
    body = spliced_readme(tables(campaign, extra))
    out = CASE / "README.md"
    if args.check:
        cur = out.read_text() if out.exists() else ""
        if MARKER not in cur:
            sys.exit("case/README.md has no manifest marker — run campaign_draw.py")

        # Compare the plan, not the progress: the Status column and the
        # "N finished" counters move as runs land and are normalised out.
        def strip(s):
            return [
                re.sub(r"— \d+ finished, \d+ running\.", "— progress.", ln.rsplit("|", 2)[0])
                for ln in s.splitlines()
            ]

        if strip(cur) != strip(body):
            sys.exit("case/README.md does not match the frozen draw — rerun campaign_draw.py")
        print("draw unchanged")
        return
    out.write_text(body, encoding="utf-8")
    n = sum(len(rs) for c in (campaign, extra) for cs in c.values() for f in cs.values() for rs in f.values())
    flagged = sum(
        1 for c in (campaign, extra) for cs in c.values() for f in cs.values() for rs in f.values() for r in rs
        if r["flags"]
    )
    print(f"wrote {out} — {n} runs across {len(CASES)} cases, both phases ({flagged} flagged)")


if __name__ == "__main__":
    main()
