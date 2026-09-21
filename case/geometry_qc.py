"""Connectivity QC for campaign geometries — reject rock the solver cannot drain.

A geometry whose pore space has no inlet->outlet path produces nothing: oil fills
what it can reach near the inlet, the displaced water has nowhere to go, and the
piston drives the pressure into its cap within a few percent of the step budget.

Two details decide the verdict:

  * **Seal the y/z faces first.** Every porous driver adds core-holder walls
    (`solid[0,:,:] = solid[-1,:,:] = solid[:,0,:] = solid[:,-1,:] = True`), and
    scipy's labelling does not treat the array edge as solid — so scanning the
    raw asset would count boundary channels the solver never sees.
  * **Label with 18-connectivity.** D3Q19 streams along faces and edges, so
    6-connectivity would call a diagonal seam disconnected when the solver can
    stream through it.

Flow is along x = axis 2 for every porous case. Verdicts are cached in
geometry_qc.json, keyed by the geometry's path below the dataset root
(`<dataset>/<domain>/<name>.npy`), so they hold wherever the dataset lives.

    .venv/bin/python case/geometry_qc.py --check <file.npy> ...   # ad hoc
    .venv/bin/python case/geometry_qc.py --rebuild                # rescan the draw
    .venv/bin/python case/geometry_qc.py --report                 # list rejects
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

CASE = Path(__file__).resolve().parent
CACHE = CASE / "geometry_qc.json"
FLOW = 2  # (nz, ny, nx); drivers drain along x


def seal_yz(solid):
    """The core-holder walls every porous driver adds before it runs."""
    s = solid.copy()
    s[0, :, :] = s[-1, :, :] = True
    s[:, 0, :] = s[:, -1, :] = True
    return s


def percolates(path):
    """True if sealed pore space connects inlet face to outlet face (18-conn)."""
    solid = seal_yz(np.load(path).astype(bool))
    lab, _ = ndimage.label(~solid, structure=ndimage.generate_binary_structure(3, 2))
    a = set(np.unique(np.moveaxis(lab, FLOW, 0)[0])) - {0}
    b = set(np.unique(np.moveaxis(lab, FLOW, 0)[-1])) - {0}
    return bool(a & b)


def load_cache():
    return json.loads(CACHE.read_text()) if CACHE.exists() else {}


def save_cache(c):
    CACHE.write_text(json.dumps(dict(sorted(c.items())), indent=1))


def cache_key(src):
    """`<dataset>/<domain>/<name>.npy`: the path below the dataset root, wherever that is."""
    return "/".join(Path(src).parts[-3:])


def ok(src, cache=None, persist=True):
    """Cached verdict for one geometry path. Computes (~1 s) on a cache miss."""
    cache = load_cache() if cache is None else cache
    key = cache_key(src)
    if key not in cache:
        cache[key] = percolates(src)
        if persist:
            save_cache(cache)
    return cache[key]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", nargs="+", help="test these .npy files")
    ap.add_argument("--rebuild", action="store_true", help="rescan every geometry the draw touches")
    ap.add_argument("--report", action="store_true", help="list cached rejects")
    args = ap.parse_args()

    if args.check:
        bad = 0
        for f in args.check:
            good = percolates(f)
            bad += not good
            print(f"{'ok      ' if good else 'REJECT  '} {f}")
        sys.exit(1 if bad else 0)

    if args.rebuild:
        # campaign_draw is loaded by file path: case/ is a folder of scripts, not a package
        import importlib.util

        spec_ = importlib.util.spec_from_file_location("case_campaign_draw", CASE / "campaign_draw.py")
        cd = importlib.util.module_from_spec(spec_)
        sys.modules[spec_.name] = cd
        spec_.loader.exec_module(cd)

        cache, n = load_cache(), 0
        for spec in cd.CASES.values():
            for datasets in spec["families"].values():
                for subdir, _n, _g, _th in spec["domains"].values():
                    for ds in datasets:
                        for stem in cd.pool(ds, subdir):
                            src = cd.DATA / ds / subdir / f"{stem}.npy"
                            if cache_key(src) not in cache:
                                cache[cache_key(src)] = percolates(src)
                                n += 1
                                if n % 25 == 0:
                                    print(f"  scanned {n} ...", flush=True)
        save_cache(cache)
        print(f"scanned {n} new geometries; cache holds {len(cache)}")

    cache = load_cache()
    bad = sorted(k for k, v in cache.items() if not v)
    print(f"\n{len(cache)} geometries cached | {len(bad)} rejected as non-percolating:")
    for b in bad:
        print(f"  {b}")


if __name__ == "__main__":
    main()
