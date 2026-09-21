"""Tests for the trapping case's full-domain PMI kernel search (case/trapping/pmi_search.py).

PMI places the initial oil with a ball of integer "kernel" size k; rock water
saturation Sw(k) rises with k (a bigger ball fits in fewer pores), and once no
ball fits anywhere Sw sits on a flat 1.0 plateau. PMI's thin-band kernel guess
can land deep in that plateau (sphere128_0031: band-best k=43), where a +-1 walk
stalls and the rock would be initialised with no oil at all. The search must
climb out of a plateau, and must stay step-for-step identical to the +-1 walk
wherever that walk converges (the campaign draw is frozen).

pmi_search is pure Python, loaded by file path: importing trapping_demo itself
would set jax_enable_x64 for the whole pytest session.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

TRAPPING = Path(__file__).resolve().parents[1] / "case" / "trapping"


def _load():
    key = "case_trapping_pmi_search"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, TRAPPING / "pmi_search.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


ps = _load()


def _recorder(sw_of_k):
    """eval_k stand-in: returns (Sw, label) like PMI.compute_saturation_parallel and
    records the kernels asked for — each call is a full-domain morphology pass."""
    calls = []

    def eval_k(k):
        assert k >= 1, "PMI kernels are positive integers"
        calls.append(k)
        return sw_of_k(k), f"label_k{k}"

    return eval_k, calls


def _sphere_pack(k):
    # shaped like sphere128_0031: Sw climbs with k, then no ball fits from k=12 up
    return min(1.0, 0.05 * k * k / 3.0) if k < 12 else 1.0


def test_climbs_out_of_the_no_oil_plateau():
    """The sphere128_0031 geometry: the band guess k=43 sits where Sw == 1.0 for every
    neighbour, so a walk that needs a strictly closer step stalls there with zero oil."""
    eval_k, calls = _recorder(_sphere_pack)
    k, sw, label = ps.closest_kernel(eval_k, best_k=43, target_sw=0.4)
    assert sw < 1.0, "search returned the no-oil plateau"
    assert abs(sw - 0.4) == min(abs(_sphere_pack(j) - 0.4) for j in range(1, 60)), "not the closest reachable Sw"
    assert label == f"label_k{k}" and sw == _sphere_pack(k)
    assert len(calls) == len(set(calls)), "a kernel was evaluated twice"
    assert len(calls) <= 12, f"{len(calls)} full-domain passes — must bisect, not walk 43 -> 5 one by one"


def test_climbs_out_of_the_all_oil_plateau():
    """Mirror case: a guess so small that every neighbour is all-oil (Sw flat at 0)."""
    eval_k, calls = _recorder(lambda k: 0.0 if k <= 6 else min(1.0, 0.04 * (k - 6)))
    k, sw, _ = ps.closest_kernel(eval_k, best_k=1, target_sw=0.4)
    assert (k, sw) == (16, pytest.approx(0.4))
    assert len(calls) <= 12


def test_working_walk_is_unchanged():
    """Where the +-1 walk converges the search must evaluate the same kernels in the same
    order and return the same one, so existing campaign runs stay reproducible."""
    eval_k, calls = _recorder(lambda k: min(1.0, 0.05 * k))
    k, sw, _ = ps.closest_kernel(eval_k, best_k=4, target_sw=0.4, tol=0.02)
    assert calls == [3, 4, 5, 6, 7, 8]  # seed triple, then +1 while strictly closer, stop inside tol
    assert (k, sw) == (8, pytest.approx(0.4))


def test_unreachable_target_returns_closest_and_terminates():
    """Integer kernels quantize Sw: if the target falls in a jump, return the nearer side."""
    eval_k, calls = _recorder(lambda k: 0.25 if k <= 3 else 0.7)
    k, sw, _ = ps.closest_kernel(eval_k, best_k=20, target_sw=0.4)
    assert sw == 0.25 and k <= 3
    assert len(calls) <= 12
