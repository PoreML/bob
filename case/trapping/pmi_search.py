"""Full-domain PMI kernel search for the trapping case's initial oil placement.

Pure Python on purpose (no numpy/jax/PMI import): trapping_demo.py decides the
solver precision at import, so the search lives here where test/ can load it.
"""


def closest_kernel(eval_k, best_k, target_sw, tol=0.02, k_max=100):
    """Integer PMI kernel whose full-domain water saturation is closest to ``target_sw``.

    ``eval_k(k) -> (Sw, label)`` is one full-domain morphology pass (expensive);
    ``best_k`` is PMI's thin-band guess. Walk outward from it one kernel at a time
    while each step is strictly closer, then bisect if that still misses by more
    than ``tol``. Wherever the walk converges, the kernels evaluated and the result
    are exactly those of the walk alone.
    Returns ``(k, Sw, label)`` of the closest kernel evaluated."""
    tried: dict[int, float] = {}
    best = None  # (k, Sw, label) — only the closest label is kept alive

    def visit(k):
        nonlocal best
        sat_k, comb_k = eval_k(k)
        tried[k] = sat_k
        if best is None or abs(sat_k - target_sw) < abs(best[1] - target_sw):
            best = (k, sat_k, comb_k)
            return True
        return False

    for k in sorted({max(1, best_k - 1), best_k, best_k + 1}):
        visit(k)
    while True:
        lo_k, hi_k = min(tried), max(tried)
        sat_full = best[1]
        if abs(sat_full - target_sw) <= tol:
            break
        if sat_full == tried[hi_k] and sat_full < target_sw:
            k = hi_k + 1
        elif sat_full == tried[lo_k] and sat_full > target_sw and lo_k > 1:
            k = lo_k - 1
        else:
            break
        if not visit(k):
            break
    if abs(best[1] - target_sw) > tol:
        _bisect(visit, tried, target_sw, k_max)
    return best


def _bisect(visit, tried, target_sw, k_max):
    """Bracket the target and halve. Sw(k) rises with k and is flat where it saturates
    (no ball fits -> Sw 1, every ball fits -> Sw 0), and the +-1 walk stalls on a flat
    (e.g. sphere128_0031, whose band guess k=43 lies on the no-oil plateau).
    ~log2(k) more full-domain passes instead of one per kernel."""
    if tried[min(tried)] > target_sw:
        lo, hi = 0, min(tried)  # k=0 is virtual (all oil, Sw 0) and never evaluated
    else:
        lo = hi = max(tried)
        while tried[hi] < target_sw and hi < k_max:
            lo, hi = hi, min(2 * hi, k_max)
            visit(hi)
        if tried[hi] < target_sw:
            return  # target above anything a kernel <= k_max reaches
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if mid not in tried:
            visit(mid)
        if tried[mid] > target_sw:
            hi = mid
        else:
            lo = mid
