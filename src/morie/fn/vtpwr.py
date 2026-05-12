# morie.fn — function file (hadesllm/morie)
"""Banzhaf and Shapley-Shubik voting-power indices (Armstrong Ch 10)."""
import numpy as np
from itertools import combinations
from math import factorial
from ._richresult import RichResult

__all__ = ["voting_power_index", "vtpwr"]


def voting_power_index(x, quota=None):
    """Banzhaf (β) and Shapley-Shubik (φ) voting-power indices for a
    weighted-voting game [q; w_1, …, w_n].

    Banzhaf: β_i = |swings of i| / Σ_k |swings of k|
    Shapley-Shubik: φ_i = (n!)⁻¹ * Σ_{orderings} I{i is pivotal}

    Parameters
    ----------
    x : array-like (n,)
        Voter weights w_i.
    quota : float, optional
        Winning threshold q (default = ceil(sum(w)/2 + 1) to be strictly
        more than half — the simple-majority rule).

    Returns
    -------
    RichResult with keys: banzhaf, shapley_shubik, quota, weights
    """
    w = np.asarray(x, dtype=float).ravel()
    n = int(w.size)
    if n == 0:
        return RichResult(payload={"banzhaf": np.array([]),
                                   "shapley_shubik": np.array([]),
                                   "quota": np.nan, "weights": w,
                                   "method": "voting_power_index"})
    total = float(w.sum())
    if quota is None:
        quota = total / 2.0 + 1e-9  # strict majority
    quota = float(quota)
    if n > 20:
        # Banzhaf via Monte Carlo for large n
        rng = np.random.default_rng(0)
        N_mc = 20000
        swings = np.zeros(n)
        for _ in range(N_mc):
            mask = rng.integers(0, 2, size=n).astype(bool)
            tot_in = w[mask].sum()
            for i in range(n):
                if mask[i]:
                    swings[i] += (tot_in >= quota
                                  and (tot_in - w[i]) < quota)
                else:
                    swings[i] += ((tot_in + w[i]) >= quota
                                  and tot_in < quota)
        banzhaf = swings / max(swings.sum(), 1)
        # Shapley-Shubik via permutation MC
        ss = np.zeros(n)
        for _ in range(N_mc):
            order = rng.permutation(n)
            cum = 0.0
            for k in order:
                prev = cum
                cum += w[k]
                if prev < quota <= cum:
                    ss[k] += 1
                    break
        shapley = ss / N_mc
        return RichResult(
            title="Voting power indices (MC, n > 20)",
            summary_lines=[("quota q", quota), ("n voters", n)],
            payload={"banzhaf": banzhaf, "shapley_shubik": shapley,
                     "quota": quota, "weights": w,
                     "method": "voting_power_index_mc"},
        )
    # Exact Banzhaf
    swings = np.zeros(n)
    for size in range(0, n + 1):
        for coalition in combinations(range(n), size):
            tot_in = float(w[list(coalition)].sum()) if size else 0.0
            for i in range(n):
                if i in coalition:
                    if tot_in >= quota and (tot_in - w[i]) < quota:
                        swings[i] += 1
                else:
                    if (tot_in + w[i]) >= quota and tot_in < quota:
                        swings[i] += 1
    banzhaf = swings / max(swings.sum(), 1)
    # Exact Shapley-Shubik by enumerating ordered pivots via combinations
    shapley = np.zeros(n)
    if n <= 10:
        from itertools import permutations
        n_perm = factorial(n)
        for order in permutations(range(n)):
            cum = 0.0
            for k in order:
                prev = cum
                cum += w[k]
                if prev < quota <= cum:
                    shapley[k] += 1
                    break
        shapley = shapley / n_perm
    else:
        # Combinatorial pivotal-coalition formula (Shapley 1953):
        # φ_i = Σ_{S not containing i, |S|=s, v(S)<q, v(S∪{i})≥q}
        #         s!(n-s-1)!/n!
        for i in range(n):
            others = [k for k in range(n) if k != i]
            for s in range(0, n):
                for S in combinations(others, s):
                    vS = float(w[list(S)].sum()) if S else 0.0
                    if vS < quota <= vS + w[i]:
                        shapley[i] += (factorial(s)
                                       * factorial(n - s - 1)
                                       / factorial(n))
    return RichResult(
        title="Voting power indices (exact)",
        summary_lines=[("quota q", quota), ("n voters", n),
                       ("Banzhaf β", list(np.round(banzhaf, 4))),
                       ("Shapley-Shubik φ", list(np.round(shapley, 4)))],
        payload={"banzhaf": banzhaf, "shapley_shubik": shapley,
                 "quota": quota, "weights": w,
                 "method": "voting_power_index_exact"},
    )


vtpwr = voting_power_index


def cheatsheet():
    return "vtpwr: Banzhaf and Shapley-Shubik voting-power indices."


# CANONICAL TEST
# >>> r = voting_power_index([3,2,1], quota=4)
# >>> assert abs(r["banzhaf"].sum() - 1.0) < 1e-9
