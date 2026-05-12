# morie.fn -- function file (hadesllm/morie)
"""General two-sample permutation test (Good 2005, *Permutation Tests*).

Tests H0: F_x = F_y by shuffling group labels and comparing the observed
test statistic to its permutation distribution.  Reports a two-sided
p-value with the standard +1/(B+1) Monte Carlo continuity correction.
"""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["permutation_test_general"]


def permutation_test_general(x, y, statistic=None, B: int = 5000,
                             alternative: str = "two-sided", seed: int = 42):
    """Two-sample permutation test.

    Parameters
    ----------
    x, y : array-like
        Group samples.
    statistic : callable, optional
        ``statistic(x, y) -> scalar``.  Default = mean difference.
    B : int
        Number of permutations (default 5000).
    alternative : {"two-sided", "less", "greater"}
    seed : int

    Returns
    -------
    RichResult with keys: statistic, p_value, n_x, n_y, B, method.

    Notes
    -----
    p-value uses ``(1 + #{T_b >= T_obs}) / (B + 1)`` per Phipson & Smyth
    (2010) so that p > 0 strictly.

    References
    ----------
    Good, P. (2005). Permutation, Parametric, and Bootstrap Tests of
    Hypotheses (3rd ed.). Springer.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n_x, n_y = x.size, y.size
    if n_x < 1 or n_y < 1:
        return RichResult(payload={"statistic": float("nan"),
                                   "p_value": float("nan"),
                                   "n_x": int(n_x), "n_y": int(n_y),
                                   "method": "permutation (empty)"})
    if statistic is None:
        statistic = lambda a, b: float(np.mean(a) - np.mean(b))
    T_obs = float(statistic(x, y))
    pool = np.concatenate([x, y])
    rng = np.random.default_rng(seed)
    T_perm = np.empty(B, dtype=float)
    for b in range(B):
        rng.shuffle(pool)
        T_perm[b] = statistic(pool[:n_x], pool[n_x:])
    if alternative == "greater":
        p = (1.0 + np.sum(T_perm >= T_obs)) / (B + 1.0)
    elif alternative == "less":
        p = (1.0 + np.sum(T_perm <= T_obs)) / (B + 1.0)
    else:
        p = (1.0 + np.sum(np.abs(T_perm) >= abs(T_obs))) / (B + 1.0)
    return RichResult(payload={
        "statistic": T_obs, "p_value": float(p),
        "n_x": int(n_x), "n_y": int(n_y), "B": int(B),
        "alternative": alternative,
        "method": "Permutation test (Good 2005)",
    })


# CANONICAL TEST
# >>> import numpy as np
# >>> rng = np.random.default_rng(0)
# >>> x = rng.normal(0.0, 1.0, 30)
# >>> y = rng.normal(0.0, 1.0, 30)
# >>> res = permutation_test_general(x, y, B=1000, seed=0)
# >>> assert 0 <= res["p_value"] <= 1
# >>> # same-distribution samples -> expect p not extreme
# >>> assert res["p_value"] > 0.01


def cheatsheet():
    return "permt(x, y, statistic=mean-diff, B=5000): permutation test."
