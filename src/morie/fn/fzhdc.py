# morie.fn -- function file (hadesllm/morie)
"""Hoeffding (H-) decomposition for U-statistics (Fauzi Ch 5).

For a symmetric kernel g(x1,x2):

    U_n = binom(n,2)^{-1} sum_{i<j} g(X_i, X_j),
    theta = E g(X1,X2),
    g_1(x) = E[g(x,X2)] - theta            (Hajek projection),
    sigma1^2 = Var g_1(X),
    Var(U_n) = (4/n) sigma1^2 + O(1/n^2).
"""
import numpy as np
from itertools import combinations
from ._richresult import RichResult

__all__ = ["fauzi_h_decomposition"]


def _default_kernel(a, b):
    """Default kernel: 0.5*(a-b)^2 (E = sigma^2)."""
    return 0.5 * (a - b) ** 2


def fauzi_h_decomposition(x, kernel=None, max_pairs=2000, rng=None):
    """H-decomposition of a degree-2 U-statistic.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if n < 4:
        return RichResult(payload={"estimate": np.nan, "n": n,
                                    "method": "fzhdc -- too few obs"})
    if kernel is None:
        kernel = _default_kernel
    if rng is None:
        rng = np.random.default_rng(0)

    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        pairs = list(combinations(range(n), 2))
    else:
        pairs = []
        seen = set()
        while len(pairs) < max_pairs:
            i, j = rng.integers(0, n, size=2)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            k = (int(i), int(j))
            if k in seen:
                continue
            seen.add(k)
            pairs.append(k)

    g_vals = np.array([kernel(x[i], x[j]) for i, j in pairs])
    theta = float(np.mean(g_vals))
    sigma2 = float(np.var(g_vals, ddof=1))

    g1 = np.zeros(n)
    counts = np.zeros(n)
    for (i, j), v in zip(pairs, g_vals):
        g1[i] += v; counts[i] += 1
        g1[j] += v; counts[j] += 1
    counts[counts == 0] = 1
    g1 = g1 / counts - theta
    sigma1_sq = float(np.var(g1, ddof=1))

    var_U = 4.0 * sigma1_sq / n
    se_U = float(np.sqrt(max(var_U, 0.0)))

    return RichResult(payload={
        "estimate": theta,
        "sigma1_sq": sigma1_sq,
        "sigma2_sq": sigma2,
        "se": se_U,
        "n": n,
        "n_pairs": len(pairs),
        "method": "Fauzi H-decomposition of degree-2 U-statistic (Ch 5)",
    })


def cheatsheet():
    return "fzhdc: Hoeffding/H-decomposition Var(U_n) ≈ 4 sigma1^2/n"


# CANONICAL TEST
# >>> import numpy as np
# >>> rng = np.random.default_rng(0)
# >>> x = rng.standard_normal(200)
# >>> r = fauzi_h_decomposition(x)
# >>> 0.7 < r["estimate"] < 1.3
# True
