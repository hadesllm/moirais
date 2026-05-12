"""Power function of the sign test (Gibbons Ch 5.4.4).

For a sample of size n testing H0: median = mu0 vs. an
alternative parametrised by p = P(X > mu0) != 0.5, the sign-test
statistic K = #{X_i > mu0} is Binomial(n, p) under both H0 (p=0.5)
and H1 (p = p_alt).

Power at a given p_alt:
    pi(p_alt) = P(K >= k_upper | p_alt) + P(K <= k_lower | p_alt)
where the critical region {K <= k_lower, K >= k_upper} is the
size-alpha rejection region under H0.

Function: given the observed sample x, computes the achieved
power at user-supplied ``p_alt`` (default 0.7).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["sign_test_power"]


def sign_test_power(x, mu0: float = 0.0, p_alt: float = 0.7, alpha: float = 0.05):
    """Power of the two-sided sign test at alternative ``p_alt``.

    Parameters
    ----------
    x : array-like
        Sample whose size determines n (values used only to count
        non-zero differences against mu0).
    mu0 : float
        Null median.
    p_alt : float in (0, 1)
        Probability P(X > mu0) under the alternative.
    alpha : float
        Nominal two-sided level.

    Returns
    -------
    RichResult with payload:
        statistic    : power (1 - beta) at p_alt
        n            : non-zero count
        p_alt, alpha : echoed
        k_lower, k_upper : two-sided critical values under H0
        size         : achieved size at H0 (because binomial is
                       discrete, achieved size <= alpha)
    """
    x = np.asarray(x, dtype=float).ravel()
    n = int((x != mu0).sum())
    if n < 1 or not (0 < p_alt < 1):
        return RichResult(payload={
            "statistic": np.nan, "n": n,
            "p_alt": float(p_alt), "alpha": float(alpha),
            "method": "Sign-test power",
        })
    # Build the two-sided rejection region under H0 (p=0.5)
    k_grid = np.arange(0, n + 1)
    null_pmf = stats.binom.pmf(k_grid, n, 0.5)
    # Two-sided: sort by null pmf ascending, accumulate until size <= alpha
    order = np.argsort(null_pmf)
    cum = 0.0
    reject = np.zeros(n + 1, dtype=bool)
    for k in order:
        if cum + null_pmf[k] <= alpha:
            reject[k] = True
            cum += null_pmf[k]
        else:
            break
    size = float(cum)
    if not reject.any():
        # Can't reject at this alpha (sample too small) — power undefined
        return RichResult(payload={
            "statistic": 0.0, "n": n, "p_alt": float(p_alt),
            "alpha": float(alpha), "size": 0.0,
            "method": "Sign-test power",
            "warnings": [f"No rejection region of size <= {alpha} exists for n={n}"]
        })
    alt_pmf = stats.binom.pmf(k_grid, n, p_alt)
    power = float(alt_pmf[reject].sum())
    rej_ks = k_grid[reject]
    return RichResult(payload={
        "statistic": power,
        "n": n,
        "p_alt": float(p_alt),
        "alpha": float(alpha),
        "size": size,
        "k_lower": int(rej_ks.min()),
        "k_upper": int(rej_ks.max()),
        "method": "Two-sided sign-test power function",
    })


def cheatsheet():
    return "sgnpw: Power function of the sign test"


# CANONICAL TEST
# >>> sign_test_power([1]*20, mu0=0, p_alt=0.7, alpha=0.05)
# n=20, p_alt=0.7: power ≈ 0.42 (exact via Binomial PMF)
