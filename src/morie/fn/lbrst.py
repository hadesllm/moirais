# morie.fn — function file (hadesllm/morie)
"""Lilliefors test for normality."""

from __future__ import annotations

import numpy as np

from ._containers import TestResult


def lilliefors_test(x, cdf=None) -> TestResult:
    """Lilliefors test (KS test with estimated parameters).

    A modification of the Kolmogorov-Smirnov test where the mean
    and variance are estimated from the data.

    Parameters
    ----------
    x : array-like
        Observations.

    Returns
    -------
    TestResult
    """
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    n = len(a)
    if n < 4:
        raise ValueError("Need at least 4 finite observations.")

    from scipy.stats import norm

    z = (a - np.mean(a)) / np.std(a, ddof=1)
    z_sorted = np.sort(z)
    cdf_vals = norm.cdf(z_sorted)
    ecdf = np.arange(1, n + 1) / n
    ecdf_lower = np.arange(0, n) / n

    d_plus = np.max(ecdf - cdf_vals)
    d_minus = np.max(cdf_vals - ecdf_lower)
    d_stat = max(d_plus, d_minus)

    cv_5 = 0.886 / np.sqrt(n)

    return TestResult(
        test_name="Lilliefors test",
        statistic=float(d_stat),
        p_value=-1.0,
        n=n,
        method="KS with estimated parameters",
        extra={
            "critical_value_5pct": float(cv_5),
            "reject_at_5pct": d_stat > cv_5,
        },
    )


lbrst = lilliefors_test


def cheatsheet() -> str:
    return "lilliefors_test(x) -> Lilliefors normality test."
