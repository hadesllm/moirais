# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bowman-Shenton normality test."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import TestResult


def bowman_shenton_test(x, cdf=None) -> TestResult:
    """Bowman-Shenton omnibus test for normality.

    Combines skewness and kurtosis into a chi-squared statistic
    (similar to Jarque-Bera but with the Bowman-Shenton correction).

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
    if n < 8:
        raise ValueError("Need at least 8 finite observations.")

    skew = sp_stats.skew(a, bias=False)
    kurt = sp_stats.kurtosis(a, bias=False)

    s2 = 6.0 * (n - 2) / ((n + 1) * (n + 3))
    k2 = 24.0 * n * (n - 2) * (n - 3) / ((n + 1) ** 2 * (n + 3) * (n + 5))

    bs = skew**2 / s2 + kurt**2 / k2
    pval = 1.0 - sp_stats.chi2.cdf(bs, df=2)

    return TestResult(
        test_name="Bowman-Shenton test",
        statistic=float(bs),
        p_value=float(pval),
        df=2.0,
        n=n,
        method="omnibus normality",
        extra={"skewness": float(skew), "kurtosis": float(kurt)},
    )


bptet = bowman_shenton_test


def cheatsheet() -> str:
    return "bowman_shenton_test(x) -> Bowman-Shenton normality test."
