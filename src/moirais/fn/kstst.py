# moirais.fn — function file (hadesllm/moirais)
"""Kolmogorov-Smirnov test."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import TestResult


def ks_test(
    x,
    y=None,
    *,
    distribution: str = "norm",
    alternative: str = "two-sided",
) -> TestResult:
    """Kolmogorov-Smirnov goodness-of-fit or two-sample test.

    If *y* is None, tests *x* against the named distribution.
    If *y* is given, performs a two-sample KS test.

    Parameters
    ----------
    x : array-like
        First sample.
    y : array-like, optional
        Second sample for two-sample test.
    distribution : str
        Reference distribution name (default ``"norm"``).
    alternative : str
        ``"two-sided"``, ``"less"``, or ``"greater"``.

    Returns
    -------
    TestResult
    """
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 1:
        raise ValueError("Need at least 1 finite observation.")

    if y is not None:
        b = np.asarray(y, dtype=float)
        b = b[np.isfinite(b)]
        if len(b) < 1:
            raise ValueError("Second sample must be non-empty.")
        stat, pval = sp_stats.ks_2samp(a, b, alternative=alternative)
        method = "two-sample"
        n = len(a) + len(b)
    else:
        stat, pval = sp_stats.kstest(a, distribution, alternative=alternative)
        method = f"one-sample vs {distribution}"
        n = len(a)

    return TestResult(
        test_name="Kolmogorov-Smirnov test",
        statistic=float(stat),
        p_value=float(pval),
        n=n,
        method=method,
    )


kstst = ks_test


def cheatsheet() -> str:
    return "ks_test(x, y=None) -> Kolmogorov-Smirnov test."
