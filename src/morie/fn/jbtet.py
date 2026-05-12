# morie.fn -- function file (hadesllm/morie)
"""Jarque-Bera test for normality."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import TestResult


def jarque_bera_test(x) -> TestResult:
    """Jarque-Bera test for normality.

    Tests whether the sample skewness and kurtosis match
    a normal distribution (H0: normal).

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

    jb, pval = sp_stats.jarque_bera(a)
    skew = float(sp_stats.skew(a, bias=False))
    kurt = float(sp_stats.kurtosis(a, bias=False))

    return TestResult(
        test_name="Jarque-Bera test",
        statistic=float(jb),
        p_value=float(pval),
        df=2.0,
        n=n,
        method="chi-squared approximation",
        extra={"skewness": skew, "kurtosis": kurt},
    )


jbtet = jarque_bera_test


def cheatsheet() -> str:
    return "jarque_bera_test(x) -> Jarque-Bera normality test."
