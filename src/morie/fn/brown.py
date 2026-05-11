# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Brown-Forsythe test for equality of variances."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import TestResult


def brown_forsythe(*groups: np.ndarray) -> TestResult:
    """Brown-Forsythe test (Levene with median).

    Parameters
    ----------
    *groups : array-like
        Two or more groups.

    Returns
    -------
    TestResult
    """
    if len(groups) < 2:
        raise ValueError("Need >= 2 groups.")
    groups = [np.asarray(g, dtype=float) for g in groups]
    stat, p = sp_stats.levene(*groups, center="median")
    k = len(groups)
    n = sum(len(g) for g in groups)

    return TestResult(
        test_name="Brown-Forsythe",
        statistic=float(stat),
        p_value=float(p),
        df=float(k - 1),
        method="Brown-Forsythe (Levene median)",
        n=n,
        extra={"k": k, "df2": float(n - k)},
    )


brown = brown_forsythe


def cheatsheet() -> str:
    return "brown_forsythe({}) -> Brown-Forsythe test for equality of variances."
