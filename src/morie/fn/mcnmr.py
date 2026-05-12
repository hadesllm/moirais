# morie.fn -- function file (hadesllm/morie)
"""McNemar's test for paired proportions."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import TestResult


def mcnemar_test(before: np.ndarray, after: np.ndarray, cdf=None, *, correction: bool = True) -> TestResult:
    """McNemar's test for paired dichotomous data.

    Parameters
    ----------
    before, after : (n,) binary arrays
    correction : bool
        Apply continuity correction.

    Returns
    -------
    TestResult
    """
    before = np.asarray(before, dtype=int).ravel()
    after = np.asarray(after, dtype=int).ravel()
    if len(before) != len(after):
        raise ValueError("before and after must have same length.")

    b = int(np.sum((before == 1) & (after == 0)))
    c = int(np.sum((before == 0) & (after == 1)))

    if b + c == 0:
        return TestResult(
            test_name="McNemar", statistic=0.0, p_value=1.0, df=1.0, method="McNemar exact", n=len(before)
        )

    if correction:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    else:
        chi2 = (b - c) ** 2 / (b + c)

    p = float(1 - sp_stats.chi2.cdf(chi2, 1))

    return TestResult(
        test_name="McNemar",
        statistic=float(chi2),
        p_value=p,
        df=1.0,
        method="McNemar chi-square" + (" (corrected)" if correction else ""),
        n=len(before),
        extra={"b": b, "c": c},
    )


mcnmr = mcnemar_test


def cheatsheet() -> str:
    return "mcnemar_test({}) -> McNemar's test for paired proportions."
