# morie.fn — function file (hadesllm/morie)
"""Rank-sum test (Wilcoxon / Mann-Whitney variant)."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import TestResult


def rank_sum_test(
    x,
    y,
    *,
    alternative: str = "two-sided",
) -> TestResult:
    """Wilcoxon rank-sum test (equivalent to Mann-Whitney U).

    Parameters
    ----------
    x, y : array-like
        Two independent samples.
    alternative : str
        ``"two-sided"``, ``"greater"``, or ``"less"``.

    Returns
    -------
    TestResult
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if len(a) < 1 or len(b) < 1:
        raise ValueError("Both samples must be non-empty.")

    stat, pval = sp_stats.ranksums(a, b, alternative=alternative)
    return TestResult(
        test_name="Wilcoxon rank-sum test",
        statistic=float(stat),
        p_value=float(pval),
        n=len(a) + len(b),
        method=f"alternative={alternative}",
        extra={"n1": len(a), "n2": len(b)},
    )


rnksm = rank_sum_test


def cheatsheet() -> str:
    return "rank_sum_test(x, y) -> Wilcoxon rank-sum test."
