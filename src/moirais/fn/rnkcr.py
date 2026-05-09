# moirais.fn — function file (hadesllm/moirais)
"""Rank correlation test."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import TestResult


def rank_correlation_test(
    x,
    y,
    *,
    method: str = "spearman",
    alternative: str = "two-sided",
) -> TestResult:
    """Test for rank correlation (Spearman or Kendall).

    Parameters
    ----------
    x, y : array-like
        Paired observations.
    method : str
        ``"spearman"`` or ``"kendall"``.
    alternative : str
        ``"two-sided"``, ``"greater"``, or ``"less"``.

    Returns
    -------
    TestResult
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3:
        raise ValueError("Need at least 3 observations.")

    if method == "spearman":
        r, pval = sp_stats.spearmanr(x, y, alternative=alternative)
        name = "Spearman rank correlation test"
    elif method == "kendall":
        r, pval = sp_stats.kendalltau(x, y, alternative=alternative)
        name = "Kendall rank correlation test"
    else:
        raise ValueError("method must be 'spearman' or 'kendall'.")

    return TestResult(
        test_name=name,
        statistic=float(r),
        p_value=float(pval),
        n=n,
        method=f"{method}, alternative={alternative}",
    )


rnkcr = rank_correlation_test


def cheatsheet() -> str:
    return "rank_correlation_test(x, y) -> Rank correlation test."
