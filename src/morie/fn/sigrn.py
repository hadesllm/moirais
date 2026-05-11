"""Wilcoxon signed-rank test."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import TestResult


def sign_rank_test(
    x,
    y=None,
    *,
    mu: float = 0.0,
    alternative: str = "two-sided",
) -> TestResult:
    """Wilcoxon signed-rank test for matched pairs or one-sample.

    Parameters
    ----------
    x : array-like
        First sample or paired differences.
    y : array-like, optional
        Second sample (paired). If given, tests x - y.
    mu : float
        Hypothesised median difference (default 0).
    alternative : str
        ``"two-sided"``, ``"greater"``, or ``"less"``.

    Returns
    -------
    TestResult
    """
    x = np.asarray(x, dtype=float)
    if y is not None:
        y = np.asarray(y, dtype=float)
        d = x - y
    else:
        d = x - mu
    d = d[np.isfinite(d)]
    if len(d) < 1:
        raise ValueError("Need at least 1 finite observation.")

    stat, pval = sp_stats.wilcoxon(d, alternative=alternative)
    return TestResult(
        test_name="Wilcoxon signed-rank test",
        statistic=float(stat),
        p_value=float(pval),
        n=len(d),
        method=f"alternative={alternative}",
    )


sigrn = sign_rank_test


def cheatsheet() -> str:
    return "sign_rank_test(x, y) -> Wilcoxon signed-rank test."
