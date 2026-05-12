# morie.fn -- function file (hadesllm/morie)
"""Kendall's coefficient of concordance (W)."""

from __future__ import annotations

import numpy as np

from ._containers import TestResult


def kendall_concordance(ratings, cdf=None) -> TestResult:
    """Kendall's W (coefficient of concordance) for multiple raters.

    Parameters
    ----------
    ratings : array-like, shape (k, n)
        k raters x n subjects matrix of ranks.

    Returns
    -------
    TestResult
    """
    from scipy.stats import chi2

    R = np.asarray(ratings, dtype=float)
    if R.ndim != 2:
        raise ValueError("ratings must be a 2-D array (k raters x n subjects).")
    k, n = R.shape
    if k < 2 or n < 2:
        raise ValueError("Need at least 2 raters and 2 subjects.")

    rank_sums = np.sum(R, axis=0)
    mean_rank_sum = np.mean(rank_sums)
    ss = np.sum((rank_sums - mean_rank_sum) ** 2)

    w = 12.0 * ss / (k**2 * (n**3 - n))
    chi2_stat = k * (n - 1) * w
    df = n - 1
    pval = 1.0 - chi2.cdf(chi2_stat, df)

    return TestResult(
        test_name="Kendall's W",
        statistic=float(w),
        p_value=float(pval),
        df=float(df),
        n=n,
        method=f"k={k} raters",
        extra={"chi2": float(chi2_stat), "k": k},
    )


kndll = kendall_concordance


def cheatsheet() -> str:
    return "kendall_concordance(ratings) -> Kendall's W concordance."
