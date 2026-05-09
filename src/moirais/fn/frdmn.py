# moirais.fn — function file (hadesllm/moirais)
"""Friedman test for repeated measures."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import TestResult


def friedman_test(data: np.ndarray) -> TestResult:
    """Friedman rank-sum test for k related samples.

    Parameters
    ----------
    data : (n_subjects, k_treatments) array

    Returns
    -------
    TestResult
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 2:
        raise ValueError("Need (n>=2, k>=2) data matrix.")

    stat, p = sp_stats.friedmanchisquare(*[data[:, j] for j in range(data.shape[1])])
    n, k = data.shape
    df = k - 1

    return TestResult(
        test_name="Friedman",
        statistic=float(stat),
        p_value=float(p),
        df=float(df),
        method="Friedman chi-square",
        n=n,
        extra={"k": k},
    )


frdmn = friedman_test


def cheatsheet() -> str:
    return "friedman_test({}) -> Friedman test for repeated measures."
