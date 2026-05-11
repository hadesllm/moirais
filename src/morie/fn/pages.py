# morie.fn — function file (hadesllm/morie)
"""Page's L trend test. 'Hope is not lost today. It is found. -- Poe'"""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._containers import TestResult


def page_trend_test(ranks_matrix: np.ndarray, cdf=None) -> TestResult:
    """Page's L test for ordered alternatives in a randomised block design.

    Tests whether k treatments have a monotone trend in their
    ranks across n blocks.  The input is an (n, k) matrix of
    ranks within each block.

    :param ranks_matrix: (n_blocks, k_treatments) rank matrix.
    :return: TestResult with Page's L statistic and z-approximation p-value.
    """
    R = np.asarray(ranks_matrix, dtype=float)
    if R.ndim != 2:
        raise ValueError("ranks_matrix must be 2-D (blocks x treatments).")
    n, k = R.shape

    col_rank_sums = R.sum(axis=0)
    L = float(np.sum(col_rank_sums * np.arange(1, k + 1)))

    E_L = n * k * (k + 1) ** 2 / 4.0
    Var_L = n * k**2 * (k + 1) * (k**2 - 1) / 144.0
    z = (L - E_L) / np.sqrt(Var_L)
    pval = float(1.0 - stats.norm.cdf(z))

    return TestResult(
        test_name="Page's L",
        statistic=L,
        p_value=pval,
        method="Page trend test (normal approx)",
        n=n,
        extra={"k": k, "z": float(z), "E_L": E_L, "Var_L": Var_L},
    )


pages = page_trend_test


def cheatsheet() -> str:
    return "page_trend_test({}) -> Page's L trend test. 'Hope is not lost today. It is found. -"
