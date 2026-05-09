# moirais.fn — function file (hadesllm/moirais)
"""Page's L trend test for ordered alternatives."""

from typing import Union

import numpy as np
import scipy.stats as stats

from ._containers import TestResult


def page_trend_test(
    data: Union[list, np.ndarray],
) -> TestResult:
    """
    Page's L test for ordered alternatives in a randomised block design.

    Tests whether *k* treatments have a monotonic ordering across *n*
    subjects/blocks. Each row is ranked and the weighted sum is computed:

    .. math::

        L = \\sum_{j=1}^{k} j \\cdot R_j

    where :math:`R_j = \\sum_{i=1}^n r_{ij}` is the rank-sum for treatment *j*.

    Under H0 (no trend), L is approximately normal for moderate n and k.

    :param data: Matrix (n x k), rows = subjects/blocks, cols = treatments
        in hypothesised order (increasing expected values).
    :return: TestResult with L statistic, z, p_value.
    :raises ValueError: If data is not 2-D with k >= 2.

    References
    ----------
    Page, E. B. (1963). Ordered hypotheses for multiple treatments: A
        significance test for linear ranks. Journal of the American
        Statistical Association, 58(301), 216-230.
    """
    mat = np.asarray(data, dtype=float)
    if mat.ndim != 2:
        raise ValueError(f"data must be 2-D (n x k), got {mat.ndim}-D.")
    n, k = mat.shape
    if k < 2:
        raise ValueError("Page's L requires at least 2 treatments (columns).")

    # Rank within each row (block)
    from scipy.stats import rankdata

    ranks = np.apply_along_axis(rankdata, axis=1, arr=mat)

    # Column rank sums R_j
    R = ranks.sum(axis=0)

    # L = sum of j * R_j (j = 1..k)
    j = np.arange(1, k + 1)
    L = float(np.sum(j * R))

    # Expected value and variance under H0
    mu_L = n * k * (k + 1) ** 2 / 4.0
    var_L = n * k**2 * (k + 1) * (k**2 - 1) / 144.0

    z = (L - mu_L) / np.sqrt(var_L) if var_L > 0 else 0.0
    # One-sided test (greater): reject for large L
    p_value = float(stats.norm.sf(z))

    return TestResult(
        test_name="Page L",
        statistic=L,
        p_value=p_value,
        df=k - 1,
        method="Page's L trend test",
        n=n,
        extra={"z": float(z), "k": k, "expected_L": mu_L},
    )


page = page_trend_test


def cheatsheet() -> str:
    return "page_trend_test({}) -> Page's L trend test for ordered alternatives."
