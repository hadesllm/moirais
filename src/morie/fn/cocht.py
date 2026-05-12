# morie.fn -- function file (hadesllm/morie)
"""Cochran's Q test for k related binary samples."""

from typing import Union

import numpy as np
import scipy.stats as stats

from ._containers import TestResult


def cochrans_q_test(
    data: Union[list, np.ndarray],
) -> TestResult:
    r"""
    Cochran's Q test for *k* related dichotomous samples.

    Extension of McNemar's test to more than two related groups. Tests
    whether the proportion of successes differs across *k* treatments
    applied to the same *n* subjects.

    .. math::

        Q = \\frac{k(k-1) \\sum_{j=1}^k (C_j - \\bar{C})^2}
                 {k \\sum_{i=1}^n R_i - \\sum_{i=1}^n R_i^2}

    where :math:`C_j` = column total, :math:`R_i` = row total.

    :param data: Binary matrix (n x k), rows = subjects, cols = treatments.
        Values must be 0 or 1.
    :return: TestResult with Q statistic, p_value, df=k-1.
    :raises ValueError: If data is not binary or has fewer than 2 columns.

    References
    ----------
    Cochran, W. G. (1950). The comparison of percentages in matched samples.
        Biometrika, 37(3-4), 256-266.
    """
    mat = np.asarray(data, dtype=float)
    if mat.ndim != 2:
        raise ValueError(f"data must be 2-D (n x k), got {mat.ndim}-D.")
    n, k = mat.shape
    if k < 2:
        raise ValueError("Cochran's Q requires at least 2 treatments (columns).")
    if not np.all((mat == 0) | (mat == 1)):
        raise ValueError("All values must be 0 or 1.")

    col_sums = mat.sum(axis=0)  # C_j
    row_sums = mat.sum(axis=1)  # R_i

    numerator = (k - 1) * (k * np.sum(col_sums**2) - np.sum(col_sums) ** 2)
    denominator = k * np.sum(row_sums) - np.sum(row_sums**2)

    if denominator == 0:
        return TestResult(
            test_name="Cochran Q",
            statistic=0.0,
            p_value=1.0,
            df=k - 1,
            method="Cochran's Q test (degenerate)",
            n=n,
        )

    Q = float(numerator / denominator)
    df = k - 1
    p_value = float(stats.chi2.sf(Q, df=df))

    return TestResult(
        test_name="Cochran Q",
        statistic=Q,
        p_value=p_value,
        df=df,
        method="Cochran's Q test",
        n=n,
        extra={"k": k},
    )


cocht = cochrans_q_test


def cheatsheet() -> str:
    return "cochrans_q_test({}) -> Cochran's Q test for k related binary samples."
