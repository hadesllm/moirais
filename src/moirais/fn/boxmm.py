# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Box's M test for covariance equality. 'I have a bad feeling about this. --'"""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._containers import TestResult


def box_m_test(X: np.ndarray, groups: np.ndarray, cdf=None) -> TestResult:
    """Box's M test for homogeneity of covariance matrices.

    Tests whether the covariance matrices of *g* groups are equal.
    Uses the chi-squared approximation to Box's M statistic.

    :param X: (n, p) data matrix.
    :param groups: 1-D array of group labels (length n).
    :return: TestResult with Box's M statistic and chi-squared p-value.
    """
    X = np.asarray(X, dtype=float)
    groups = np.asarray(groups)
    if X.ndim != 2:
        raise ValueError("X must be 2-D.")
    n, p = X.shape
    labels = np.unique(groups)
    g = len(labels)
    if g < 2:
        raise ValueError("Need at least 2 groups.")

    S_pooled = np.zeros((p, p))
    log_det_sum = 0.0
    n_total = 0
    ns = []
    for lab in labels:
        mask = groups == lab
        Xi = X[mask]
        ni = len(Xi) - 1
        ns.append(ni)
        Si = np.cov(Xi, rowvar=False, ddof=1)
        S_pooled += ni * Si
        sign, logdet = np.linalg.slogdet(Si)
        log_det_sum += ni * logdet
        n_total += ni

    S_pooled /= n_total
    sign_p, logdet_p = np.linalg.slogdet(S_pooled)
    M = n_total * logdet_p - log_det_sum

    c1 = (sum(1.0 / ni for ni in ns) - 1.0 / n_total) * (2 * p**2 + 3 * p - 1) / (6 * (p + 1) * (g - 1))
    df = p * (p + 1) * (g - 1) / 2
    chi2 = M * (1.0 - c1)
    pval = float(1.0 - stats.chi2.cdf(chi2, df))

    return TestResult(
        test_name="Box's M",
        statistic=float(chi2),
        p_value=pval,
        df=df,
        method="Box's M (chi-squared approx)",
        n=n,
        extra={"M": float(M), "correction_c1": c1, "n_groups": g},
    )


boxmm = box_m_test


def cheatsheet() -> str:
    return "box_m_test({}) -> Box's M test for covariance equality. 'I have a bad feeling "
