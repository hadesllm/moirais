# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Box's M test for equality of covariance matrices."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import TestResult


def box_m_test(X: np.ndarray, groups: np.ndarray, cdf=None) -> TestResult:
    """Box's M test.

    Parameters
    ----------
    X : (n, p) array
    groups : (n,) labels

    Returns
    -------
    TestResult
    """
    X = np.asarray(X, dtype=float)
    groups = np.asarray(groups)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    labels = np.unique(groups)
    k = len(labels)
    if k < 2:
        raise ValueError("Need >= 2 groups.")

    ns = []
    Ss = []
    Sp = np.zeros((p, p))
    for lab in labels:
        Xi = X[groups == lab]
        ni = len(Xi)
        Si = np.cov(Xi, rowvar=False)
        if Si.ndim == 0:
            Si = Si.reshape(1, 1)
        ns.append(ni)
        Ss.append(Si)
        Sp += (ni - 1) * Si

    N = sum(ns)
    Sp /= N - k

    log_det_Sp = np.log(max(np.linalg.det(Sp), 1e-300))
    M = 0.0
    for i, (ni, Si) in enumerate(zip(ns, Ss)):
        log_det_Si = np.log(max(np.linalg.det(Si), 1e-300))
        M += (ni - 1) * (log_det_Sp - log_det_Si)

    c1 = sum(1 / (ni - 1) for ni in ns) - 1 / (N - k)
    c1 *= (2 * p**2 + 3 * p - 1) / (6 * (p + 1) * (k - 1))
    M_adj = M * (1 - c1)

    df = p * (p + 1) * (k - 1) / 2
    p_val = float(1 - sp_stats.chi2.cdf(M_adj, df))

    return TestResult(
        test_name="Box M",
        statistic=float(M_adj),
        p_value=p_val,
        df=float(df),
        method="Box M chi-square approximation",
        n=N,
        extra={"M_raw": float(M), "k": k, "p": p},
    )


box_m = box_m_test


def cheatsheet() -> str:
    return "box_m_test({}) -> Box's M test for equality of covariance matrices."
