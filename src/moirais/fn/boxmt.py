# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Box's M test for equality of covariance matrices."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import DescriptiveResult


def box_m_test(data: np.ndarray, groups: np.ndarray, cdf=None) -> DescriptiveResult:
    """Box's M test for homogeneity of covariance matrices.

    Parameters
    ----------
    data : ndarray (n, p)
        Data matrix.
    groups : ndarray (n,)
        Group labels.

    Returns
    -------
    DescriptiveResult
        ``value`` is the M statistic.
        ``extra`` has ``chi2``, ``df``, ``p_value``.
    """
    X = np.asarray(data, dtype=np.float64)
    g = np.asarray(groups)
    n, p = X.shape
    classes = np.unique(g)
    k = len(classes)

    ns = []
    covs = []
    Sp = np.zeros((p, p))

    for c in classes:
        Xc = X[g == c]
        nc = Xc.shape[0]
        ns.append(nc)
        Sc = np.cov(Xc, rowvar=False, ddof=1)
        covs.append(Sc)
        Sp += (nc - 1) * Sc

    N = sum(ns)
    Sp /= (N - k)

    _, logdet_Sp = np.linalg.slogdet(Sp + np.eye(p) * 1e-12)
    M = 0.0
    for i, c in enumerate(classes):
        ni = ns[i]
        _, logdet_Si = np.linalg.slogdet(covs[i] + np.eye(p) * 1e-12)
        M += (ni - 1) * (logdet_Sp - logdet_Si)

    c1 = (2 * p ** 2 + 3 * p - 1) / (6 * (p + 1) * (k - 1))
    c1 *= sum(1.0 / (ni - 1) for ni in ns) - 1.0 / (N - k)

    chi2 = (1 - c1) * M
    df = p * (p + 1) * (k - 1) // 2
    p_value = float(1 - sp_stats.chi2.cdf(chi2, df))

    return DescriptiveResult(
        name="BoxM",
        value=float(M),
        extra={
            "chi2": float(chi2),
            "df": df,
            "p_value": p_value,
        },
    )


boxmt = box_m_test


def cheatsheet() -> str:
    return "box_m_test({}) -> Box's M test for equality of covariances."
