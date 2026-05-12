# morie.fn -- function file (hadesllm/morie)
"""Mauchly's sphericity test."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import DescriptiveResult


def mauchly_test(data: np.ndarray, cdf=None) -> DescriptiveResult:
    """Mauchly's test of sphericity for repeated measures.

    Tests whether the covariance matrix of the differences between
    conditions is proportional to an identity matrix.

    Parameters
    ----------
    data : ndarray (n, k)
        Data matrix: n subjects x k conditions.

    Returns
    -------
    DescriptiveResult
        ``value`` is the W statistic.
        ``extra`` has ``chi2``, ``df``, ``p_value``, ``epsilon_gg``
        (Greenhouse-Geisser correction).
    """
    X = np.asarray(data, dtype=np.float64)
    n, k = X.shape

    C = np.eye(k) - np.ones((k, k)) / k
    C = C[:, :k - 1]

    Y = X @ C
    S = np.cov(Y, rowvar=False, ddof=1)
    p = k - 1

    det_S = np.linalg.det(S)
    trace_S = np.trace(S)
    W = det_S / (trace_S / p) ** p if trace_S > 0 else 0.0

    f = 1.0 - (2 * p ** 2 + p + 2) / (6 * p * (n - 1))
    chi2 = -f * (n - 1) * np.log(max(W, 1e-300))
    df = p * (p + 1) // 2 - 1
    p_value = float(1 - sp_stats.chi2.cdf(chi2, max(df, 1)))

    eigvals = np.linalg.eigvalsh(S)
    eigvals = np.maximum(eigvals, 0)
    eps_gg = np.sum(eigvals) ** 2 / (p * np.sum(eigvals ** 2)) if np.sum(eigvals ** 2) > 0 else 1.0

    return DescriptiveResult(
        name="MauchlySphericity",
        value=float(W),
        extra={
            "chi2": float(chi2),
            "df": df,
            "p_value": p_value,
            "epsilon_gg": float(eps_gg),
        },
    )


mrtsq = mauchly_test


def cheatsheet() -> str:
    return "mauchly_test({}) -> Mauchly's test of sphericity."
