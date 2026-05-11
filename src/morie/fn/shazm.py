"""SHAZAM simultaneous hypothesis test. 'SHAZAM!' -- Billy Batson"""

from __future__ import annotations

import numpy as np

from ._containers import TestResult


def simultaneous_test(
    p_values: np.ndarray | list[float],
    *,
    method: str = "bonferroni",
    alpha: float = 0.05,
) -> TestResult:
    """Multiple hypothesis testing with family-wise error rate control.

    Adjusts p-values for multiplicity using Bonferroni, Holm, or
    Benjamini-Hochberg (FDR) correction.

    Parameters
    ----------
    p_values : array
        Raw p-values from individual tests.
    method : str
        ``'bonferroni'``, ``'holm'``, or ``'bh'`` (Benjamini-Hochberg).
    alpha : float
        Family-wise significance level.

    Returns
    -------
    TestResult
        Statistic = number of significant tests after correction.
    """
    p = np.asarray(p_values, dtype=float).ravel()
    m = len(p)
    if m == 0:
        raise ValueError("Need at least 1 p-value")
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("p-values must be in [0, 1]")
    if method == "bonferroni":
        adjusted = np.minimum(p * m, 1.0)
    elif method == "holm":
        order = np.argsort(p)
        adjusted = np.empty(m)
        for i, idx in enumerate(order):
            adjusted[idx] = min(p[idx] * (m - i), 1.0)
        cummax = adjusted[order].copy()
        for i in range(1, m):
            cummax[i] = max(cummax[i], cummax[i - 1])
        adjusted[order] = cummax
    elif method == "bh":
        order = np.argsort(p)
        adjusted = np.empty(m)
        for i, idx in enumerate(order):
            adjusted[idx] = min(p[idx] * m / (i + 1), 1.0)
        cummin = adjusted[order].copy()
        for i in range(m - 2, -1, -1):
            cummin[i] = min(cummin[i], cummin[i + 1])
        adjusted[order] = cummin
    else:
        raise ValueError(f"Unknown method: {method}. Use bonferroni/holm/bh.")
    n_sig = int(np.sum(adjusted < alpha))
    return TestResult(
        test_name=f"Simultaneous test ({method})",
        statistic=float(n_sig),
        p_value=float(np.min(adjusted)),
        method=method,
        n=m,
        extra={
            "n_tests": m,
            "n_significant": n_sig,
            "alpha": alpha,
            "adjusted_p": adjusted.tolist(),
            "rejected": (adjusted < alpha).tolist(),
        },
    )


shazm = simultaneous_test


def cheatsheet() -> str:
    return "simultaneous_test({}) -> SHAZAM simultaneous hypothesis test. 'SHAZAM!' -- Billy Bats"
