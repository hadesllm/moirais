# morie.fn — function file (hadesllm/morie)
"""Schoenfeld residuals test for PH assumption."""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._containers import TestResult


def schoenfeld_test(
    residuals: np.ndarray | list,
    times: np.ndarray | list,
) -> TestResult:
    """
    Test the proportional hazards assumption using Schoenfeld residuals.

    Tests correlation between scaled Schoenfeld residuals and time.
    A significant result suggests violation of PH assumption.

    Parameters
    ----------
    residuals : array-like
        Schoenfeld residuals (one per event).
    times : array-like
        Event times corresponding to residuals.

    Returns
    -------
    TestResult

    References
    ----------
    Grambsch, P. M., & Therneau, T. M. (1994). Proportional hazards
    tests and diagnostics based on weighted residuals. *Biometrika*,
    81(3), 515-526.
    """
    r = np.asarray(residuals, dtype=float)
    t = np.asarray(times, dtype=float)
    if len(r) != len(t):
        raise ValueError("residuals and times must match.")
    if len(r) < 3:
        raise ValueError("Need at least 3 residuals.")

    rho, p_val = stats.spearmanr(t, r)

    n = len(r)
    chi2 = rho**2 * n

    return TestResult(
        test_name="schoenfeld",
        statistic=float(chi2),
        p_value=float(p_val),
        df=1,
        method="Grambsch-Therneau",
        n=n,
        extra={"rho": float(rho)},
    )


schfd = schoenfeld_test


def cheatsheet() -> str:
    return "schoenfeld_test({}) -> Schoenfeld residuals test for PH assumption."
