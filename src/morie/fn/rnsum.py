# morie.fn -- function file (hadesllm/morie)
"""Wald-Wolfowitz runs test for randomness."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import TestResult


def runs_test(x: np.ndarray) -> TestResult:
    """Wald-Wolfowitz runs test.

    Dichotomises at the median and counts runs.

    Parameters
    ----------
    x : array-like

    Returns
    -------
    TestResult
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        raise ValueError("Need >= 10 observations.")

    med = np.median(x)
    binary = (x >= med).astype(int)
    n1 = int(binary.sum())
    n0 = n - n1

    runs = 1 + int(np.sum(np.diff(binary) != 0))

    mu = 1 + 2 * n0 * n1 / n
    var = 2 * n0 * n1 * (2 * n0 * n1 - n) / (n**2 * (n - 1))
    z = (runs - mu) / np.sqrt(var) if var > 0 else 0.0
    p = float(2 * sp_stats.norm.sf(abs(z)))

    return TestResult(
        test_name="Runs",
        statistic=float(z),
        p_value=p,
        method="Wald-Wolfowitz runs test",
        n=n,
        extra={"runs": runs, "n0": n0, "n1": n1, "expected_runs": float(mu)},
    )


rnsum = runs_test


def cheatsheet() -> str:
    return "runs_test({}) -> Wald-Wolfowitz runs test for randomness."
