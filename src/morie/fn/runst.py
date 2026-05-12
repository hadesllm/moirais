# morie.fn -- function file (hadesllm/morie)
"""Wald-Wolfowitz runs test for randomness."""

from __future__ import annotations

import numpy as np

from ._containers import TestResult


def runs_test(x) -> TestResult:
    """Wald-Wolfowitz runs test for randomness.

    Dichotomises the sample at the median, then counts runs
    (consecutive sequences above/below).

    Parameters
    ----------
    x : array-like
        Observations.

    Returns
    -------
    TestResult
    """
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    n = len(a)
    if n < 2:
        raise ValueError("Need at least 2 finite observations.")

    med = np.median(a)
    binary = (a >= med).astype(int)

    n1 = int(np.sum(binary))
    n0 = n - n1
    if n0 == 0 or n1 == 0:
        return TestResult(
            test_name="Runs test",
            statistic=1.0,
            p_value=1.0,
            n=n,
            method="all values on one side of median",
        )

    runs = 1 + int(np.sum(np.diff(binary) != 0))

    mu = 1.0 + 2.0 * n0 * n1 / n
    var = (2.0 * n0 * n1 * (2.0 * n0 * n1 - n)) / (n**2 * (n - 1.0))
    if var <= 0:
        return TestResult(
            test_name="Runs test", statistic=float(runs), p_value=1.0, n=n
        )

    z = (runs - mu) / np.sqrt(var)
    from scipy.stats import norm

    pval = 2.0 * norm.sf(abs(z))

    return TestResult(
        test_name="Runs test",
        statistic=float(z),
        p_value=float(pval),
        n=n,
        method="normal approximation",
        extra={"runs": runs, "n_above": n1, "n_below": n0, "expected_runs": float(mu)},
    )


runst = runs_test


def cheatsheet() -> str:
    return "runs_test(x) -> Wald-Wolfowitz runs test for randomness."
