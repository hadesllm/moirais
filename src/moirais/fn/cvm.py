# moirais.fn — function file (hadesllm/moirais)
"""Cramer-von Mises test."""

from __future__ import annotations

import numpy as np

from ._containers import TestResult


def cramer_von_mises_test(
    x,
    *,
    distribution: str = "norm",
) -> TestResult:
    """Cramer-von Mises goodness-of-fit test.

    Parameters
    ----------
    x : array-like
        Observations.
    distribution : str
        Reference distribution (default ``"norm"``).

    Returns
    -------
    TestResult
    """
    from scipy.stats import cramervonmises

    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 2:
        raise ValueError("Need at least 2 finite observations.")

    result = cramervonmises(a, distribution)
    return TestResult(
        test_name="Cramer-von Mises test",
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        n=len(a),
        method=f"dist={distribution}",
    )


cvm = cramer_von_mises_test


def cheatsheet() -> str:
    return "cramer_von_mises_test(x) -> Cramer-von Mises test."
