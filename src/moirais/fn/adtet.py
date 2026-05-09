# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Anderson-Darling test for normality."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import TestResult


def anderson_darling_test(
    x,
    *,
    dist: str = "norm",
) -> TestResult:
    """Anderson-Darling test for a specified distribution.

    Parameters
    ----------
    x : array-like
        Observations.
    dist : str
        Distribution to test against (default ``"norm"``).

    Returns
    -------
    TestResult
    """
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 3:
        raise ValueError("Need at least 3 finite observations.")

    result = sp_stats.anderson(a, dist=dist)
    cv_5 = result.critical_values[2] if len(result.critical_values) > 2 else None
    reject = result.statistic > cv_5 if cv_5 is not None else None

    return TestResult(
        test_name="Anderson-Darling test",
        statistic=float(result.statistic),
        p_value=-1.0,
        n=len(a),
        method=f"dist={dist}",
        extra={
            "critical_values": result.critical_values.tolist(),
            "significance_levels": list(result.significance_level),
            "reject_at_5pct": reject,
        },
    )


adtet = anderson_darling_test


def cheatsheet() -> str:
    return "anderson_darling_test(x) -> Anderson-Darling test."
