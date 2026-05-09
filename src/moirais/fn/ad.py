# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Anderson-Darling test for normality."""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._containers import TestResult


def anderson_darling(x) -> TestResult:
    """Anderson-Darling test for normality.

    Uses scipy.stats.anderson against the normal distribution.
    The p-value is approximate, based on the 5% critical value.

    :param x: array-like of observations.
    :return: TestResult with A-D statistic and approximate p-value.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    result = stats.anderson(x, dist="norm")
    # Use 5% significance level (index 2)
    crit_5 = result.critical_values[2] if len(result.critical_values) > 2 else 0
    p_approx = 0.05 if result.statistic > crit_5 else 0.10
    return TestResult(
        test_name="Anderson-Darling",
        statistic=float(result.statistic),
        p_value=p_approx,
        n=len(x),
        method="Anderson-Darling test for normality",
        extra={
            "critical_values": list(result.critical_values),
            "significance_levels": list(result.significance_level),
        },
    )


ad = anderson_darling


def cheatsheet() -> str:
    return "anderson_darling({}) -> Anderson-Darling test for normality."
