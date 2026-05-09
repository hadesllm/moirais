# moirais.fn — function file (hadesllm/moirais)
"""Grubbs' test for a single outlier."""

from typing import Union

import numpy as np
import scipy.stats as stats

from ._containers import TestResult


def grubbs_test(
    x: Union[list, np.ndarray],
    *,
    alpha: float = 0.05,
) -> TestResult:
    """
    Grubbs' test for detecting a single outlier.

    Tests whether the value farthest from the sample mean is an outlier.

    .. math::

        G = \\frac{\\max_i |x_i - \\bar{x}|}{s}

    The critical value is derived from the t-distribution:

    .. math::

        G_{\\text{crit}} = \\frac{(n-1)}{\\sqrt{n}}
        \\sqrt{\\frac{t^2_{\\alpha/(2n),\\,n-2}}{n - 2 + t^2_{\\alpha/(2n),\\,n-2}}}

    :param x: Sample data (1-D array-like, n >= 3).
    :param alpha: Significance level. Default 0.05.
    :return: TestResult with G statistic and two-sided p_value.
    :raises ValueError: If x has fewer than 3 observations.

    References
    ----------
    Grubbs, F. E. (1950). Sample criteria for testing outlying observations.
        Annals of Mathematical Statistics, 21(1), 27-58.
    """
    arr = np.asarray(x, dtype=float)
    n = len(arr)
    if n < 3:
        raise ValueError("Grubbs' test requires at least 3 observations.")

    mean = np.mean(arr)
    sd = np.std(arr, ddof=1)
    if sd == 0:
        return TestResult(
            test_name="Grubbs",
            statistic=0.0,
            p_value=1.0,
            method="Grubbs' test (zero variance)",
            n=n,
        )

    abs_dev = np.abs(arr - mean)
    G = float(np.max(abs_dev) / sd)
    outlier_idx = int(np.argmax(abs_dev))

    # Two-sided p-value via t-distribution
    # Under H0, T^2 = G^2 * (n-2) / (n-1-G^2) ~ F(1, n-2)
    denom = n - 1 - G**2
    if denom <= 0:
        # G too large for this formula (extreme outlier)
        p_value = 0.0
    else:
        t2 = G**2 * (n - 2) / denom
        t_val = np.sqrt(max(t2, 0.0))
        p_single = float(stats.t.sf(t_val, df=n - 2))
        p_value = min(float(n * p_single), 1.0)

    return TestResult(
        test_name="Grubbs",
        statistic=G,
        p_value=p_value,
        df=n - 2,
        method="Grubbs' test (two-sided)",
        n=n,
        extra={"outlier_index": outlier_idx, "outlier_value": float(arr[outlier_idx])},
    )


grubs = grubbs_test


def cheatsheet() -> str:
    return "grubbs_test({}) -> Grubbs' test for a single outlier."
