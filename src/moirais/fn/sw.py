"""Shapiro-Wilk test for normality."""

from typing import Union

import numpy as np
import scipy.stats as stats


def shapiro_wilk_test(x: Union[list, np.ndarray]) -> dict:
    """
    Shapiro-Wilk test for normality.

    Tests H0: the sample comes from a normal distribution.
    Recommended for n <= 2000; use Kolmogorov-Smirnov for larger samples.

    :param x: Sample data (1-D array-like, 3 <= n <= 5000).
    :return: dict with keys ``W``, ``p_value``, ``is_normal``
        (True if p_value > 0.05).
    :raises ValueError: If x has fewer than 3 observations.

    References
    ----------
    Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test for
        normality. Biometrika, 52(3-4), 591-611.
    """
    arr = np.asarray(x, dtype=float)
    if len(arr) < 3:
        raise ValueError("Shapiro-Wilk test requires at least 3 observations.")
    w_stat, p_val = stats.shapiro(arr)
    return {
        "W": float(w_stat),
        "p_value": float(p_val),
        "is_normal": float(p_val) > 0.05,
        "method": "Shapiro-Wilk test",
    }


sw = shapiro_wilk_test


def cheatsheet() -> str:
    return "shapiro_wilk_test({}) -> Shapiro-Wilk test for normality."
