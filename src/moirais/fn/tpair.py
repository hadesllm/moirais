"""Paired samples t-test."""

from typing import Union

import numpy as np
import scipy.stats as stats


def paired_t_test(
    x1: Union[list, np.ndarray],
    x2: Union[list, np.ndarray],
    *,
    alternative: str = "two-sided",
) -> dict:
    """
    Paired samples t-test.

    Tests H0: mean(x1 - x2) = 0. Requires equal-length samples.

    :param x1: First sample (1-D array-like).
    :param x2: Second sample (same length as x1).
    :param alternative: ``"two-sided"``, ``"greater"``, or ``"less"``.
    :return: dict with keys ``t``, ``df``, ``p_value``, ``mean_diff``,
        ``se_diff``, ``ci_lower``, ``ci_upper``.
    :raises ValueError: If samples have different lengths or fewer than 2 pairs.

    References
    ----------
    Student (1908). The probable error of a mean. Biometrika, 6(1), 1-25.
    """
    a1 = np.asarray(x1, dtype=float)
    a2 = np.asarray(x2, dtype=float)
    if len(a1) != len(a2):
        raise ValueError("x1 and x2 must have the same length for a paired test.")
    if len(a1) < 2:
        raise ValueError("At least 2 pairs are required.")
    valid_alternatives = {"two-sided", "greater", "less"}
    if alternative not in valid_alternatives:
        raise ValueError(f"alternative must be one of {valid_alternatives}.")
    diff = a1 - a2
    t_stat, p_val = stats.ttest_rel(a1, a2, alternative=alternative)
    n = len(diff)
    mean_diff = float(np.mean(diff))
    se_diff = float(np.std(diff, ddof=1) / np.sqrt(n))
    t_crit = float(stats.t(df=n - 1).ppf(0.975))
    return {
        "t": float(t_stat),
        "df": float(n - 1),
        "p_value": float(p_val),
        "mean_diff": mean_diff,
        "se_diff": se_diff,
        "ci_lower": mean_diff - t_crit * se_diff,
        "ci_upper": mean_diff + t_crit * se_diff,
        "method": "Paired t-test",
    }


tpair = paired_t_test


def cheatsheet() -> str:
    return "paired_t_test({}) -> Paired samples t-test."
