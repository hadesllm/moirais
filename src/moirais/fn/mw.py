# moirais.fn — function file (hadesllm/moirais)
"""Mann-Whitney U test (Wilcoxon rank-sum test)."""

from typing import Union

import numpy as np
import scipy.stats as stats


def mann_whitney_test(
    x1: Union[list, np.ndarray],
    x2: Union[list, np.ndarray],
    *,
    alternative: str = "two-sided",
) -> dict:
    """
    Mann-Whitney U test (Wilcoxon rank-sum test).

    Non-parametric alternative to the two-sample t-test for ordinal or
    non-normally distributed data.

    :param x1: First sample (1-D array-like).
    :param x2: Second sample (1-D array-like).
    :param alternative: ``"two-sided"``, ``"greater"``, or ``"less"``.
    :return: dict with keys ``U``, ``p_value``.
    :raises ValueError: If x1 or x2 is empty or alternative is invalid.

    References
    ----------
    Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random
        variables is stochastically larger than the other. Annals of Mathematical
        Statistics, 18(1), 50-60.
    """
    a1 = np.asarray(x1, dtype=float)
    a2 = np.asarray(x2, dtype=float)
    if len(a1) < 1:
        raise ValueError("x1 must not be empty.")
    if len(a2) < 1:
        raise ValueError("x2 must not be empty.")
    valid_alternatives = {"two-sided", "greater", "less"}
    if alternative not in valid_alternatives:
        raise ValueError(f"alternative must be one of {valid_alternatives}.")
    u_stat, p_val = stats.mannwhitneyu(a1, a2, alternative=alternative)
    return {
        "U": float(u_stat),
        "p_value": float(p_val),
        "method": "Mann-Whitney U test",
    }


mw = mann_whitney_test


def cheatsheet() -> str:
    return "mann_whitney_test({}) -> Mann-Whitney U test (Wilcoxon rank-sum test)."
