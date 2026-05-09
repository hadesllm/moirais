"""Wilcoxon signed-rank test for paired samples."""

from typing import Union

import numpy as np
import scipy.stats as stats


def wilcoxon_signed_rank_test(
    x1: Union[list, np.ndarray],
    x2: Union[list, np.ndarray],
    *,
    alternative: str = "two-sided",
) -> dict:
    """
    Wilcoxon signed-rank test for paired samples.

    Non-parametric alternative to the paired t-test.

    :param x1: First sample (1-D array-like).
    :param x2: Second sample (same length as x1).
    :param alternative: ``"two-sided"``, ``"greater"``, or ``"less"``.
    :return: dict with keys ``statistic``, ``p_value``.
    :raises ValueError: If samples have different lengths or fewer than 2 pairs.

    References
    ----------
    Wilcoxon, F. (1945). Individual comparisons by ranking methods. Biometrics
        Bulletin, 1(6), 80-83.
    """
    a1 = np.asarray(x1, dtype=float)
    a2 = np.asarray(x2, dtype=float)
    if len(a1) != len(a2):
        raise ValueError("x1 and x2 must have the same length.")
    if len(a1) < 2:
        raise ValueError("At least 2 pairs are required.")
    valid_alternatives = {"two-sided", "greater", "less"}
    if alternative not in valid_alternatives:
        raise ValueError(f"alternative must be one of {valid_alternatives}.")
    w_stat, p_val = stats.wilcoxon(a1, a2, alternative=alternative)
    return {
        "statistic": float(w_stat),
        "p_value": float(p_val),
        "method": "Wilcoxon signed-rank test",
    }


wilcox = wilcoxon_signed_rank_test


def cheatsheet() -> str:
    return "wilcoxon_signed_rank_test({}) -> Wilcoxon signed-rank test for paired samples."
