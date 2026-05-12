# morie.fn -- function file (hadesllm/morie)
"""Mood's median test."""

from typing import Union

import numpy as np
import scipy.stats as stats

from ._containers import TestResult


def mood_median_test(
    *groups: Union[list, np.ndarray],
) -> TestResult:
    """
    Mood's median test for comparing medians of two or more groups.

    Counts observations above and at/below the grand median and performs
    a chi-square test on the resulting contingency table.

    :param groups: Two or more array-like groups of observations.
    :return: TestResult with chi2 statistic, p_value, grand_median.
    :raises ValueError: If fewer than 2 groups provided.

    References
    ----------
    Mood, A. M. (1954). On the asymptotic efficiency of certain nonparametric
        two-sample tests. Annals of Mathematical Statistics, 25(3), 514-522.
    """
    if len(groups) < 2:
        raise ValueError("Mood's median test requires at least 2 groups.")

    arrays = [np.asarray(g, dtype=float) for g in groups]
    combined = np.concatenate(arrays)
    grand_median = float(np.median(combined))
    n = len(combined)

    # Build contingency table: rows = above/at-or-below, cols = groups
    table = np.zeros((2, len(arrays)), dtype=int)
    for j, arr in enumerate(arrays):
        table[0, j] = int(np.sum(arr > grand_median))
        table[1, j] = int(np.sum(arr <= grand_median))

    # If all values equal median, chi2 is undefined
    if table[0].sum() == 0 or table[1].sum() == 0:
        return TestResult(
            test_name="Mood median",
            statistic=0.0,
            p_value=1.0,
            df=len(arrays) - 1,
            method="Mood's median test (degenerate)",
            n=n,
            extra={"grand_median": grand_median},
        )

    chi2, p_val, df, _ = stats.chi2_contingency(table)

    return TestResult(
        test_name="Mood median",
        statistic=float(chi2),
        p_value=float(p_val),
        df=int(df),
        method="Mood's median test",
        n=n,
        extra={"grand_median": grand_median, "k": len(arrays)},
    )


mood = mood_median_test


def cheatsheet() -> str:
    return "mood_median_test({}) -> Mood's median test."
