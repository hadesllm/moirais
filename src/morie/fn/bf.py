# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Brown-Forsythe test for equality of variances."""

from __future__ import annotations

from scipy import stats

from ._containers import TestResult


def brown_forsythe(*groups) -> TestResult:
    """Brown-Forsythe test (Levene with median). Tests homogeneity of variances.

    :param groups: Two or more arrays of observations.
    :return: TestResult with F-statistic and p-value.
    :raises ValueError: If fewer than 2 groups provided.
    """
    if len(groups) < 2:
        raise ValueError("Need at least 2 groups")
    stat_val, p_val = stats.levene(*groups, center="median")
    n = sum(len(g) for g in groups)
    k = len(groups)
    return TestResult(
        test_name="Brown-Forsythe",
        statistic=float(stat_val),
        p_value=float(p_val),
        df=float(k - 1),
        n=n,
        method="Brown-Forsythe test (Levene with median)",
    )


bf = brown_forsythe


def cheatsheet() -> str:
    return "brown_forsythe({}) -> Brown-Forsythe test for equality of variances."
