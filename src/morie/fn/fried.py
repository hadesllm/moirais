# morie.fn — function file (hadesllm/morie)
"""Friedman test for repeated measures."""

from __future__ import annotations

from scipy import stats

from ._containers import TestResult


def friedman(*groups) -> TestResult:
    """Friedman chi-squared test for repeated measures (non-parametric).

    Requires at least 3 groups of equal length (repeated measures).

    :param groups: Three or more equal-length arrays of observations.
    :return: TestResult with chi-squared statistic and p-value.
    :raises ValueError: If fewer than 3 groups provided.
    """
    if len(groups) < 3:
        raise ValueError("Need at least 3 groups")
    stat_val, p_val = stats.friedmanchisquare(*groups)
    k = len(groups)
    return TestResult(
        test_name="Friedman",
        statistic=float(stat_val),
        p_value=float(p_val),
        df=float(k - 1),
        n=len(groups[0]),
        method="Friedman chi-squared test",
    )


fried = friedman


def cheatsheet() -> str:
    return "friedman({}) -> Friedman test for repeated measures."
