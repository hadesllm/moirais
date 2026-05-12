# morie.fn -- function file (hadesllm/morie)
"""Mood's median test."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import TestResult


def mood_median_test(*groups: np.ndarray) -> TestResult:
    """Mood's median test for k independent groups.

    Parameters
    ----------
    *groups : array-like
        Two or more groups.

    Returns
    -------
    TestResult
    """
    if len(groups) < 2:
        raise ValueError("Need >= 2 groups.")
    groups = [np.asarray(g, dtype=float)[np.isfinite(np.asarray(g, dtype=float))] for g in groups]
    all_vals = np.concatenate(groups)
    grand_median = np.median(all_vals)

    above = np.array([np.sum(g > grand_median) for g in groups])
    below = np.array([len(g) - a for g, a in zip(groups, above)])
    table = np.vstack([above, below])

    chi2, p, dof, _ = sp_stats.chi2_contingency(table, correction=False)

    return TestResult(
        test_name="Mood median",
        statistic=float(chi2),
        p_value=float(p),
        df=float(dof),
        method="Mood median test",
        n=len(all_vals),
        extra={"grand_median": float(grand_median), "k": len(groups)},
    )


moodm = mood_median_test


def cheatsheet() -> str:
    return "mood_median_test({}) -> Mood's median test."
