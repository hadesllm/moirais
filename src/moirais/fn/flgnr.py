# moirais.fn — function file (hadesllm/moirais)
"""Fligner-Killeen test for homogeneity of variance."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import TestResult


def fligner_killeen(*groups: np.ndarray) -> TestResult:
    """Fligner-Killeen median test for equal variances.

    Non-parametric alternative to Bartlett's test.

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
    groups = [np.asarray(g, dtype=float) for g in groups]
    stat, p = sp_stats.fligner(*groups)
    n = sum(len(g) for g in groups)

    return TestResult(
        test_name="Fligner-Killeen",
        statistic=float(stat),
        p_value=float(p),
        df=float(len(groups) - 1),
        method="Fligner-Killeen",
        n=n,
        extra={"k": len(groups)},
    )


flgnr = fligner_killeen


def cheatsheet() -> str:
    return "fligner_killeen({}) -> Fligner-Killeen test for homogeneity of variance."
