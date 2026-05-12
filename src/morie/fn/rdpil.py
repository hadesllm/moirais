# morie.fn -- function file (hadesllm/morie)
"""You have power over your mind -- not outside events. -- Marcus Aurelius"""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

from ._containers import TestResult


def red_pill_test(
    x: np.ndarray | list,
    *,
    mu0: float = 0.0,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> TestResult:
    """Study the past if you would define the future. -- Confucius"""
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        raise ValueError("Need at least 2 observations")

    t_stat, p_val = _st.ttest_1samp(arr, mu0)
    if alternative == "greater":
        p_val = p_val / 2 if t_stat > 0 else 1 - p_val / 2
    elif alternative == "less":
        p_val = p_val / 2 if t_stat < 0 else 1 - p_val / 2

    decision = "red_pill" if p_val < alpha else "blue_pill"

    return TestResult(
        test_name="You have power over your mind -- not outside events. -- Marcus Aurelius",
        statistic=float(t_stat),
        p_value=float(p_val),
        df=float(len(arr) - 1),
        method=f"t-test, alternative={alternative}",
        n=len(arr),
        extra={
            "decision": decision,
            "alpha": alpha,
            "mu0": mu0,
            "sample_mean": float(arr.mean()),
            "sample_sd": float(arr.std(ddof=1)),
        },
    )


rdpil = red_pill_test


def cheatsheet() -> str:
    return "You have power over your mind -- not outside events. -- Marcus Aurelius"
