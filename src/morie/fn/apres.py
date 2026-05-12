# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Aggregate PRE across roll calls."""

from __future__ import annotations

from ._containers import DescriptiveResult


def apre_statistic(all_pre) -> DescriptiveResult:
    """Aggregate proportional reduction in error across multiple roll calls.

    .. epigraph:: "Yeah science!" -- Jesse, Breaking Bad
    """
    import numpy as np

    pre_vals = np.asarray(all_pre, dtype=float)
    apre = float(np.mean(pre_vals))
    return DescriptiveResult(
        name="apre_statistic",
        value=apre,
        extra={
            "mean_pre": apre,
            "median_pre": float(np.median(pre_vals)),
            "min_pre": float(np.min(pre_vals)),
            "max_pre": float(np.max(pre_vals)),
            "n_roll_calls": len(pre_vals),
        },
    )


apres = apre_statistic


def cheatsheet() -> str:
    return "apre_statistic({}) -> Aggregate PRE across roll calls."
