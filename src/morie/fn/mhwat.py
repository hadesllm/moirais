# morie.fn — function file (hadesllm/morie)
"""Wait time for mental health services analysis."""

import numpy as np

from ._containers import DescriptiveResult


def wait_time_analysis(
    wait_times: list | np.ndarray,
    benchmark_days: float = 30.0,
) -> DescriptiveResult:
    """Analyse wait times for mental health services.

    Parameters
    ----------
    wait_times : array-like
        Wait times in days.
    benchmark_days : float
        Acceptable wait time benchmark (default 30 days).

    Returns
    -------
    DescriptiveResult
    """
    w = np.asarray(wait_times, dtype=float)
    w = w[~np.isnan(w)]
    if len(w) == 0:
        raise ValueError("No valid wait times")

    return DescriptiveResult(
        name="wait_time_analysis",
        value=float(np.median(w)),
        extra={
            "mean": float(np.mean(w)),
            "std": float(np.std(w, ddof=1)) if len(w) > 1 else 0.0,
            "p90": float(np.percentile(w, 90)),
            "pct_over_benchmark": float(np.mean(w > benchmark_days) * 100),
            "benchmark_days": benchmark_days,
            "n": len(w),
        },
    )


mhwat = wait_time_analysis


def cheatsheet() -> str:
    return "wait_time_analysis({}) -> Wait time for mental health services analysis."
