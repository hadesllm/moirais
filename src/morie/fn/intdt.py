# morie.fn — function file (hadesllm/morie)
"""Detect inter-event intervals and statistics."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The belonging you seek is not behind you, it is ahead."


def interval_detect(events, min_interval=0.1, **kwargs) -> DescriptiveResult:
    """Detect inter-event intervals and compute statistics.

    Parameters
    ----------
    events : array-like
        Sorted event times.
    min_interval : float
        Minimum interval to report. Default 0.1.

    Returns
    -------
    DescriptiveResult
    """
    events = np.asarray(events, dtype=float)
    events = np.sort(events)

    if len(events) < 2:
        raise ValueError("Need at least 2 events for interval analysis.")

    intervals = np.diff(events)
    mask = intervals >= min_interval
    filtered = intervals[mask]

    mean_int = float(np.mean(filtered)) if len(filtered) > 0 else 0.0
    std_int = float(np.std(filtered)) if len(filtered) > 0 else 0.0

    return DescriptiveResult(
        name="interval_detect",
        value=mean_int,
        extra={
            "intervals": intervals,
            "filtered_intervals": filtered,
            "mean": mean_int,
            "std": std_int,
            "n_total": len(intervals),
            "n_kept": len(filtered),
            "min_interval": min_interval,
        },
    )


intdt = interval_detect


def cheatsheet() -> str:
    return "interval_detect({}) -> Detect inter-event intervals and statistics."
