# morie.fn -- function file (hadesllm/morie)
"""Heart rate variability metrics."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def hrv_metrics_fn(
    rr_intervals: np.ndarray,
) -> DescriptiveResult:
    """We are what we repeatedly do. Excellence is not an act, but a habit. -- Aristotle"""
    from morie._bioplot import hrv_metrics

    rr_intervals = np.asarray(rr_intervals, dtype=float)
    metrics = hrv_metrics(rr_intervals)
    return DescriptiveResult(
        name="hrv_metrics",
        value=metrics["sdnn"],
        extra=metrics,
    )


hrvmt = hrv_metrics_fn


def cheatsheet() -> str:
    return "hrv_metrics_fn({}) -> Heart rate variability metrics."
