"""Response time distribution analysis."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import DescriptiveResult


def tps_response_time(
    times: np.ndarray | list[float],
) -> DescriptiveResult:
    """Analyse police response time distribution.

    Parameters
    ----------
    times : array-like
        Response times in minutes.

    Returns
    -------
    DescriptiveResult
    """
    t = np.asarray(times, dtype=float)
    t = t[np.isfinite(t)]
    if len(t) == 0:
        raise ValueError("No valid response times")
    return DescriptiveResult(
        name="response_time",
        value=float(np.median(t)),
        extra={
            "mean": float(np.mean(t)),
            "median": float(np.median(t)),
            "std": float(np.std(t, ddof=1)) if len(t) > 1 else 0.0,
            "p90": float(np.percentile(t, 90)),
            "p95": float(np.percentile(t, 95)),
            "min": float(np.min(t)),
            "max": float(np.max(t)),
            "n": len(t),
        },
    )


tpsrs = tps_response_time


def cheatsheet() -> str:
    return "tps_response_time({}) -> Response time distribution analysis."
