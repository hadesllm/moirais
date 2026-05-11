# morie.fn — function file (hadesllm/morie)
"""Compute event firing rate over time windows."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Rebellions are built on hope."


def event_rate(events, duration=1.0, window=1.0, **kwargs) -> DescriptiveResult:
    """Compute event firing rate over time windows.

    Parameters
    ----------
    events : array-like
        Event times.
    duration : float
        Total observation duration. Default 1.0.
    window : float
        Window size for rate estimation. Default 1.0.

    Returns
    -------
    DescriptiveResult
    """
    events = np.asarray(events, dtype=float)
    events = np.sort(events)

    overall_rate = float(len(events)) / duration if duration > 0 else 0.0

    bin_edges = np.arange(0, duration + window, window)
    counts, _ = np.histogram(events, bins=bin_edges)
    rates = counts.astype(float) / window

    return DescriptiveResult(
        name="event_rate",
        value=overall_rate,
        extra={
            "rates": rates,
            "bin_edges": bin_edges,
            "counts": counts,
            "overall_rate": overall_rate,
            "n_events": len(events),
            "duration": duration,
            "window": window,
        },
    )


evtrt = event_rate


def cheatsheet() -> str:
    return "event_rate({}) -> Compute event firing rate over time windows."
