# morie.fn — function file (hadesllm/morie)
"""Measure rise time of detected events."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "We are what they grow beyond."


def rise_time_detect(x, events, threshold=0.1, **kwargs) -> DescriptiveResult:
    """Measure rise time of detected events.

    Rise time is measured from *threshold* to (1 - *threshold*) of peak amplitude
    for each event.

    Parameters
    ----------
    x : array-like
        Input signal.
    events : list of tuples (start, end)
        Event boundaries as ``(start_index, end_index)`` pairs.
    threshold : float
        Fraction of peak for rise measurement. Default 0.1.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    rise_times = []

    for start, end in events:
        start, end = int(start), int(end)
        seg = x[start : end + 1]
        if len(seg) < 2:
            rise_times.append(0.0)
            continue
        peak = np.max(seg)
        base = np.min(seg)
        amp = peak - base
        if amp <= 0:
            rise_times.append(0.0)
            continue
        lo = base + threshold * amp
        hi = base + (1 - threshold) * amp
        lo_idx = np.argmax(seg >= lo)
        hi_idx = np.argmax(seg >= hi)
        rise_times.append(float(hi_idx - lo_idx))

    rise_times = np.array(rise_times)
    mean_rise = float(np.mean(rise_times)) if len(rise_times) > 0 else 0.0

    return DescriptiveResult(
        name="rise_time_detect",
        value=mean_rise,
        extra={
            "rise_times": rise_times,
            "mean": mean_rise,
            "n_events": len(events),
            "threshold": threshold,
        },
    )


risdt = rise_time_detect


def cheatsheet() -> str:
    return "rise_time_detect({}) -> Measure rise time of detected events."
