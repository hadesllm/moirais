# morie.fn — function file (hadesllm/morie)
"""Measure fall time of detected events."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Impressive. Most impressive."


def fall_time_detect(x, events, threshold=0.1, **kwargs) -> DescriptiveResult:
    """Measure fall time of detected events.

    Fall time is measured from (1 - *threshold*) to *threshold* of peak amplitude
    after the peak for each event.

    Parameters
    ----------
    x : array-like
        Input signal.
    events : list of tuples (start, end)
        Event boundaries as ``(start_index, end_index)`` pairs.
    threshold : float
        Fraction of peak for fall measurement. Default 0.1.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    fall_times = []

    for start, end in events:
        start, end = int(start), int(end)
        seg = x[start : end + 1]
        if len(seg) < 2:
            fall_times.append(0.0)
            continue
        peak_idx = int(np.argmax(seg))
        tail = seg[peak_idx:]
        if len(tail) < 2:
            fall_times.append(0.0)
            continue
        peak = seg[peak_idx]
        base = np.min(tail)
        amp = peak - base
        if amp <= 0:
            fall_times.append(0.0)
            continue
        hi = base + (1 - threshold) * amp
        lo = base + threshold * amp
        hi_idx = np.argmax(tail <= hi)
        lo_idx = np.argmax(tail <= lo)
        fall_times.append(float(lo_idx - hi_idx))

    fall_times = np.array(fall_times)
    mean_fall = float(np.mean(fall_times)) if len(fall_times) > 0 else 0.0

    return DescriptiveResult(
        name="fall_time_detect",
        value=mean_fall,
        extra={
            "fall_times": fall_times,
            "mean": mean_fall,
            "n_events": len(events),
            "threshold": threshold,
        },
    )


faldt = fall_time_detect


def cheatsheet() -> str:
    return "fall_time_detect({}) -> Measure fall time of detected events."
