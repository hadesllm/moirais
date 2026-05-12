# morie.fn -- function file (hadesllm/morie)
"""Double-threshold detection (hysteresis)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Never tell me the odds."


def double_threshold(x, low=0.3, high=0.7, **kwargs) -> DescriptiveResult:
    """Double-threshold detection with hysteresis.

    Signal must cross *high* to start an event and drop below *low* to end it.

    Parameters
    ----------
    x : array-like
        Input signal.
    low : float
        Lower threshold. Default 0.3.
    high : float
        Upper threshold. Default 0.7.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    active = False
    events = []
    start = 0

    for i in range(n):
        if not active and x[i] >= high:
            active = True
            start = i
        elif active and x[i] < low:
            active = False
            events.append((start, i))

    if active:
        events.append((start, n - 1))

    mask = np.zeros(n, dtype=bool)
    for s, e in events:
        mask[s : e + 1] = True

    return DescriptiveResult(
        name="double_threshold",
        value=float(len(events)),
        extra={
            "events": events,
            "mask": mask,
            "n_events": len(events),
            "low": low,
            "high": high,
        },
    )


dbldt = double_threshold


def cheatsheet() -> str:
    return "double_threshold({}) -> Double-threshold detection (hysteresis)."
