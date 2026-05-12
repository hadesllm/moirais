# morie.fn -- function file (hadesllm/morie)
"""Hysteresis-based event detector."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "I find your lack of faith disturbing."


def hysteresis_detect(x, low=0.3, high=0.7, **kwargs) -> DescriptiveResult:
    """Hysteresis-based event detector.

    Events start when the signal exceeds *high* and end when it falls
    below *low*.

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
    onsets = []
    offsets = []

    for i in range(n):
        if not active and x[i] >= high:
            active = True
            onsets.append(i)
        elif active and x[i] < low:
            active = False
            offsets.append(i)

    if active:
        offsets.append(n - 1)

    durations = [off - on for on, off in zip(onsets, offsets)]

    return DescriptiveResult(
        name="hysteresis_detect",
        value=float(len(onsets)),
        extra={
            "onsets": np.array(onsets, dtype=int),
            "offsets": np.array(offsets, dtype=int),
            "durations": np.array(durations, dtype=int),
            "n_events": len(onsets),
            "low": low,
            "high": high,
        },
    )


hysdt = hysteresis_detect


def cheatsheet() -> str:
    return "hysteresis_detect({}) -> Hysteresis-based event detector."
