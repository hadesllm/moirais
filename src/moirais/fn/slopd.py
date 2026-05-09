"""Slope-based onset/offset detection."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "We suffer more often in imagination than in reality. — Seneca"


def slope_detect(x, fs: float = 1.0, threshold: float | None = None, **kwargs) -> DescriptiveResult:
    """Detect onset/offset events based on signal slope (first derivative).

    Parameters
    ----------
    x : array-like
        Input signal.
    fs : float
        Sampling frequency in Hz.
    threshold : float or None
        Slope threshold. If None, uses 2 * std of derivative.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    dx = np.diff(x) * fs
    if threshold is None:
        threshold = 2.0 * np.std(dx) if len(dx) > 0 else 1.0
    onsets = []
    offsets = []
    active = False
    for i in range(len(dx)):
        if not active and dx[i] > threshold:
            onsets.append(i)
            active = True
        elif active and dx[i] < -threshold:
            offsets.append(i)
            active = False
    return DescriptiveResult(
        name="slope_detect",
        value=float(len(onsets)),
        extra={
            "onsets": np.array(onsets, dtype=int),
            "offsets": np.array(offsets, dtype=int),
            "derivative": dx,
            "threshold": threshold,
            "fs": fs,
        },
    )


slopd = slope_detect


def cheatsheet() -> str:
    return "slope_detect({}) -> Slope-based onset/offset detection."
