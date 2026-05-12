# morie.fn -- function file (hadesllm/morie)
"""Baseline drift magnitude estimation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "So this is how liberty dies, with thunderous applause."


def drift_magnitude(x, window=100, **kwargs) -> DescriptiveResult:
    """Estimate baseline drift magnitude.

    Uses a moving average to estimate the baseline trend and
    reports the peak-to-peak drift.

    Parameters
    ----------
    x : array-like
        Input signal.
    window : int
        Moving average window size.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    if window > len(x):
        window = len(x)
    kernel = np.ones(window) / window
    baseline = np.convolve(x, kernel, mode="same")
    drift = float(np.max(baseline) - np.min(baseline))
    drift_rate = drift / (len(x) - 1) if len(x) > 1 else 0.0
    return DescriptiveResult(
        name="drift_magnitude",
        value=drift,
        extra={
            "drift": drift,
            "drift_rate": float(drift_rate),
            "baseline_mean": float(np.mean(baseline)),
            "window": window,
            "n": len(x),
        },
    )


drfmg = drift_magnitude


def cheatsheet() -> str:
    return "drift_magnitude({}) -> Baseline drift magnitude estimation."
