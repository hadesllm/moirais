# morie.fn -- function file (hadesllm/morie)
"""Rational resampling."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult

_QUOTE = "Let the past die. Kill it, if you have to."


def resample_signal(x, up: int, down: int) -> SignalResult:
    """Resample signal by rational factor up/down.

    Parameters
    ----------
    x : array-like
        Input signal.
    up : int
        Upsampling factor.
    down : int
        Downsampling factor.

    Returns
    -------
    SignalResult
    """
    from scipy.signal import resample_poly

    x = np.asarray(x, dtype=float)
    y = resample_poly(x, up, down)
    return SignalResult(
        name="resample_signal",
        filtered=y,
        fs=0.0,
        n_samples=len(y),
        extra={"up": up, "down": down},
    )


rsmpl = resample_signal


def cheatsheet() -> str:
    return "resample_signal({}) -> Rational resampling."
