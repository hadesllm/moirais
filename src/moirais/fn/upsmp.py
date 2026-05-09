"""Upsample a signal by zero-insertion."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult

_QUOTE = "The garbage will do!"


def upsample(x, factor: int) -> SignalResult:
    """Upsample signal by integer factor (zero-insertion).

    Parameters
    ----------
    x : array-like
        Input signal.
    factor : int
        Upsampling factor.

    Returns
    -------
    SignalResult
    """
    x = np.asarray(x, dtype=float)
    factor = int(factor)
    if factor < 1:
        raise ValueError("factor must be >= 1")
    y = np.zeros(len(x) * factor)
    y[::factor] = x
    return SignalResult(
        name="upsample",
        filtered=y,
        fs=0.0,
        n_samples=len(y),
        extra={"factor": factor},
    )


upsmp = upsample


def cheatsheet() -> str:
    return "upsample({}) -> Upsample a signal by zero-insertion."
