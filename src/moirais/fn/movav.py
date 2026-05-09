# moirais.fn — function file (hadesllm/moirais)
"""Moving average filter."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult


def moving_average(x, window: int = 5) -> SignalResult:
    """Apply a moving average filter to signal *x*.

    Parameters
    ----------
    x : array-like
        Input signal.
    window : int
        Window size. Default 5.

    Returns
    -------
    SignalResult
    """
    from moirais._filters import moving_average as _ma

    x = np.asarray(x, dtype=float)
    result = _ma(x, window=window)
    return SignalResult(name="moving_average", filtered=result, fs=0.0, n_samples=len(x))


movav = moving_average


def cheatsheet() -> str:
    return "moving_average({}) -> Moving average filter."
