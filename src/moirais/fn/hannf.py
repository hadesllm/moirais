# moirais.fn — function file (hadesllm/moirais)
"""Hann window filter."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult


def hann_filter(x, window: int = 5) -> SignalResult:
    """Apply a Hann window filter to signal *x*.

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
    from moirais._filters import hann_filter as _hf

    x = np.asarray(x, dtype=float)
    result = _hf(x, window=window)
    return SignalResult(name="hann_filter", filtered=result, fs=0.0, n_samples=len(x), extra={"window": window})


hannf = hann_filter


def cheatsheet() -> str:
    return "hann_filter({}) -> Hann window filter."
