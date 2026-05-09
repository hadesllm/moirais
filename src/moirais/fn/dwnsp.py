# moirais.fn — function file (hadesllm/moirais)
"""Downsample a signal."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult

_QUOTE = "Chewie, we're home."


def downsample(x, factor: int) -> SignalResult:
    """Downsample signal by integer factor (keep every factor-th sample).

    Parameters
    ----------
    x : array-like
        Input signal.
    factor : int
        Downsampling factor.

    Returns
    -------
    SignalResult
    """
    x = np.asarray(x, dtype=float)
    factor = int(factor)
    if factor < 1:
        raise ValueError("factor must be >= 1")
    y = x[::factor]
    return SignalResult(
        name="downsample",
        filtered=y,
        fs=0.0,
        n_samples=len(y),
        extra={"factor": factor},
    )


dwnsp = downsample


def cheatsheet() -> str:
    return "downsample({}) -> Downsample a signal."
