# morie.fn -- function file (hadesllm/morie)
"""Fundamental frequency estimation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Errors using inadequate data are much less than those using none. -- Charles Babbage"


def fundamental_freq(x, fs: float = 1.0, **kwargs) -> DescriptiveResult:
    """Estimate the fundamental frequency (F0) via autocorrelation.

    Finds the first significant peak of the autocorrelation function
    beyond the zero-lag origin.

    Parameters
    ----------
    x : array-like
        Input signal.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    N = len(x)
    acf = np.correlate(x, x, mode="full")
    acf = acf[N - 1 :]
    acf = acf / (acf[0] if acf[0] != 0 else 1.0)
    diff = np.diff(acf)
    min_lag = max(2, int(fs / (fs / 2.0))) if fs > 0 else 2
    peak_idx = None
    for i in range(min_lag, len(diff)):
        if diff[i - 1] > 0 and diff[i] <= 0:
            peak_idx = i
            break
    if peak_idx is None:
        f0 = 0.0
    else:
        f0 = float(fs / peak_idx)
    return DescriptiveResult(
        name="fundamental_freq",
        value=f0,
        extra={"f0": f0, "peak_lag": peak_idx, "fs": fs},
    )


fndmn = fundamental_freq


def cheatsheet() -> str:
    return "fundamental_freq({}) -> Fundamental frequency estimation."
