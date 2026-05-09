"""Spectral containment bandwidth."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Never tell me the odds."


def spectral_bound(x, fs=1.0, fraction=0.99, **kwargs) -> DescriptiveResult:
    """Compute the bandwidth containing a given fraction of spectral energy.

    Parameters
    ----------
    x : array-like
        Input signal.
    fs : float
        Sampling frequency.
    fraction : float
        Energy fraction (default 0.99).

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    X = np.fft.rfft(x)
    power = np.abs(X) ** 2
    total_power = np.sum(power)
    if total_power == 0:
        raise ValueError("Signal has zero energy.")
    cum_power = np.cumsum(power) / total_power
    idx = int(np.searchsorted(cum_power, fraction))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    bw = float(freqs[min(idx, len(freqs) - 1)])
    return DescriptiveResult(
        name="spectral_bound",
        value=bw,
        extra={
            "bandwidth_hz": bw,
            "fraction": fraction,
            "total_power": float(total_power),
            "n": len(x),
            "fs": fs,
        },
    )


spcbd = spectral_bound


def cheatsheet() -> str:
    return "spectral_bound({}) -> Spectral containment bandwidth."
