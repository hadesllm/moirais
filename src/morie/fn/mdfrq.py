# morie.fn — function file (hadesllm/morie)
"""Median frequency."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "There are no facts, only interpretations. — Friedrich Nietzsche"


def median_frequency(x, fs: float = 1.0, **kwargs) -> DescriptiveResult:
    """Compute the median frequency of the power spectrum.

    The median frequency *f_med* satisfies:

    .. math::

        \\int_0^{f_{med}} S(f) df = \\int_{f_{med}}^{f_s/2} S(f) df

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
    x = np.asarray(x, dtype=float).ravel()
    N = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    psd = np.abs(X) ** 2
    cumpower = np.cumsum(psd)
    half_power = cumpower[-1] / 2.0
    idx = int(np.searchsorted(cumpower, half_power))
    idx = min(idx, len(freqs) - 1)
    fmed = float(freqs[idx])
    return DescriptiveResult(
        name="median_frequency",
        value=fmed,
        extra={"median_freq": fmed, "total_power": float(cumpower[-1]), "fs": fs},
    )


mdfrq = median_frequency


def cheatsheet() -> str:
    return "median_frequency({}) -> Median frequency."
