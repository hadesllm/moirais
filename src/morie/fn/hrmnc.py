# morie.fn — function file (hadesllm/morie)
"""Harmonics-to-noise ratio."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "I know what I have to do but I don't know if I have the strength to do it."


def harmonic_ratio(x, fs: float = 1.0, **kwargs) -> DescriptiveResult:
    """Compute the harmonics-to-noise ratio (HNR) in decibels.

    Uses the autocorrelation method: HNR = 10 log10(r_max / (1 - r_max))
    where r_max is the maximum of the normalised ACF beyond lag 0.

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
    if acf[0] == 0:
        return DescriptiveResult(name="harmonic_ratio", value=0.0, extra={"hnr_db": 0.0})
    acf_norm = acf / acf[0]
    min_lag = max(2, int(fs / (fs / 2.0))) if fs > 0 else 2
    if min_lag >= len(acf_norm):
        r_max = 0.0
    else:
        r_max = float(np.max(acf_norm[min_lag:]))
    r_max = np.clip(r_max, 1e-10, 1.0 - 1e-10)
    hnr_db = float(10.0 * np.log10(r_max / (1.0 - r_max)))
    return DescriptiveResult(
        name="harmonic_ratio",
        value=hnr_db,
        extra={"hnr_db": hnr_db, "r_max": r_max, "fs": fs},
    )


hrmnc = harmonic_ratio


def cheatsheet() -> str:
    return "harmonic_ratio({}) -> Harmonics-to-noise ratio."
