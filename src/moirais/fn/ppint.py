# moirais.fn — function file (hadesllm/moirais)
"""P-P interval computation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Stay on target."


def pp_interval(peaks, fs: float = 1.0, **kwargs) -> DescriptiveResult:
    """Compute P-P interval series from detected P-wave peak indices.

    Parameters
    ----------
    peaks : array-like of int
        P-wave peak sample indices.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    DescriptiveResult
    """
    peaks = np.asarray(peaks, dtype=int)
    if len(peaks) < 2:
        pp = np.array([])
        mean_pp = 0.0
    else:
        pp = np.diff(peaks) / fs
        mean_pp = float(np.mean(pp))
    return DescriptiveResult(
        name="pp_interval",
        value=mean_pp,
        extra={
            "pp_intervals": pp,
            "n_peaks": len(peaks),
            "std_pp": float(np.std(pp, ddof=1)) if len(pp) > 1 else 0.0,
            "fs": fs,
        },
    )


ppint = pp_interval


def cheatsheet() -> str:
    return "pp_interval({}) -> P-P interval computation."
