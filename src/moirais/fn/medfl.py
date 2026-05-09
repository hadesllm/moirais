# moirais.fn — function file (hadesllm/moirais)
"""Median filter (nonlinear smoothing).

Reference: Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
Analysis*, 3rd ed. IEEE/Wiley, Chapter 3.
"""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult

__all__ = ['medfl']
def medfl(
    x: np.ndarray,
    fs: float = 1.0,
    *,
    kernel_size: int = 5,
) -> SignalResult:
    """Apply a running median filter.

    Parameters
    ----------
    x : array-like
        1-D input signal.
    fs : float
        Sampling frequency in Hz.
    kernel_size : int
        Window length (must be odd; even values incremented by 1).

    Returns
    -------
    SignalResult
    """
    from scipy.ndimage import median_filter

    x = np.asarray(x, dtype=float).ravel()
    if kernel_size % 2 == 0:
        kernel_size += 1
    y = median_filter(x, size=kernel_size).astype(float)

    return SignalResult(
        name="medfl",
        filtered=y,
        fs=fs,
        n_samples=len(x),
        extra={"kernel_size": kernel_size},
    )


def cheatsheet() -> str:
    return "medfl({}) -> Median filter."
