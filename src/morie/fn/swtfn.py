"""Stationary (undecimated) Wavelet Transform."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "It is during our darkest moments that we must focus to see the light. -- Aristotle"


def swt_decompose(
    x: np.ndarray,
    wavelet: str = "db4",
    level: int = 3,
) -> DescriptiveResult:
    """Stationary Wavelet Transform (algorithme a trous).

    Unlike the DWT, the SWT does NOT downsample; instead it upsamples the
    filters at each level, yielding translation-invariant coefficients.

    Parameters
    ----------
    x : array-like
        1-D input signal.
    wavelet : str
        Wavelet name (default 'db4').
    level : int
        Decomposition level (default 3).

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``coeffs`` (list of (cA, cD) tuples per level).
    """
    from .dwtfn import _wavelet_filter

    x = np.asarray(x, dtype=float).ravel()
    lo, hi = _wavelet_filter(wavelet)
    result_coeffs = []
    approx = x.copy()
    for j in range(level):
        lo_up = np.zeros(len(lo) + (len(lo) - 1) * (2**j - 1))
        hi_up = np.zeros_like(lo_up)
        lo_up[:: 2**j] = lo
        hi_up[:: 2**j] = hi
        ca = np.convolve(approx, lo_up, mode="same")
        cd = np.convolve(approx, hi_up, mode="same")
        result_coeffs.append((ca, cd))
        approx = ca
    return DescriptiveResult(
        name="swt_decompose",
        value=float(level),
        extra={"coeffs": result_coeffs, "wavelet": wavelet, "level": level},
    )


swtfn = swt_decompose


def cheatsheet() -> str:
    return "swt_decompose({}) -> Stationary (undecimated) Wavelet Transform."
