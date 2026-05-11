# morie.fn — function file (hadesllm/morie)
"""MODWT (maximal overlap DWT, translation-invariant)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "I know what I have to do, but I don't know if I have the strength to do it."


def modwt_decompose(
    x: np.ndarray,
    wavelet: str = "db4",
    level: int = 3,
) -> DescriptiveResult:
    """Maximal Overlap Discrete Wavelet Transform (MODWT).

    Unlike the DWT, the MODWT does not downsample and is translation-invariant.
    Filters are rescaled by :math:`1/\\sqrt{2^j}` at each level *j*.

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
    N = len(x)
    result_coeffs = []
    approx = x.copy()
    for j in range(level):
        scale = 2**j
        lo_j = np.zeros(len(lo) * scale - (scale - 1))
        hi_j = np.zeros_like(lo_j)
        lo_j[::scale] = lo / np.sqrt(2 ** (j + 1))
        hi_j[::scale] = hi / np.sqrt(2 ** (j + 1))
        ca = np.convolve(approx, lo_j, mode="same")
        cd = np.convolve(approx, hi_j, mode="same")
        result_coeffs.append((ca, cd))
        approx = ca
    return DescriptiveResult(
        name="modwt_decompose",
        value=float(level),
        extra={"coeffs": result_coeffs, "wavelet": wavelet, "level": level},
    )


modwt = modwt_decompose


def cheatsheet() -> str:
    return "modwt_decompose({}) -> MODWT (maximal overlap DWT, translation-invariant)."
