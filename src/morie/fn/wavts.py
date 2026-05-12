# morie.fn — function file (hadesllm/morie)
"""Discrete wavelet decomposition for time series (Percival & Walden 2000)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["wavelet_time_series"]


def wavelet_time_series(x, wavelet="haar", level=None):
    r"""Multiresolution decomposition via the discrete wavelet transform.

    .. math::

        x(t) = \sum_j \sum_k c_{j,k}\,\psi_{j,k}(t)

    Parameters
    ----------
    x : array-like
        Univariate time series.
    wavelet : str, default ``"haar"``
        Wavelet family (passed to ``pywt`` when available).
    level : int, optional
        Decomposition depth; defaults to ``floor(log2 n)``.

    Returns
    -------
    RichResult
        keys: ``approximation`` (cA at the deepest level), ``details``
        (list of cD coefficients level-by-level), ``energies`` (variance
        per level), ``level``, ``n``, ``method``.

    References
    ----------
    Percival DB, Walden AT (2000). *Wavelet Methods for Time Series
    Analysis*. Cambridge UP.
    """
    y = np.asarray(x, dtype=float).ravel()
    n = y.size
    if n < 4:
        raise ValueError(f"Need at least 4 observations, got {n}.")
    max_level = int(np.floor(np.log2(n)))
    if level is None:
        level = min(max(max_level, 1), 6)
    level = int(min(level, max_level))

    try:
        import pywt
        coeffs = pywt.wavedec(y, wavelet, level=level)
        cA = coeffs[0]
        cDs = coeffs[1:]
        energies = [float(np.sum(c ** 2)) for c in coeffs]
        return RichResult(payload={
            "approximation": np.asarray(cA),
            "details": [np.asarray(c) for c in cDs],
            "energies": energies,
            "level": int(level), "n": int(n),
            "wavelet": wavelet,
            "method": f"DWT via pywt (wavelet={wavelet}, level={level})",
        })
    except Exception:
        pass

    # Pure-NumPy Haar DWT fallback.
    cA = y.copy()
    cDs = []
    for _ in range(level):
        if cA.size < 2:
            break
        if cA.size % 2 == 1:
            cA = np.concatenate([cA, cA[-1:]])
        even = cA[0::2]
        odd = cA[1::2]
        cA_new = (even + odd) / np.sqrt(2.0)
        cD = (even - odd) / np.sqrt(2.0)
        cDs.append(cD)
        cA = cA_new
    energies = [float(np.sum(cA ** 2))] + [float(np.sum(c ** 2)) for c in cDs]
    return RichResult(payload={
        "approximation": cA,
        "details": cDs[::-1],
        "energies": energies,
        "level": int(level), "n": int(n),
        "wavelet": "haar",
        "method": "Haar DWT (numpy fallback)",
    })


def cheatsheet():
    return "wavts: Wavelet decomposition (Percival & Walden 2000)."
