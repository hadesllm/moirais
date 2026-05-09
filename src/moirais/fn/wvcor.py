"""Wavelet correlation (scale-by-scale correlation)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The greatest teacher, failure is."


def wavelet_correlation(
    x: np.ndarray,
    y: np.ndarray,
    wavelet: str = "db4",
    level: int = 3,
) -> DescriptiveResult:
    """Scale-by-scale wavelet correlation between two signals.

    Decomposes both signals and computes Pearson correlation between
    corresponding wavelet coefficient vectors at each level.

    Parameters
    ----------
    x, y : array-like
        Input signals (same length).
    wavelet : str
        Wavelet name (default 'db4').
    level : int
        Decomposition level (default 3).

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``correlations`` (per level), ``mean_correlation``.
    """
    from .dwtfn import dwt_decompose

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    res_x = dwt_decompose(x, wavelet=wavelet, level=level)
    res_y = dwt_decompose(y, wavelet=wavelet, level=level)
    cx = res_x.extra["coeffs"]
    cy = res_y.extra["coeffs"]
    correlations = []
    for a, b in zip(cx, cy):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            correlations.append(0.0)
        else:
            correlations.append(float(np.corrcoef(a, b)[0, 1]))
    mean_corr = float(np.mean(correlations))
    return DescriptiveResult(
        name="wavelet_correlation",
        value=mean_corr,
        extra={"correlations": correlations, "mean_correlation": mean_corr},
    )


wvcor = wavelet_correlation


def cheatsheet() -> str:
    return "wavelet_correlation({}) -> Wavelet correlation (scale-by-scale correlation)."
