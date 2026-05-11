"""Wavelet threshold selection (VisuShrink, SureShrink)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Judge me by my size, do you?"


def wavelet_threshold(
    coeffs: np.ndarray,
    method: str = "universal",
    sigma: float | None = None,
) -> DescriptiveResult:
    """Select threshold for wavelet coefficient shrinkage.

    Methods:
    - ``universal``: VisuShrink :math:`\\lambda = \\sigma \\sqrt{2 \\ln N}`
    - ``sure``: SureShrink (Stein's Unbiased Risk Estimate)

    Parameters
    ----------
    coeffs : array-like
        Wavelet detail coefficients.
    method : str
        'universal' or 'sure' (default 'universal').
    sigma : float or None
        Noise std dev. Estimated from MAD if None.

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``threshold``, ``method``, ``sigma``.
    """
    coeffs = np.asarray(coeffs, dtype=float).ravel()
    N = len(coeffs)
    if N == 0:
        raise ValueError("Empty coefficient array")
    if sigma is None:
        sigma = float(np.median(np.abs(coeffs)) / 0.6745)
    if method == "universal":
        threshold = sigma * np.sqrt(2 * np.log(N)) if N > 1 else 0.0
    elif method == "sure":
        sorted_c2 = np.sort(coeffs**2)
        cumsum = np.cumsum(sorted_c2)
        n = np.arange(1, N + 1)
        risk = (N - 2 * n + cumsum + (N - n) * sorted_c2) / N
        idx = int(np.argmin(risk))
        sure_thresh = sigma * np.sqrt(sorted_c2[idx]) if sigma > 0 else 0.0
        universal = sigma * np.sqrt(2 * np.log(N)) if N > 1 else 0.0
        threshold = min(sure_thresh, universal)
    else:
        raise ValueError(f"method must be 'universal' or 'sure', got '{method}'")
    return DescriptiveResult(
        name="wavelet_threshold",
        value=float(threshold),
        extra={"threshold": threshold, "method": method, "sigma": sigma},
    )


wvthr = wavelet_threshold


def cheatsheet() -> str:
    return "wavelet_threshold({}) -> Wavelet threshold selection (VisuShrink, SureShrink)."
