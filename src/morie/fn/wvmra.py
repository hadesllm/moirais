"""Wavelet multiresolution analysis."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Great, kid. Don't get cocky."


def wavelet_mra(
    x: np.ndarray,
    wavelet: str = "db4",
    level: int = 3,
) -> DescriptiveResult:
    """Multiresolution analysis: reconstruct detail and approximation at each level.

    Decomposes *x* into additive components: one approximation signal at the
    coarsest level plus detail signals at each level, such that their sum
    reconstructs the original signal.

    Parameters
    ----------
    x : array-like
        1-D input signal.
    wavelet : str
        Wavelet name (default 'db4').
    level : int
        Number of decomposition levels (default 3).

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``details`` (list of detail signals per level),
        ``approximation`` (coarsest approximation).
    """
    from .dwtfn import dwt_decompose
    from .idwtf import idwt_reconstruct

    x = np.asarray(x, dtype=float).ravel()
    res = dwt_decompose(x, wavelet=wavelet, level=level)
    coeffs = res.extra["coeffs"]
    n_levels = len(coeffs) - 1
    details = []
    for i in range(1, len(coeffs)):
        zero_coeffs = [np.zeros_like(c) for c in coeffs]
        zero_coeffs[i] = coeffs[i]
        rec = idwt_reconstruct(zero_coeffs, wavelet=wavelet)
        details.append(rec.extra["signal"][: len(x)])
    zero_coeffs = [np.zeros_like(c) for c in coeffs]
    zero_coeffs[0] = coeffs[0]
    rec_a = idwt_reconstruct(zero_coeffs, wavelet=wavelet)
    approx = rec_a.extra["signal"][: len(x)]
    return DescriptiveResult(
        name="wavelet_mra",
        value=float(n_levels),
        extra={"details": details, "approximation": approx, "level": n_levels},
    )


wvmra = wavelet_mra


def cheatsheet() -> str:
    return "wavelet_mra({}) -> Wavelet multiresolution analysis."
