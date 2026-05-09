"""Discrete wavelet transform (Haar)."""

import numpy as np

from ._containers import DescriptiveResult


def wavelet_decompose(y: np.ndarray, n_levels: int | None = None) -> DescriptiveResult:
    """
    Haar discrete wavelet transform decomposition.

    Decomposes the signal into approximation and detail coefficients
    at multiple resolution levels.

    :param y: (n,) time series (length should be power of 2, padded if not).
    :param n_levels: Number of decomposition levels (default: max possible).
    :return: DescriptiveResult with coefficients per level.

    References
    ----------
    Daubechies I (1992). Ten Lectures on Wavelets. SIAM.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    n = len(y)
    next_pow2 = 1 << int(np.ceil(np.log2(max(n, 2))))
    if n < next_pow2:
        y = np.pad(y, (0, next_pow2 - n), mode="constant")
    if n_levels is None:
        n_levels = int(np.log2(len(y)))
    details = []
    approx = y.copy()
    for level in range(n_levels):
        m = len(approx)
        if m < 2:
            break
        half = m // 2
        a = (approx[0::2] + approx[1::2]) / np.sqrt(2)
        d = (approx[0::2] - approx[1::2]) / np.sqrt(2)
        details.append(d)
        approx = a
    return DescriptiveResult(
        name="wavelet_decompose",
        value=float(n_levels),
        extra={"approximation": approx, "details": details, "n_levels": len(details), "original_length": n},
    )


wavlt = wavelet_decompose


def cheatsheet() -> str:
    return "wavelet_decompose({}) -> Discrete wavelet transform (Haar)."
