# morie.fn — function file (hadesllm/morie)
"""Minimum Variance Distortionless Response (Capon) spectral estimation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Life is really simple, but we insist on making it complicated. — Confucius"


def mvdr_spectrum_fn(
    x: np.ndarray,
    order: int = 16,
    nfft: int = 512,
    fs: float = 1.0,
) -> DescriptiveResult:
    r"""Compute MVDR (Capon) spectral estimate.

    .. math::

        S_{\\text{MVDR}}(f) = \\frac{1}{\\mathbf{e}^H(f) R^{-1} \\mathbf{e}(f)}

    :param x: 1-D input signal.
    :param order: Correlation matrix dimension (default 16).
    :param nfft: Number of frequency bins (default 512).
    :param fs: Sampling frequency in Hz (default 1.0).
    :return: DescriptiveResult with frequency vector and MVDR PSD.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    x_c = x - np.mean(x)
    R = np.zeros((order, order), dtype=complex)
    for i in range(order, n):
        seg = x_c[i - order : i][::-1]
        R += np.outer(seg, seg.conj())
    R /= n - order
    R += 1e-10 * np.eye(order)
    try:
        R_inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        R_inv = np.linalg.pinv(R)
    freqs = np.linspace(0, fs / 2, nfft)
    psd = np.zeros(nfft)
    for i, f in enumerate(freqs):
        w = 2 * np.pi * f / fs
        e = np.exp(-1j * w * np.arange(order))
        psd[i] = 1.0 / np.abs(e.conj() @ R_inv @ e).real
    return DescriptiveResult(
        name="mvdr_spectrum",
        value=None,
        extra={"frequencies": freqs, "psd": psd, "order": order, "fs": fs},
    )


mvdr = mvdr_spectrum_fn


def cheatsheet() -> str:
    return "mvdr_spectrum_fn({}) -> Minimum Variance Distortionless Response (Capon) spectral es"
