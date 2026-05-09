"""Wavelet denoising with thresholding."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "When I let go of what I am, I become what I might be. — Lao Tzu"


def wavelet_denoise(
    x: np.ndarray,
    wavelet: str = "db4",
    level: int | None = None,
    mode: str = "soft",
) -> DescriptiveResult:
    """Wavelet denoising via decomposition, thresholding, and reconstruction.

    Implements the Donoho-Johnstone VisuShrink universal threshold
    (Rangayyan & Krishnan, Ch. 8):

    .. math::

        \\lambda = \\sigma \\sqrt{2 \\ln N}

    Parameters
    ----------
    x : array-like
        Noisy 1-D signal.
    wavelet : str
        Wavelet name (default 'db4').
    level : int or None
        Decomposition level. Auto if None.
    mode : str
        'soft' or 'hard' thresholding (default 'soft').

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``denoised`` signal, ``threshold``, ``snr_improvement``.
    """
    from .dwtfn import dwt_decompose
    from .idwtf import idwt_reconstruct

    x = np.asarray(x, dtype=float).ravel()
    res = dwt_decompose(x, wavelet=wavelet, level=level)
    coeffs = [np.copy(c) for c in res.extra["coeffs"]]
    if len(coeffs) < 2:
        return DescriptiveResult(
            name="wavelet_denoise",
            value=0.0,
            extra={"denoised": x.copy(), "threshold": 0.0, "snr_improvement": 0.0},
        )
    finest_detail = coeffs[-1]
    sigma = float(np.median(np.abs(finest_detail)) / 0.6745)
    N = len(x)
    threshold = sigma * np.sqrt(2 * np.log(N)) if N > 1 else 0.0
    for i in range(1, len(coeffs)):
        c = coeffs[i]
        if mode == "soft":
            coeffs[i] = np.sign(c) * np.maximum(np.abs(c) - threshold, 0.0)
        else:
            coeffs[i] = c * (np.abs(c) >= threshold)
    rec = idwt_reconstruct(coeffs, wavelet=wavelet)
    denoised = rec.extra["signal"][: len(x)]
    noise_power = np.var(x - denoised) if np.var(x) > 0 else 1e-12
    snr_imp = 10 * np.log10(np.var(x) / max(noise_power, 1e-12))
    return DescriptiveResult(
        name="wavelet_denoise",
        value=float(snr_imp),
        extra={"denoised": denoised, "threshold": threshold, "snr_improvement": snr_imp},
    )


wvdns = wavelet_denoise


def cheatsheet() -> str:
    return "wavelet_denoise({}) -> Wavelet denoising with thresholding."
