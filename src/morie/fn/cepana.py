"""Compute real cepstrum and cepstral coefficients."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def cepstral_analysis(
    x: np.ndarray,
    *,
    n_coeffs: int = 13,
    fs: float = 1.0,
    lifter: int = 0,
) -> DescriptiveResult:
    """Compute real cepstrum and cepstral coefficients.

    The real cepstrum is defined as: c(n) = IFFT(log(|FFT(x)|))

    Parameters
    ----------
    x : array-like
        Input signal.
    n_coeffs : int
        Number of cepstral coefficients to return.
    fs : float
        Sampling frequency.
    lifter : int
        If > 0, apply sinusoidal liftering to coefficients.

    Returns
    -------
    DescriptiveResult
        With ``value`` = cepstral coefficients and
        ``extra`` containing full cepstrum.
    """
    x = np.asarray(x, dtype=float).ravel()
    if len(x) < 4:
        raise ValueError("Signal must have at least 4 samples")

    nfft = max(256, 2 ** int(np.ceil(np.log2(len(x)))))
    spectrum = np.fft.fft(x, n=nfft)
    log_mag = np.log(np.abs(spectrum) + 1e-30)
    cepstrum = np.real(np.fft.ifft(log_mag))

    n_coeffs = min(n_coeffs, nfft // 2)
    coeffs = cepstrum[:n_coeffs].copy()

    if lifter > 0:
        lift = 1 + (lifter / 2) * np.sin(np.pi * np.arange(n_coeffs) / lifter)
        coeffs *= lift

    quefrencies = np.arange(n_coeffs) / fs

    return DescriptiveResult(
        name="cepstral_analysis",
        value=coeffs,
        extra={
            "cepstrum": cepstrum[: nfft // 2],
            "quefrencies": quefrencies,
            "nfft": nfft,
            "n_coeffs": n_coeffs,
            "fs": fs,
        },
    )


cepana = cepstral_analysis


def cheatsheet() -> str:
    return 'cepstral_analysis({}) -> Cepstral analysis.'
