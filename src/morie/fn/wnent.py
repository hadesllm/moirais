"""Wiener entropy."""

import numpy as np

from ._containers import ESRes


def wiener_entropy(x, fs: float = 1.0, nfft: int | None = None, **kwargs) -> ESRes:
    r"""
    Compute Wiener entropy (spectral flatness).

    Ratio of geometric mean to arithmetic mean of the power spectrum.
    Values near 0 indicate tonal signals; values near 1 indicate noise.

    .. math::

        WE = \\frac{\\exp\\left(\\frac{1}{N}\\sum_k \\ln S(k)\\right)}
             {\\frac{1}{N}\\sum_k S(k)}

    :param x: 1-D array-like signal.
    :param fs: Sampling frequency (default 1.0).
    :param nfft: FFT length (default len(x)).
    :return: ESRes with Wiener entropy (spectral flatness).

    References
    ----------
    Johnston JD (1988). Transform coding of audio signals using
    perceptual noise criteria. IEEE JSAC, 6(2), 314-323.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if n < 2:
        raise ValueError("Need at least 2 observations.")
    if nfft is None:
        nfft = n
    X = np.fft.rfft(x, n=nfft)
    psd = np.abs(X) ** 2
    psd = psd[psd > 0]
    if len(psd) < 1:
        return ESRes(measure="wiener_entropy", estimate=0.0, n=n, extra={})

    log_mean = float(np.mean(np.log(psd)))
    arith_mean = float(np.mean(psd))
    we = float(np.exp(log_mean) / arith_mean) if arith_mean > 0 else 0.0

    return ESRes(
        measure="wiener_entropy",
        estimate=we,
        n=n,
        extra={"spectral_flatness": we, "fs": fs, "nfft": nfft},
    )


wnent = wiener_entropy


def cheatsheet() -> str:
    return "wiener_entropy(x) -> Spectral flatness (Wiener entropy)."
