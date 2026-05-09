"""Spectral analysis (periodogram)."""

import numpy as np

from ._containers import DescriptiveResult


def periodogram(y: np.ndarray, fs: float = 1.0) -> DescriptiveResult:
    """
    Compute the periodogram spectral density estimate.

    .. math::

        I(\\omega_k) = \\frac{1}{n} \\left| \\sum_{t=0}^{n-1}
        y_t e^{-i 2\\pi \\omega_k t} \\right|^2

    :param y: 1-D time series.
    :param fs: Sampling frequency. Default 1.0.
    :return: DescriptiveResult with frequencies and power spectral density.
    :raises ValueError: If series too short.

    References
    ----------
    Schuster A. (1898). On the investigation of hidden periodicities.
    *Terrestrial Magnetism*, 3(1), 13-41.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    if n < 4:
        raise ValueError(f"Need at least 4 observations, got {n}.")
    y = y - y.mean()
    fft_vals = np.fft.rfft(y)
    psd = np.abs(fft_vals) ** 2 / n
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    peak_idx = int(np.argmax(psd[1:])) + 1
    peak_freq = float(freqs[peak_idx])
    peak_period = 1.0 / peak_freq if peak_freq > 0 else float("inf")
    return DescriptiveResult(
        name="periodogram",
        value=float(psd[peak_idx]),
        extra={
            "frequencies": freqs,
            "psd": psd,
            "peak_frequency": peak_freq,
            "peak_period": peak_period,
            "fs": fs,
            "n": n,
        },
    )


specn = periodogram


def cheatsheet() -> str:
    return "periodogram({}) -> Spectral density via periodogram."
