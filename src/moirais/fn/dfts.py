# moirais.fn — function file (hadesllm/moirais)
"""Power spectral density via FFT."""

import numpy as np

from ._containers import DescriptiveResult


def dft_spectrum(y: np.ndarray, sampling_rate: float = 1.0) -> DescriptiveResult:
    """
    Compute the power spectral density via the Fast Fourier Transform.

    :param y: (n,) time series.
    :param sampling_rate: Sampling rate in Hz (default 1.0).
    :return: DescriptiveResult with frequencies and power spectrum.
    :raises ValueError: If series too short.

    References
    ----------
    Cooley JW, Tukey JW (1965). An algorithm for the machine
    calculation of complex Fourier series.
    Mathematics of Computation, 19(90), 297-301.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    n = len(y)
    if n < 4:
        raise ValueError("Need at least 4 observations.")
    y_centered = y - y.mean()
    fft_vals = np.fft.rfft(y_centered)
    psd = (np.abs(fft_vals) ** 2) / n
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
    dominant_idx = np.argmax(psd[1:]) + 1
    dominant_freq = float(freqs[dominant_idx])
    return DescriptiveResult(
        name="dft_spectrum",
        value=dominant_freq,
        extra={
            "frequencies": freqs,
            "power": psd,
            "dominant_frequency": dominant_freq,
            "dominant_period": 1.0 / dominant_freq if dominant_freq > 0 else float("inf"),
            "n": n,
            "sampling_rate": sampling_rate,
        },
    )


dfts = dft_spectrum


def cheatsheet() -> str:
    return "dft_spectrum({}) -> Power spectral density via FFT."
