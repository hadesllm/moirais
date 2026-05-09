# moirais.fn — function file (hadesllm/moirais)
"""It does not matter how slowly you go as long as you do not stop. — Confucius"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def fft_peaks(
    signal: np.ndarray,
    fs: float = 1.0,
    n_peaks: int = 5,
) -> DescriptiveResult:
    """Identify dominant frequency peaks in a signal using the FFT.

    Computes the single-sided amplitude spectrum and returns the
    top *n_peaks* frequencies by magnitude.

    Parameters
    ----------
    signal : ndarray, shape (n_samples,)
        Input time-domain signal.
    fs : float
        Sampling frequency in Hz.
    n_peaks : int
        Number of top peaks to return.

    Returns
    -------
    DescriptiveResult
        name='FFT Peaks', value=dominant frequency (Hz),
        extra has 'frequencies' (Hz), 'magnitudes', 'spectrum'
        (full single-sided amplitude), 'freq_axis'.

    References
    ----------
    Cooley, J.W. & Tukey, J.W. (1965). An algorithm for the machine
    calculation of complex Fourier series. *Mathematics of
    Computation*, 19(90), 297-301. doi:10.1090/S0025-5718-1965-0178586-1
    """
    x = np.asarray(signal, dtype=np.float64).ravel()
    N = len(x)
    if N == 0:
        return DescriptiveResult(
            name="FFT Peaks",
            value=0.0,
            extra={"frequencies": [], "magnitudes": [], "spectrum": np.array([]), "freq_axis": np.array([])},
        )

    X = np.fft.rfft(x)
    magnitudes = (2.0 / N) * np.abs(X)
    magnitudes[0] /= 2.0
    freq_axis = np.fft.rfftfreq(N, d=1.0 / fs)

    mag_no_dc = magnitudes.copy()
    mag_no_dc[0] = 0.0

    n_peaks = min(n_peaks, len(mag_no_dc))
    peak_idx = np.argsort(mag_no_dc)[::-1][:n_peaks]

    peak_freqs = freq_axis[peak_idx].tolist()
    peak_mags = magnitudes[peak_idx].tolist()

    return DescriptiveResult(
        name="FFT Peaks",
        value=float(peak_freqs[0]) if peak_freqs else 0.0,
        extra={
            "frequencies": peak_freqs,
            "magnitudes": peak_mags,
            "spectrum": magnitudes,
            "freq_axis": freq_axis,
            "n_samples": N,
            "fs": fs,
        },
    )


def cheatsheet() -> str:
    return "It does not matter how slowly you go as long as you do not stop. — Confucius"
