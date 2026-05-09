# moirais.fn — function file (hadesllm/moirais)
"""Marginal Hilbert spectrum from IMFs."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "I find your lack of faith disturbing."


def hilbert_spectrum(imfs, fs: float = 1.0, n_bins: int = 256, **kwargs) -> DescriptiveResult:
    """Compute the marginal Hilbert spectrum from a set of IMFs.

    Applies the Hilbert transform to each IMF to obtain instantaneous
    frequency and amplitude, then accumulates energy into frequency bins.

    Parameters
    ----------
    imfs : array-like, shape (n_imfs, N)
        Array of IMFs (rows).
    fs : float
        Sampling frequency (default 1.0).
    n_bins : int
        Number of frequency bins (default 256).

    Returns
    -------
    DescriptiveResult
        ``value`` is dominant frequency; ``extra`` has ``spectrum``,
        ``frequencies``, ``inst_freqs``, ``inst_amps``.

    References
    ----------
    Huang, N. E., et al. (1998). The empirical mode decomposition and
    the Hilbert spectrum. *Proc. R. Soc. Lond. A*, 454, 903-995.
    """
    from scipy.signal import hilbert

    imfs = np.asarray(imfs, dtype=float)
    if imfs.ndim == 1:
        imfs = imfs[np.newaxis, :]
    n_imfs, N = imfs.shape
    freq_edges = np.linspace(0, fs / 2.0, n_bins + 1)
    spectrum = np.zeros(n_bins)
    all_inst_freqs = []
    all_inst_amps = []
    for k in range(n_imfs):
        analytic = hilbert(imfs[k])
        inst_amp = np.abs(analytic)
        phase = np.unwrap(np.angle(analytic))
        inst_freq = np.diff(phase) / (2.0 * np.pi) * fs
        inst_freq = np.clip(inst_freq, 0, fs / 2.0)
        all_inst_freqs.append(inst_freq)
        all_inst_amps.append(inst_amp[:-1])
        for j in range(len(inst_freq)):
            idx = int(np.searchsorted(freq_edges[1:], inst_freq[j]))
            idx = min(idx, n_bins - 1)
            spectrum[idx] += inst_amp[j] ** 2
    frequencies = (freq_edges[:-1] + freq_edges[1:]) / 2.0
    dominant = float(frequencies[np.argmax(spectrum)])
    return DescriptiveResult(
        name="hilbert_spectrum",
        value=dominant,
        extra={
            "spectrum": spectrum,
            "frequencies": frequencies,
            "inst_freqs": all_inst_freqs,
            "inst_amps": all_inst_amps,
        },
    )


hilsp = hilbert_spectrum


def cheatsheet() -> str:
    return "hilbert_spectrum({}) -> Marginal Hilbert spectrum from IMFs."
