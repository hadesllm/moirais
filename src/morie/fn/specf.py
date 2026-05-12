# morie.fn — function file (hadesllm/morie)
"""Spectral density estimation (Welch periodogram)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["spectral_density"]


def spectral_density(x, fs=1.0, nperseg=None):
    r"""Welch power-spectral-density estimate.

    .. math::

        S(f) = \frac{1}{K U}\sum_{k=1}^K |X_k(f)|^2

    where :math:`X_k` is the windowed DFT of segment :math:`k`,
    :math:`U` the window normalisation, and :math:`K` the number of
    averaged segments.

    Parameters
    ----------
    x : array-like
        Time series.
    fs : float, default 1.0
        Sampling frequency (Hz).
    nperseg : int, optional
        Segment length; default ``max(n//4, 8)``.

    Returns
    -------
    RichResult
        keys: ``frequencies``, ``psd``, ``n_segments``, ``nperseg``,
        ``fs``, ``n``, ``method``.

    References
    ----------
    Welch PD (1967). The Use of Fast Fourier Transform for the
    Estimation of Power Spectra. *IEEE Trans. Audio Electroacoust.*
    15(2), 70-73.
    """
    r = np.asarray(x, dtype=float).ravel()
    n = r.size
    if n < 8:
        raise ValueError(f"Need at least 8 observations, got {n}.")
    if nperseg is None:
        nperseg = max(n // 4, 8)
    nperseg = int(min(nperseg, n))

    try:
        from scipy import signal as sps
        f, S = sps.welch(r, fs=fs, nperseg=nperseg)
        return RichResult(payload={
            "frequencies": f, "psd": S,
            "n_segments": int(max(1, (n - nperseg) // (nperseg // 2) + 1)),
            "nperseg": int(nperseg), "fs": float(fs), "n": int(n),
            "method": "Welch PSD via scipy.signal.welch",
        })
    except Exception:
        pass

    # Pure-NumPy Welch with Hann window, 50% overlap.
    step = max(nperseg // 2, 1)
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(nperseg) / max(nperseg - 1, 1))
    U = float(np.sum(win ** 2))
    nfreq = nperseg // 2 + 1
    S = np.zeros(nfreq)
    nseg = 0
    start = 0
    while start + nperseg <= n:
        seg = (r[start:start + nperseg] - r[start:start + nperseg].mean()) * win
        F = np.fft.rfft(seg)
        S += np.abs(F) ** 2
        nseg += 1
        start += step
    S /= (nseg * U * fs)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    return RichResult(payload={
        "frequencies": freqs, "psd": S,
        "n_segments": int(nseg), "nperseg": int(nperseg),
        "fs": float(fs), "n": int(n),
        "method": "Welch PSD (Hann window, 50% overlap, numpy)",
    })


def cheatsheet():
    return "specf: Spectral density estimate (Welch 1967)."
