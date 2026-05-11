"""Short-time quadratic frequency transform.

Reference: Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
Analysis*, 3rd ed. IEEE/Wiley, Chapter 7.
"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

__all__ = ['stqft']
def stqft(
    x: np.ndarray,
    fs: float = 1.0,
    *,
    nperseg: int = 128,
    noverlap: int | None = None,
    nfft: int | None = None,
) -> DescriptiveResult:
    """Short-time quadratic frequency transform.

    Computes a squared-magnitude STFT with instantaneous frequency
    weighting for enhanced time-frequency concentration.

    Parameters
    ----------
    x : array-like
        1-D input signal.
    fs : float
        Sampling frequency in Hz.
    nperseg : int
        Segment length.
    noverlap : int or None
        Overlap (default nperseg // 2).
    nfft : int or None
        FFT length (default nperseg).

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if noverlap is None:
        noverlap = nperseg // 2
    if nfft is None:
        nfft = nperseg

    hop = nperseg - noverlap
    n_frames = max(1, (n - nperseg) // hop + 1)
    nf = nfft // 2 + 1
    window = np.hanning(nperseg)
    twindow = np.arange(nperseg) * window

    stqf = np.zeros((nf, n_frames))
    times = np.zeros(n_frames)

    for i in range(n_frames):
        start = i * hop
        seg = x[start:start + nperseg]
        if len(seg) < nperseg:
            seg = np.pad(seg, (0, nperseg - len(seg)))
        X = np.fft.rfft(seg * window, n=nfft)
        Xt = np.fft.rfft(seg * twindow, n=nfft)
        mag2 = np.abs(X) ** 2 + 1e-20
        inst_freq = np.real(Xt * np.conj(X)) / mag2
        stqf[:, i] = mag2 * (1 + inst_freq ** 2)
        times[i] = (start + nperseg / 2) / fs

    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

    return DescriptiveResult(
        name="stqft",
        value=float(np.mean(stqf)),
        extra={"stqf": stqf, "frequencies": freqs, "times": times},
    )


def cheatsheet() -> str:
    return "stqft({}) -> Short-time quadratic frequency transform."
