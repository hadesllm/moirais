# morie.fn -- function file (hadesllm/morie)
"""Coherence function between two time series."""

import numpy as np

from ._containers import DescriptiveResult


def coherence(
    x: np.ndarray, y: np.ndarray, nperseg: int | None = None, fs: float = 1.0
) -> DescriptiveResult:
    """
    Magnitude-squared coherence between two time series.

    .. math::

        C_{xy}(f) = \\frac{|S_{xy}(f)|^2}{S_{xx}(f) S_{yy}(f)}

    Uses Welch's method with overlapping segments.

    :param x: 1-D first series.
    :param y: 1-D second series (same length).
    :param nperseg: Segment length. Default n//4.
    :param fs: Sampling frequency. Default 1.0.
    :return: DescriptiveResult with frequencies and coherence values.
    :raises ValueError: On length mismatch.

    References
    ----------
    Welch P.D. (1967). The use of FFT for the estimation of power
    spectra. *IEEE Trans. Audio Electroacoustics*, 15(2), 70-73.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if len(x) != len(y):
        raise ValueError(f"x and y must have same length, got {len(x)} and {len(y)}.")
    n = len(x)
    if n < 8:
        raise ValueError(f"Need at least 8 observations, got {n}.")
    if nperseg is None:
        nperseg = max(n // 4, 4)
    nperseg = min(nperseg, n)
    step = nperseg // 2
    nfreq = nperseg // 2 + 1
    Sxx = np.zeros(nfreq)
    Syy = np.zeros(nfreq)
    Sxy = np.zeros(nfreq, dtype=complex)
    nseg = 0
    start = 0
    while start + nperseg <= n:
        xseg = x[start : start + nperseg]
        yseg = y[start : start + nperseg]
        xseg = xseg - xseg.mean()
        yseg = yseg - yseg.mean()
        fx = np.fft.rfft(xseg)
        fy = np.fft.rfft(yseg)
        Sxx += np.abs(fx) ** 2
        Syy += np.abs(fy) ** 2
        Sxy += fx * np.conj(fy)
        nseg += 1
        start += step
    Sxx /= nseg
    Syy /= nseg
    Sxy /= nseg
    denom = Sxx * Syy
    denom = np.where(denom < 1e-15, 1e-15, denom)
    coh = np.abs(Sxy) ** 2 / denom
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    return DescriptiveResult(
        name="coherence",
        value=float(np.mean(coh)),
        extra={
            "frequencies": freqs,
            "coherence": coh,
            "n_segments": nseg,
            "nperseg": nperseg,
            "fs": fs,
            "n": n,
        },
    )


cohrc = coherence


def cheatsheet() -> str:
    return "coherence({}) -> Magnitude-squared coherence between two series."
