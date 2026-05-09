# moirais.fn — function file (hadesllm/moirais)
"""Gabor-Wigner distribution.

Reference: Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
Analysis*, 3rd ed. IEEE/Wiley, Chapter 7.
"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

__all__ = ['gbrwv']

_QUOTE = "Between two worlds, the truth lies. -- Qui-Gon"


def gbrwv(
    x: np.ndarray,
    fs: float = 1.0,
    *,
    alpha: float = 0.5,
    nfft: int | None = None,
) -> DescriptiveResult:
    """Gabor-Wigner distribution (hybrid TF representation).

    Computes a weighted combination of the spectrogram (Gabor) and
    the Wigner-Ville distribution to balance cross-term suppression
    with time-frequency resolution:

    .. math::

        GW(t, f) = \\alpha \\, |STFT|^2 + (1 - \\alpha) \\, WVD

    Parameters
    ----------
    x : array-like
        1-D input signal.
    fs : float
        Sampling frequency in Hz.
    alpha : float
        Blending weight (0 = pure Wigner, 1 = pure spectrogram).
    nfft : int or None
        FFT length (default len(x)).

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if nfft is None:
        nfft = n

    nf = nfft // 2 + 1
    gabor = np.zeros((nf, n))
    wigner = np.zeros((nf, n))
    window = np.hanning(min(64, n))
    wlen = len(window)
    half = wlen // 2

    for ti in range(n):
        lo = max(0, ti - half)
        hi = min(n, ti + half + 1)
        seg = x[lo:hi]
        w = window[:len(seg)]
        X = np.fft.rfft(seg * w, n=nfft)
        gabor[:, ti] = np.abs(X) ** 2

    xa = np.zeros(2 * n)
    xa[:n] = x
    for ti in range(n):
        tau_max = min(ti, n - 1 - ti, nfft // 2)
        r = np.zeros(nfft)
        for tau in range(-tau_max, tau_max + 1):
            r[tau % nfft] = xa[ti + tau] * xa[ti - tau]
        W = np.fft.rfft(r, n=nfft)
        wigner[:, ti] = np.real(W[:nf])

    gw = alpha * gabor + (1 - alpha) * wigner
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    times = np.arange(n) / fs

    return DescriptiveResult(
        name="gbrwv",
        value=float(np.mean(gw)),
        extra={"gw": gw, "frequencies": freqs, "times": times, "alpha": alpha},
    )


def cheatsheet() -> str:
    return "gbrwv({}) -> Gabor-Wigner distribution."
