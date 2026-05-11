# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bispectrum estimation.

Reference: Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
Analysis*, 3rd ed. IEEE/Wiley, Chapter 6.
"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

__all__ = ['bspec']
def bspec(
    x: np.ndarray,
    fs: float = 1.0,
    *,
    nfft: int = 256,
    nseg: int | None = None,
) -> DescriptiveResult:
    """Direct bispectrum estimation via segment averaging.

    The bispectrum is defined as:

    .. math::

        B(f_1, f_2) = E[X(f_1) X(f_2) X^*(f_1 + f_2)]

    Parameters
    ----------
    x : array-like
        1-D input signal.
    fs : float
        Sampling frequency in Hz.
    nfft : int
        FFT length per segment.
    nseg : int or None
        Number of segments (default: as many as possible).

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if nseg is None:
        nseg = max(1, n // nfft)
    seg_len = nfft

    nf = nfft // 2 + 1
    bisp = np.zeros((nf, nf), dtype=complex)
    count = 0

    for s in range(nseg):
        start = s * seg_len
        end = start + seg_len
        if end > n:
            break
        seg = x[start:end]
        X = np.fft.rfft(seg, n=nfft)
        for i in range(nf):
            for j in range(i, nf):
                k = i + j
                if k < nf:
                    bisp[i, j] += X[i] * X[j] * np.conj(X[k])
        count += 1

    if count > 0:
        bisp /= count

    for i in range(nf):
        for j in range(i):
            bisp[i, j] = np.conj(bisp[j, i])

    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

    return DescriptiveResult(
        name="bspec",
        value=float(np.max(np.abs(bisp))),
        extra={
            "bispectrum": bisp,
            "frequencies": freqs,
            "n_segments": count,
        },
    )


def cheatsheet() -> str:
    return "bspec({}) -> Bispectrum estimation."
