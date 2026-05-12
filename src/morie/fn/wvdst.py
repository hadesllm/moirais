"""Wigner-Ville distribution for time-frequency analysis.

Reference: Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
Analysis*, 3rd ed. IEEE/Wiley, Chapter 6.
"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult
def wigner_ville(
    x: np.ndarray,
    fs: float = 1.0,
    *,
    nfft: int | None = None,
) -> DescriptiveResult:
    r"""Compute the Wigner-Ville distribution (WVD).

    The WVD is a bilinear time-frequency representation defined as:

    .. math::

        W_x(t, f) = \\int_{-\\infty}^{\\infty} x(t + \\tau/2) \\,
        x^*(t - \\tau/2) \\, e^{-j 2\\pi f \\tau} \\, d\\tau

    Provides perfect time-frequency resolution but suffers from
    cross-term interference for multi-component signals.

    Parameters
    ----------
    x : array-like
        1-D input signal (real or analytic).
    fs : float
        Sampling frequency in Hz (default 1.0).
    nfft : int or None
        FFT size for frequency axis.  Defaults to ``2 * len(x)``.

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``wvd`` (n_freq x n_time), ``times``,
        ``frequencies``.

    References
    ----------
    Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
    Analysis*, 3rd ed. IEEE/Wiley, Chapter 6.

    Claasen, T.A.C.M. & Mecklenbr\"auker, W.F.G. (1980). The Wigner
    distribution -- A tool for time-frequency signal analysis. *Philips
    J. Res.*, 35, 217--250.
    """
    from scipy.signal import hilbert

    x_raw = np.asarray(x, dtype=float).ravel()
    n = len(x_raw)
    if nfft is None:
        nfft = 2 * n

    xa = hilbert(x_raw)

    wvd = np.zeros((nfft, n))
    for t_idx in range(n):
        tau_max = min(t_idx, n - 1 - t_idx)
        kernel = np.zeros(nfft, dtype=complex)
        for tau in range(-tau_max, tau_max + 1):
            kernel[tau % nfft] = xa[t_idx + tau] * np.conj(xa[t_idx - tau])
        wvd[:, t_idx] = np.real(np.fft.fft(kernel, n=nfft))

    times = np.arange(n) / fs
    half = nfft // 2
    frequencies = np.arange(half) * (fs / (2 * nfft))
    wvd = wvd[:half, :]

    return DescriptiveResult(
        name="wigner_ville",
        value=float(n),
        extra={
            "wvd": wvd,
            "times": times,
            "frequencies": frequencies,
        },
    )


wvdst = wigner_ville


def cheatsheet() -> str:
    return "wigner_ville({}) -> Wigner-Ville distribution for time-frequency analysis."
