# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Burg AR spectral estimation.

Reference: Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
Analysis*, 3rd ed. IEEE/Wiley, Chapter 6.
"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult
def burg_psd(
    x: np.ndarray,
    order: int = 16,
    nfft: int = 512,
    fs: float = 1.0,
) -> DescriptiveResult:
    r"""Burg autoregressive spectral estimation.

    Estimates the AR coefficients using the Burg method (minimizes
    forward + backward prediction error simultaneously) and computes
    the power spectral density from the AR model:

    .. math::

        P(f) = \\frac{\\sigma^2}{\\left| 1 - \\sum_{k=1}^{p}
        a_k e^{-j 2\\pi f k / f_s} \\right|^2}

    Parameters
    ----------
    x : array-like
        1-D input signal.
    order : int
        AR model order (default 16).
    nfft : int
        Number of frequency bins for the PSD (default 512).
    fs : float
        Sampling frequency in Hz (default 1.0).

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``frequencies``, ``psd``, ``ar_coeffs``,
        ``noise_variance``.

    References
    ----------
    Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
    Analysis*, 3rd ed. IEEE/Wiley, Chapter 6.

    Burg, J.P. (1967). Maximum entropy spectral analysis. *Proc. 37th
    Meeting of the Society of Exploration Geophysicists*.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)

    ef = x.copy()
    eb = x.copy()
    sigma2 = np.dot(x, x) / n
    ar = np.array([1.0])

    for p in range(1, order + 1):
        efp = ef[p:]
        ebp = eb[p - 1 : n - 1]
        num = -2.0 * np.dot(efp, ebp)
        den = np.dot(efp, efp) + np.dot(ebp, ebp)
        if abs(den) < 1e-20:
            break
        k = num / den

        ar_new = np.zeros(p + 1)
        ar_new[0] = 1.0
        ar_new[p] = k
        for i in range(1, p):
            ar_new[i] = ar[i] + k * ar[p - i]
        ar = ar_new

        sigma2 *= 1.0 - k * k

        ef_new = np.zeros(n)
        eb_new = np.zeros(n)
        for i in range(p, n):
            ef_new[i] = ef[i] + k * eb[i - 1]
            eb_new[i] = eb[i - 1] + k * ef[i]
        ef = ef_new
        eb = eb_new

    ar_coeffs = ar

    p_actual = len(ar_coeffs)
    freqs = np.linspace(0, fs / 2, nfft)
    psd = np.zeros(nfft)
    for i, f in enumerate(freqs):
        z = np.exp(-1j * 2 * np.pi * f / fs * np.arange(p_actual))
        denom = np.abs(np.dot(ar_coeffs, z)) ** 2
        psd[i] = sigma2 / max(denom, 1e-20)

    return DescriptiveResult(
        name="burg_psd",
        value=float(order),
        extra={
            "frequencies": freqs,
            "psd": psd,
            "ar_coeffs": ar_coeffs[1:],
            "noise_variance": sigma2,
        },
    )


burgp = burg_psd


def cheatsheet() -> str:
    return "burg_psd({}) -> Burg AR spectral estimation."
