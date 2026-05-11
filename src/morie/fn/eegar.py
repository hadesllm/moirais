# morie.fn — function file (hadesllm/morie)
"""EEG autoregressive modeling.

Reference: Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
Analysis*, 3rd ed. IEEE/Wiley, Chapter 13.
"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

__all__ = ['eegar']
def eegar(
    x: np.ndarray,
    fs: float = 256.0,
    *,
    order: int = 12,
    nfft: int = 512,
) -> DescriptiveResult:
    """Autoregressive model for EEG spectral estimation.

    Uses the Yule-Walker method to fit AR coefficients and compute
    the model-based PSD.

    Parameters
    ----------
    x : array-like
        1-D EEG signal.
    fs : float
        Sampling frequency in Hz.
    order : int
        AR model order.
    nfft : int
        Number of frequency bins.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x - np.mean(x)
    n = len(x)

    r = np.correlate(x, x, mode="full")[n - 1:]
    r = r[:order + 1] / n

    R = np.zeros((order, order))
    for i in range(order):
        for j in range(order):
            R[i, j] = r[abs(i - j)]
    rhs = r[1:order + 1]

    try:
        ar = np.linalg.solve(R, rhs)
    except np.linalg.LinAlgError:
        ar = np.zeros(order)

    sigma2 = r[0] - np.dot(ar, rhs)
    sigma2 = max(sigma2, 1e-20)

    freqs = np.linspace(0, fs / 2, nfft)
    psd = np.zeros(nfft)
    for i, f in enumerate(freqs):
        z = np.exp(-1j * 2 * np.pi * f / fs * np.arange(1, order + 1))
        denom = np.abs(1 - np.dot(ar, z)) ** 2
        psd[i] = sigma2 / max(denom, 1e-20)

    return DescriptiveResult(
        name="eegar",
        value=float(order),
        extra={
            "frequencies": freqs,
            "psd": psd,
            "ar_coeffs": ar,
            "noise_variance": sigma2,
        },
    )


def cheatsheet() -> str:
    return "eegar({}) -> EEG AR modeling."
