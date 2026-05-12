# morie.fn -- function file (hadesllm/morie)
"""Burg autoregressive power spectral density estimation."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult


def burg_psd(
    x: np.ndarray,
    fs: float,
    *,
    order: int = 16,
    nfft: int = 256,
) -> SignalResult:
    """Burg AR PSD estimation (parametric, suited for short HRV windows).

    :param x: 1-D input signal.
    :param fs: Sampling frequency in Hz.
    :param order: AR model order (default 16).
    :param nfft: FFT length for PSD evaluation (default 256).
    :return: SignalResult with PSD in ``filtered`` and frequencies in ``extra["freqs"]``.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if order >= n:
        order = n - 1

    ef = x.copy()
    eb = x.copy()
    a = np.zeros(order + 1)
    a[0] = 1.0
    pe = float(np.dot(x, x) / n)

    for m in range(1, order + 1):
        efm = ef[m:]
        ebm = eb[m - 1 : -1]
        num = -2.0 * np.dot(efm, ebm)
        den = float(np.dot(efm, efm) + np.dot(ebm, ebm))
        if den == 0:
            break
        km = num / den
        a_new = np.zeros(m + 1)
        a_new[0] = 1.0
        for j in range(1, m):
            a_new[j] = a[j] + km * a[m - j]
        a_new[m] = km
        a = np.zeros(order + 1)
        a[: m + 1] = a_new
        pe *= 1.0 - km * km
        ef_old = ef.copy()
        ef[m:] = ef_old[m:] + km * eb[m - 1 : -1]
        eb[m:] = eb[m - 1 : -1] + km * ef_old[m:]

    freqs = np.linspace(0, fs / 2, nfft // 2 + 1)
    omega = 2 * np.pi * freqs / fs
    ar_coeffs = a[: order + 1]
    denom = np.zeros(len(omega), dtype=complex)
    for k, ak in enumerate(ar_coeffs):
        denom += ak * np.exp(-1j * omega * k)
    psd = pe / (np.abs(denom) ** 2 + 1e-30)

    return SignalResult(
        name="burg_psd",
        filtered=psd,
        fs=fs,
        n_samples=len(psd),
        extra={"freqs": freqs, "order": order, "ar_coefficients": ar_coeffs},
    )


pburg = burg_psd


def cheatsheet() -> str:
    return "burg_psd({}) -> Burg autoregressive power spectral density estimation."
