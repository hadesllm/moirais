"""Wiener filter (frequency-domain, optimal linear).

Reference: Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
Analysis*, 3rd ed. IEEE/Wiley, Chapter 9.
"""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult

__all__ = ['wienf']
def wienf(
    x: np.ndarray,
    fs: float = 1.0,
    *,
    noise_psd: np.ndarray | None = None,
    nperseg: int = 256,
    noise_fraction: float = 0.1,
) -> SignalResult:
    """Frequency-domain Wiener filter for noise reduction.

    .. math::

        H(f) = \\frac{P_{ss}(f)}{P_{ss}(f) + P_{nn}(f)}

    Parameters
    ----------
    x : array-like
        1-D noisy input signal.
    fs : float
        Sampling frequency in Hz.
    noise_psd : array-like or None
        Noise PSD estimate. If None, estimated from lowest spectral bins.
    nperseg : int
        Segment length for PSD estimation.
    noise_fraction : float
        Fraction of lowest-power bins used for noise floor estimation.

    Returns
    -------
    SignalResult
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    X = np.fft.rfft(x)
    power = np.abs(X) ** 2 / n

    if noise_psd is None:
        sorted_p = np.sort(power)
        k = max(1, int(len(sorted_p) * noise_fraction))
        noise_est = np.full_like(power, np.mean(sorted_p[:k]))
    else:
        noise_est = np.asarray(noise_psd, dtype=float).ravel()
        if len(noise_est) != len(power):
            noise_est = np.interp(
                np.linspace(0, 1, len(power)),
                np.linspace(0, 1, len(noise_est)),
                noise_est,
            )

    gain = power / (power + noise_est + 1e-12)
    Y = X * gain
    filtered = np.fft.irfft(Y, n=n)

    return SignalResult(
        name="wienf",
        filtered=filtered,
        fs=fs,
        n_samples=n,
        extra={"wiener_gain": gain, "noise_psd": noise_est},
    )


def cheatsheet() -> str:
    return "wienf({}) -> Wiener filter (optimal linear)."
