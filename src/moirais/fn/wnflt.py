"""Wiener filter for optimal noise reduction.

Reference: Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
Analysis*, 3rd ed. IEEE/Wiley, Chapter 3.
"""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult

_QUOTE = "The noise is strong with this one. --"


def wiener_filter(
    x: np.ndarray,
    fs: float = 1.0,
    *,
    noise_psd: np.ndarray | None = None,
    nperseg: int = 256,
) -> SignalResult:
    """Apply a Wiener filter for optimal noise reduction.

    The Wiener filter minimizes the mean-square error between the
    estimated and desired signal in the frequency domain:

    .. math::

        H(f) = \\frac{P_{xx}(f)}{P_{xx}(f) + P_{nn}(f)}

    where :math:`P_{xx}` is the signal PSD and :math:`P_{nn}` is the
    noise PSD.

    Parameters
    ----------
    x : array-like
        1-D noisy input signal.
    fs : float
        Sampling frequency in Hz (default 1.0).
    noise_psd : array-like or None
        Noise power spectral density estimate.  If *None*, the noise
        floor is estimated from the lowest 10% of spectral bins.
    nperseg : int
        Segment length for PSD estimation (default 256).

    Returns
    -------
    SignalResult
        ``filtered`` contains the denoised signal, ``extra`` has
        ``wiener_gain`` and ``noise_psd``.

    References
    ----------
    Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
    Analysis*, 3rd ed. IEEE/Wiley, Chapter 3.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)

    X = np.fft.rfft(x)
    power = np.abs(X) ** 2 / n

    if noise_psd is None:
        sorted_power = np.sort(power)
        noise_floor = np.mean(sorted_power[: max(1, len(sorted_power) // 10)])
        noise_est = np.full_like(power, noise_floor)
    else:
        noise_est = np.asarray(noise_psd, dtype=float).ravel()
        if len(noise_est) != len(power):
            from scipy.interpolate import interp1d

            old_f = np.linspace(0, 1, len(noise_est))
            new_f = np.linspace(0, 1, len(power))
            noise_est = interp1d(old_f, noise_est, fill_value="extrapolate")(new_f)

    gain = power / (power + noise_est + 1e-12)
    Y = X * gain
    filtered = np.fft.irfft(Y, n=n)

    return SignalResult(
        name="wiener_filter",
        filtered=filtered,
        fs=fs,
        n_samples=n,
        extra={"wiener_gain": gain, "noise_psd": noise_est},
    )


wnflt = wiener_filter


def cheatsheet() -> str:
    return "wiener_filter({}) -> Wiener filter for optimal noise reduction."
