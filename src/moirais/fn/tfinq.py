"""Time-frequency uncertainty principle measurement."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Train yourself to let go of everything you fear to lose."


def time_freq_uncertainty(x, fs=1.0, **kwargs) -> DescriptiveResult:
    """Measure Heisenberg time-frequency uncertainty.

    Computes the product of time spread (sigma_t) and frequency
    spread (sigma_f). The lower bound is 1/(4*pi).

    Parameters
    ----------
    x : array-like
        Input signal.
    fs : float
        Sampling frequency.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    t = np.arange(n) / fs
    energy = np.sum(x**2)
    if energy == 0:
        raise ValueError("Signal has zero energy.")
    t_mean = np.sum(t * x**2) / energy
    sigma_t = np.sqrt(np.sum((t - t_mean) ** 2 * x**2) / energy)
    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(n, d=1.0 / fs)
    power = np.abs(X) ** 2
    f_mean = np.sum(freqs * power) / np.sum(power)
    sigma_f = np.sqrt(np.sum((freqs - f_mean) ** 2 * power) / np.sum(power))
    product = sigma_t * sigma_f
    lower_bound = 1.0 / (4.0 * np.pi)
    return DescriptiveResult(
        name="time_freq_uncertainty",
        value=float(product),
        extra={
            "sigma_t": float(sigma_t),
            "sigma_f": float(sigma_f),
            "product": float(product),
            "lower_bound": float(lower_bound),
            "above_bound": bool(product >= lower_bound - 1e-10),
        },
    )


tfinq = time_freq_uncertainty


def cheatsheet() -> str:
    return "time_freq_uncertainty({}) -> Time-frequency uncertainty principle measurement."
