# morie.fn — function file (hadesllm/morie)
"""Planck distribution (blackbody radiation)."""

__all__ = ["plank"]

import numpy as np


def plank(
    nu: np.ndarray,
    T: float,
    h: float = 6.62607015e-34,
    c: float = 299792458.0,
    kB: float = 1.380649e-23,
) -> dict:
    r"""
    Compute the Planck spectral radiance for blackbody radiation.

    .. math::

        B(\\nu, T) = \\frac{2 h \\nu^3}{c^2}
                     \\frac{1}{\\exp(h\\nu / k_B T) - 1}

    Parameters
    ----------
    nu : np.ndarray
        Frequencies (Hz).
    T : float
        Temperature (K, > 0).
    h : float
        Planck constant (J*s).
    c : float
        Speed of light (m/s).
    kB : float
        Boltzmann constant (J/K).

    Returns
    -------
    dict
        Keys: spectral_radiance (W/sr/m^2/Hz), peak_frequency (Hz),
        total_power (W/m^2, Stefan-Boltzmann), wien_peak.
    """
    nu = np.asarray(nu, dtype=float)
    if T <= 0:
        raise ValueError("Temperature must be > 0.")

    x = h * nu / (kB * T)
    x = np.clip(x, 0, 500)
    B = (2.0 * h * nu ** 3 / c ** 2) / (np.exp(x) - 1.0 + 1e-300)

    nu_peak = 2.8214393 * kB * T / h

    sigma = 2.0 * np.pi ** 5 * kB ** 4 / (15.0 * h ** 3 * c ** 2)
    total_power = sigma * T ** 4

    return {
        "spectral_radiance": B,
        "peak_frequency": float(nu_peak),
        "total_power": float(total_power),
        "wien_peak": float(nu_peak),
    }
