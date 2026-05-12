# morie.fn -- function file (hadesllm/morie)
"""Hawking radiation temperature."""

__all__ = ["hawkr"]

import numpy as np


def hawkr(
    M: float,
    G: float = 6.67430e-11,
    c: float = 299792458.0,
    hbar: float = 1.0545718e-34,
    kB: float = 1.380649e-23,
) -> dict:
    r"""
    Compute the Hawking temperature of a Schwarzschild black hole.

    .. math::

        T_H = \\frac{\\hbar c^3}{8 \\pi G M k_B}

    Parameters
    ----------
    M : float
        Black hole mass (kg, > 0).
    G : float
        Gravitational constant.
    c : float
        Speed of light.
    hbar : float
        Reduced Planck constant.
    kB : float
        Boltzmann constant.

    Returns
    -------
    dict
        Keys: temperature_K, r_schwarzschild, luminosity_W,
        evaporation_time_s, peak_wavelength_m.
    """
    if M <= 0:
        raise ValueError("Mass must be > 0.")

    T_H = hbar * c ** 3 / (8.0 * np.pi * G * M * kB)

    rs = 2.0 * G * M / c ** 2

    sigma_sb = 2.0 * np.pi ** 5 * kB ** 4 / (15.0 * hbar ** 3 * c ** 6)
    sigma_sb_real = 5.670374419e-8
    A = 4.0 * np.pi * rs ** 2
    L = sigma_sb_real * A * T_H ** 4

    t_evap = 5120.0 * np.pi * G ** 2 * M ** 3 / (hbar * c ** 4)

    lambda_peak = 2.8977719e-3 / (T_H + 1e-300)

    return {
        "temperature_K": float(T_H),
        "r_schwarzschild": float(rs),
        "luminosity_W": float(L),
        "evaporation_time_s": float(t_evap),
        "peak_wavelength_m": float(lambda_peak),
    }
