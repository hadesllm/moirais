# morie.fn — function file (hadesllm/morie)
"""Fermi-Dirac distribution."""

__all__ = ["fermd"]

import numpy as np


def fermd(
    energies: np.ndarray,
    mu: float,
    T: float,
    kB: float = 1.380649e-23,
) -> dict:
    r"""
    Compute the Fermi-Dirac distribution.

    .. math::

        f(E) = \\frac{1}{\\exp\\left(\\frac{E - \\mu}{k_B T}\\right) + 1}

    Parameters
    ----------
    energies : np.ndarray
        Energy values (J).
    mu : float
        Chemical potential / Fermi energy (J).
    T : float
        Temperature (K, > 0).
    kB : float
        Boltzmann constant (J/K).

    Returns
    -------
    dict
        Keys: occupation (ndarray), mean_occupation (float),
        total_energy (sum of E*f(E)).
    """
    energies = np.asarray(energies, dtype=float)
    if T <= 0:
        raise ValueError("Temperature must be > 0.")

    x = (energies - mu) / (kB * T)
    x = np.clip(x, -500, 500)
    f = 1.0 / (np.exp(x) + 1.0)

    return {
        "occupation": f,
        "mean_occupation": float(np.mean(f)),
        "total_energy": float(np.sum(energies * f)),
    }
