# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bose-Einstein distribution."""

__all__ = ["bosed"]

import numpy as np


def bosed(
    energies: np.ndarray,
    mu: float = 0.0,
    T: float = 300.0,
    kB: float = 1.380649e-23,
) -> dict:
    r"""
    Compute the Bose-Einstein distribution.

    .. math::

        n(E) = \\frac{1}{\\exp\\left(\\frac{E - \\mu}{k_B T}\\right) - 1}

    For photons/phonons, mu = 0.

    Parameters
    ----------
    energies : np.ndarray
        Energy values (J, must be > mu for convergence).
    mu : float
        Chemical potential (J). Must be < min(energies).
    T : float
        Temperature (K, > 0).
    kB : float
        Boltzmann constant (J/K).

    Returns
    -------
    dict
        Keys: occupation (ndarray), mean_occupation, total_energy.
    """
    energies = np.asarray(energies, dtype=float)
    if T <= 0:
        raise ValueError("Temperature must be > 0.")
    if np.any(energies <= mu):
        raise ValueError("All energies must be > mu for Bose-Einstein.")

    x = (energies - mu) / (kB * T)
    x = np.clip(x, 1e-15, 500)
    n = 1.0 / (np.exp(x) - 1.0)

    return {
        "occupation": n,
        "mean_occupation": float(np.mean(n)),
        "total_energy": float(np.sum(energies * n)),
    }
