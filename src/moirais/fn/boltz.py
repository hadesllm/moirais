# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Boltzmann distribution and partition function."""

__all__ = ["boltz"]

import numpy as np


def boltz(
    energies: np.ndarray,
    T: float,
    degeneracies: np.ndarray = None,
    kB: float = 1.380649e-23,
) -> dict:
    """
    Compute the Boltzmann distribution and partition function.

    .. math::

        P_i = \\frac{g_i \\exp(-E_i / k_B T)}{Z}

    .. math::

        Z = \\sum_i g_i \\exp(-E_i / k_B T)

    Parameters
    ----------
    energies : np.ndarray
        Energy levels (J).
    T : float
        Temperature (K, must be > 0).
    degeneracies : np.ndarray, optional
        Degeneracy of each level. Default all 1.
    kB : float
        Boltzmann constant (J/K).

    Returns
    -------
    dict
        Keys: probabilities, partition_function, mean_energy, entropy,
        free_energy.
    """
    energies = np.asarray(energies, dtype=float)
    if T <= 0:
        raise ValueError("Temperature must be > 0.")

    if degeneracies is None:
        g = np.ones_like(energies)
    else:
        g = np.asarray(degeneracies, dtype=float)

    beta = 1.0 / (kB * T)
    exps = g * np.exp(-beta * (energies - energies.min()))
    Z = np.sum(exps)
    probs = exps / Z

    Z_exact = np.sum(g * np.exp(-beta * energies))
    mean_E = np.sum(probs * energies)
    entropy = -kB * np.sum(probs * np.log(probs + 1e-300))
    free_energy = -kB * T * np.log(Z_exact)

    return {
        "probabilities": probs,
        "partition_function": float(Z_exact),
        "mean_energy": float(mean_E),
        "entropy": float(entropy),
        "free_energy": float(free_energy),
    }
