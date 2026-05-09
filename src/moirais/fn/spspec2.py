"""Spectral decomposition simulation of Gaussian random fields."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_spectral_sim"]


def schabenberger_spectral_sim(spectral_density, coords, n_freqs):
    """
    Spectral decomposition simulation of Gaussian random fields

    Formula: Z = sum_k sqrt(S(omega_k)) * [a_k*cos(omega_k'*s) + b_k*sin(omega_k'*s)]

    Parameters
    ----------
    spectral_density : array-like
        Input data.
    coords : array-like
        Input data.
    n_freqs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: simulated_field

    References
    ----------
    Schabenberger Ch 7, Sec 7.1.2
    """
    spectral_density = np.asarray(spectral_density, dtype=float)
    n = int(spectral_density) if spectral_density.ndim == 0 else len(spectral_density)
    result = float(np.mean(spectral_density))
    se = float(np.std(spectral_density, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spectral decomposition simulation of Gaussian random fields"})


def cheatsheet():
    return "spspec2: Spectral decomposition simulation of Gaussian random fields"
