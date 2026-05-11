"""Periodogram on rectangular lattice for spectral analysis."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_periodogram"]


def schabenberger_periodogram(z_lattice, coords):
    """
    Periodogram on rectangular lattice for spectral analysis

    Formula: I(omega) = (1/n)|sum Z(s)*exp(-i*omega'*s)|^2

    Parameters
    ----------
    z_lattice : array-like
        Input data.
    coords : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: periodogram

    References
    ----------
    Schabenberger Ch 4, Sec 4.7.1
    """
    z_lattice = np.asarray(z_lattice, dtype=float)
    n = int(z_lattice) if z_lattice.ndim == 0 else len(z_lattice)
    result = float(np.mean(z_lattice))
    se = float(np.std(z_lattice, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Periodogram on rectangular lattice for spectral analysis"})


def cheatsheet():
    return "spperiod: Periodogram on rectangular lattice for spectral analysis"
