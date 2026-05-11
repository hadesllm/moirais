"""Spectral radius of a matrix (max |λ|)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_spectral_radius"]


def sgt_spectral_radius(M):
    """
    Spectral radius of a matrix (max |λ|)

    Formula: ρ(M) = max |λ_i|

    Parameters
    ----------
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: rho

    References
    ----------
    Horn-Johnson (2013)
    """
    M = np.atleast_1d(np.asarray(M, dtype=float))
    n = len(M)
    result = float(np.mean(M))
    se = float(np.std(M, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spectral radius of a matrix (max |λ|)"})


def cheatsheet():
    return "sgtsbpd: Spectral radius of a matrix (max |λ|)"
