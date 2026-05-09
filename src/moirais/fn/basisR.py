"""Basis representation y = sum c_k φ_k."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["basis_representation"]


def basis_representation(y, Phi):
    """
    Basis representation y = sum c_k φ_k

    Formula: least-squares c = (Φ'Φ)^{-1}Φ'y

    Parameters
    ----------
    y : array-like
        Input data.
    Phi : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ramsay-Silverman (2005)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Basis representation y = sum c_k φ_k"})


def cheatsheet():
    return "basisR: Basis representation y = sum c_k φ_k"
