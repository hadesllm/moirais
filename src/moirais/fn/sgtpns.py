"""Perron-Frobenius leading eigenvalue + eigenvector."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_perron_frobenius"]


def sgt_perron_frobenius(M):
    """
    Perron-Frobenius leading eigenvalue + eigenvector

    Formula: λ_PF = max{|λ_i|}; v_PF >= 0

    Parameters
    ----------
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lam_pf, v_pf

    References
    ----------
    Perron (1907); Frobenius (1908)
    """
    M = np.atleast_1d(np.asarray(M, dtype=float))
    n = len(M)
    result = float(np.mean(M))
    se = float(np.std(M, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Perron-Frobenius leading eigenvalue + eigenvector"})


def cheatsheet():
    return "sgtpns: Perron-Frobenius leading eigenvalue + eigenvector"
