"""Heterogeneous mixing reproduction."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["heterogeneous_mixing"]


def heterogeneous_mixing(contact_matrix, gamma):
    """
    Heterogeneous mixing reproduction

    Formula: R0 = rho(C * D) where D = diag(1/gamma)

    Parameters
    ----------
    contact_matrix : array-like
        Input data.
    gamma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Diekmann-Heesterbeek (2000)
    """
    contact_matrix = np.atleast_1d(np.asarray(contact_matrix, dtype=float))
    n = len(contact_matrix)
    result = float(np.mean(contact_matrix))
    se = float(np.std(contact_matrix, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Heterogeneous mixing reproduction"})


def cheatsheet():
    return "hetmix: Heterogeneous mixing reproduction"
