"""Estrada index -- sum exp eigenvalues."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_estrada_index"]


def sgt_estrada_index(A):
    """
    Estrada index -- sum exp eigenvalues

    Formula: EE(G) = Σ_i exp(λ_i)

    Parameters
    ----------
    A : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: EE

    References
    ----------
    Estrada (2000)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Estrada index -- sum exp eigenvalues"})


def cheatsheet():
    return "sgtest: Estrada index -- sum exp eigenvalues"
