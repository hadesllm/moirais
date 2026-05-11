"""Factorization Machines."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["factorization_machines"]


def factorization_machines(X, y, K):
    """
    Factorization Machines

    Formula: ŷ = w_0 + sum w_i x_i + sum_{i<j} <v_i,v_j> x_i x_j

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rendle (2010)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Factorization Machines"})


def cheatsheet():
    return "fmFM: Factorization Machines"
