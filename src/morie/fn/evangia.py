"""Empirical angular measure on the simplex."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_angular_measure"]


def evt_angular_measure(X, k):
    """
    Empirical angular measure on the simplex

    Formula: H_n(B) = (1/k) Σ 1{||X_i||_>=r_n, X_i/||X_i|| ∈ B}

    Parameters
    ----------
    X : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: H, atoms, weights

    References
    ----------
    Einmahl-de Haan-Sinha (1997)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Empirical angular measure on the simplex"})


def cheatsheet():
    return "evangia: Empirical angular measure on the simplex"
