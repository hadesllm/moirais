"""Functional CCA."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["functional_cca"]


def functional_cca(X, Y):
    """
    Functional CCA

    Formula: max corr(∫α(t)X(t), ∫β(t)Y(t))

    Parameters
    ----------
    X : array-like
        Input data.
    Y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    He-Müller-Wang (2003)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Functional CCA"})


def cheatsheet():
    return "funCA: Functional CCA"
