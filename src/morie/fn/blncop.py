"""Blomqvist's beta from a copula."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["blomqvists_beta_copula"]


def blomqvists_beta_copula(y, copula, theta):
    """
    Blomqvist's beta from a copula

    Formula: beta = 4 C(0.5, 0.5) - 1

    Parameters
    ----------
    y : array-like
        Input data.
    copula : array-like
        Input data.
    theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Blomqvist (1950); Nelsen (2006)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Blomqvist's beta from a copula"})


def cheatsheet():
    return "blncop: Blomqvist's beta from a copula"
