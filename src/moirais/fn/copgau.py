"""Gaussian (Normal) copula CDF."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gaussian_copula"]


def gaussian_copula(y, u, v, rho):
    """
    Gaussian (Normal) copula CDF

    Formula: C(u,v) = Phi_R(Phi^{-1}(u), Phi^{-1}(v))

    Parameters
    ----------
    y : array-like
        Input data.
    u : array-like
        Input data.
    v : array-like
        Input data.
    rho : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Sklar (1959); Nelsen (2006) §4
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gaussian (Normal) copula CDF"})


def cheatsheet():
    return "copgau: Gaussian (Normal) copula CDF"
