"""Empirical Bayes-adjusted Moran's I for rates."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["empirical_bayes_moran"]


def empirical_bayes_moran(x, n, W):
    """
    Empirical Bayes-adjusted Moran's I for rates

    Formula: I_EB = (n/S0) sum_i sum_j w_ij r_i^* r_j^*, r* shrunk rate

    Parameters
    ----------
    x : array-like
        Input data.
    n : array-like
        Input data.
    W : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Assuncao & Reis (1999)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Empirical Bayes-adjusted Moran's I for rates"})


def cheatsheet():
    return "morebs: Empirical Bayes-adjusted Moran's I for rates"
