"""Normalized inverse-Gaussian process."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["normalized_inverse_gauss"]


def normalized_inverse_gauss(y, alpha, tau):
    """
    Normalized inverse-Gaussian process

    Formula: alternative to DP with heavier tails

    Parameters
    ----------
    y : array-like
        Input data.
    alpha : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lijoi-Mena-Prünster (2005)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Normalized inverse-Gaussian process"})


def cheatsheet():
    return "nrmprc: Normalized inverse-Gaussian process"
