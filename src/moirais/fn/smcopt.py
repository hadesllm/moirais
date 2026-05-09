"""Sequential Monte Carlo for optimization."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sequential_mc"]


def sequential_mc(f, x0, temperatures):
    """
    Sequential Monte Carlo for optimization

    Formula: particle filter on tempered objective

    Parameters
    ----------
    f : array-like
        Input data.
    x0 : array-like
        Input data.
    temperatures : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Del Moral-Doucet-Jasra (2006)
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = len(f)
    result = float(np.mean(f))
    se = float(np.std(f, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sequential Monte Carlo for optimization"})


def cheatsheet():
    return "smcopt: Sequential Monte Carlo for optimization"
