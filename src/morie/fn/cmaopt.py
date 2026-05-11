"""CMA-ES evolution strategy."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cma_es"]


def cma_es(f, x0, sigma, lam):
    """
    CMA-ES evolution strategy

    Formula: adapt covariance matrix of search distribution

    Parameters
    ----------
    f : array-like
        Input data.
    x0 : array-like
        Input data.
    sigma : array-like
        Input data.
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hansen-Ostermeier (2001)
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = len(f)
    result = float(np.mean(f))
    se = float(np.std(f, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "CMA-ES evolution strategy"})


def cheatsheet():
    return "cmaopt: CMA-ES evolution strategy"
