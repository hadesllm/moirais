"""Finite-population correction (1 - n/N)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["finite_population_corr"]


def finite_population_corr(n, N):
    """
    Finite-population correction (1 - n/N)

    Formula: variance scale factor

    Parameters
    ----------
    n : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cochran (1977)
    """
    n = np.atleast_1d(np.asarray(n, dtype=float))
    n = len(n)
    result = float(np.mean(n))
    se = float(np.std(n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Finite-population correction (1 - n/N)"})


def cheatsheet():
    return "fpcadj: Finite-population correction (1 - n/N)"
