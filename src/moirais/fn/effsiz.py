"""Effective sample size n / DEFF."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["effective_sample_size"]


def effective_sample_size(n, deff):
    """
    Effective sample size n / DEFF

    Formula: n_eff = n / DEFF

    Parameters
    ----------
    n : array-like
        Input data.
    deff : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kish (1965)
    """
    n = np.atleast_1d(np.asarray(n, dtype=float))
    n = len(n)
    result = float(np.mean(n))
    se = float(np.std(n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Effective sample size n / DEFF"})


def cheatsheet():
    return "effsiz: Effective sample size n / DEFF"
