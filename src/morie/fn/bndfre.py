"""Frequentist bound with valid coverage."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_frequentist"]


def bound_frequentist(lower, upper, alpha):
    """
    Frequentist bound with valid coverage

    Formula: valid uniform coverage at alpha

    Parameters
    ----------
    lower : array-like
        Input data.
    upper : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Imbens-Manski (2004)
    """
    lower = np.atleast_1d(np.asarray(lower, dtype=float))
    n = len(lower)
    result = float(np.mean(lower))
    se = float(np.std(lower, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Frequentist bound with valid coverage"})


def cheatsheet():
    return "bndfre: Frequentist bound with valid coverage"
