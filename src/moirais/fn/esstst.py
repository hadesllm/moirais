"""Effective sample size of weighted population."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["effective_sample_size"]


def effective_sample_size(weights):
    """
    Effective sample size of weighted population

    Formula: ESS = (sum w)^2 / sum w^2

    Parameters
    ----------
    weights : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kish (1965)
    """
    weights = np.atleast_1d(np.asarray(weights, dtype=float))
    n = len(weights)
    result = float(np.mean(weights))
    se = float(np.std(weights, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Effective sample size of weighted population"})


def cheatsheet():
    return "esstst: Effective sample size of weighted population"
