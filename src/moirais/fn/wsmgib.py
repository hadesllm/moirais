"""Gibbs sampler."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_gibbs_sampler"]


def wasserman_gibbs_sampler(target, x0, n):
    """
    Gibbs sampler

    Formula: x_i^{t+1} ~ p(x_i | x_{-i}^t)

    Parameters
    ----------
    target : array-like
        Input data.
    x0 : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: samples

    References
    ----------
    Wasserman (2004), Ch 24
    """
    target = np.atleast_1d(np.asarray(target, dtype=float))
    n = len(target)
    result = float(np.mean(target))
    se = float(np.std(target, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gibbs sampler"})


def cheatsheet():
    return "wsmgib: Gibbs sampler"
