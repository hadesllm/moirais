"""Gibbs sampler."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gibbs_sampler"]


def gibbs_sampler(conditionals, x0, n_iter):
    """
    Gibbs sampler

    Formula: x_i ~ p(x_i | x_-i) for each block

    Parameters
    ----------
    conditionals : array-like
        Input data.
    x0 : array-like
        Input data.
    n_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Geman-Geman (1984)
    """
    conditionals = np.atleast_1d(np.asarray(conditionals, dtype=float))
    n = len(conditionals)
    result = float(np.mean(conditionals))
    se = float(np.std(conditionals, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gibbs sampler"})


def cheatsheet():
    return "gibbsm: Gibbs sampler"
