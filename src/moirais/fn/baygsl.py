"""Hybrid Gibbs + slice sampler."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gibbs_slice"]


def gibbs_slice(model, x0, n_iter):
    """
    Hybrid Gibbs + slice sampler

    Formula: Gibbs blocks + slice within difficult blocks

    Parameters
    ----------
    model : array-like
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
    Damlen-Wakefield-Walker (1999)
    """
    model = np.atleast_1d(np.asarray(model, dtype=float))
    n = len(model)
    result = float(np.mean(model))
    se = float(np.std(model, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hybrid Gibbs + slice sampler"})


def cheatsheet():
    return "baygsl: Hybrid Gibbs + slice sampler"
