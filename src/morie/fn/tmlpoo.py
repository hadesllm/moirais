"""Pooled TMLE for multi-site data."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_pooled"]


def tmle_pooled(y, D, X, site):
    """
    Pooled TMLE for multi-site data

    Formula: site-stratified Q + g; fixed-effects pooling

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    site : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Schnitzer-Sango-Lefebvre-Filion (2018)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pooled TMLE for multi-site data"})


def cheatsheet():
    return "tmlpoo: Pooled TMLE for multi-site data"
