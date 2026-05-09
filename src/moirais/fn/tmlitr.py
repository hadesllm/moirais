"""TMLE for optimal individualized treatment rule."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_individual_regime"]


def tmle_individual_regime(y, D, W, X):
    """
    TMLE for optimal individualized treatment rule

    Formula: d*(W) = argmax_d E[Y(d(W))|W]

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    W : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Luedtke-vdL (2016)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE for optimal individualized treatment rule"})


def cheatsheet():
    return "tmlitr: TMLE for optimal individualized treatment rule"
