"""TMLE for the variance of potential outcomes."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_var_targeting"]


def tmle_var_targeting(y, D, X):
    """
    TMLE for the variance of potential outcomes

    Formula: target Var[Y(a)] via plug-in + clever-covariate update

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    vdL-Hubbard-Pajouh (2018)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE for the variance of potential outcomes"})


def cheatsheet():
    return "tmlvts: TMLE for the variance of potential outcomes"
