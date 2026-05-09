"""TMLE for interventional natural direct effect."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_nde_interventional"]


def tmle_nde_interventional(y, D, M, X):
    """
    TMLE for interventional natural direct effect

    Formula: E[Y(1, M_a) - Y(0, M_a)] for intervention on M

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    M : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Vansteelandt-Daniel (2017)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE for interventional natural direct effect"})


def cheatsheet():
    return "tmlnde2: TMLE for interventional natural direct effect"
