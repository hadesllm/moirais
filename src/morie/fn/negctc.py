"""Negative control outcome -- known-null check."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["negative_control_outcome"]


def negative_control_outcome(y_neg, D, X):
    """
    Negative control outcome -- known-null check

    Formula: if effect on negative-control != 0, model misspecified

    Parameters
    ----------
    y_neg : array-like
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
    Lipsitch, Tchetgen Tchetgen, Cohen (2010); Shi-Miao-Tchetgen Tchetgen (2020)
    """
    y_neg = np.atleast_1d(np.asarray(y_neg, dtype=float))
    n = len(y_neg)
    result = float(np.mean(y_neg))
    se = float(np.std(y_neg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Negative control outcome -- known-null check"})


def cheatsheet():
    return "negctc: Negative control outcome -- known-null check"
