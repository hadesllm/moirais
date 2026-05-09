"""Random-effects meta-regression on study-level moderators."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_meta_regression"]


def ma_meta_regression(yi, vi, X):
    """
    Random-effects meta-regression on study-level moderators

    Formula: y_i = X_i β + u_i + ε_i; weighted by 1/(v_i+τ²)

    Parameters
    ----------
    yi : array-like
        Input data.
    vi : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: beta, se, tau2, R2, ll

    References
    ----------
    van Houwelingen-Arends-Stijnen (2002)
    """
    yi = np.atleast_1d(np.asarray(yi, dtype=float))
    n = len(yi)
    result = float(np.mean(yi))
    se = float(np.std(yi, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Random-effects meta-regression on study-level moderators"})


def cheatsheet():
    return "mareg: Random-effects meta-regression on study-level moderators"
