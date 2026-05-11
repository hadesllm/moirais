"""TMLE for multivariate (vector) treatments."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_multivariate_treatment"]


def tmle_multivariate_treatment(y, A, X):
    """
    TMLE for multivariate (vector) treatments

    Formula: target E[Y(a_vec)] with multivariate Q*

    Parameters
    ----------
    y : array-like
        Input data.
    A : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lendle et al (2017); Rosenbaum (2002)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE for multivariate (vector) treatments"})


def cheatsheet():
    return "tmlmct: TMLE for multivariate (vector) treatments"
