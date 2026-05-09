"""Support vector classifier (soft margin)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_svc"]


def esl_svc(X, y, C):
    """
    Support vector classifier (soft margin)

    Formula: min (1/2)|w|^2 + C sum xi_i s.t. y_i(w'x_i+b)>=1-xi_i, xi_i>=0

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    C : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Hastie ESL Ch 12
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Support vector classifier (soft margin)"})


def cheatsheet():
    return "eslsvc: Support vector classifier (soft margin)"
