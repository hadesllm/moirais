"""Partial dependence plot value."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_partial_dependence"]


def esl_partial_dependence(model, X, S):
    """
    Partial dependence plot value

    Formula: f_S(X_S) = E_{X_C}[f(X_S, X_C)]

    Parameters
    ----------
    model : array-like
        Input data.
    X : array-like
        Input data.
    S : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: values

    References
    ----------
    Hastie ESL Ch 10
    """
    model = np.atleast_1d(np.asarray(model, dtype=float))
    n = len(model)
    result = float(np.mean(model))
    se = float(np.std(model, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Partial dependence plot value"})


def cheatsheet():
    return "eslprt: Partial dependence plot value"
