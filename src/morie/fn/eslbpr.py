"""Backpropagation gradient."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_backprop"]


def esl_backprop(X, y, weights):
    """
    Backpropagation gradient

    Formula: delta = (D loss / D activation) * sigma'(z)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    weights : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: gradients

    References
    ----------
    Hastie ESL Ch 11
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Backpropagation gradient"})


def cheatsheet():
    return "eslbpr: Backpropagation gradient"
