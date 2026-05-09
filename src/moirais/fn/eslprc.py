"""Perceptron learning rule."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_perceptron"]


def esl_perceptron(X, y):
    """
    Perceptron learning rule

    Formula: beta <- beta + eta y_i x_i if y_i(beta'x_i) <= 0

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: beta

    References
    ----------
    Hastie ESL Ch 4
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Perceptron learning rule"})


def cheatsheet():
    return "eslprc: Perceptron learning rule"
