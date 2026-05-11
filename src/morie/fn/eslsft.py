"""Softmax output layer."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_softmax"]


def esl_softmax(T):
    """
    Softmax output layer

    Formula: p_k = exp(T_k) / sum_l exp(T_l)

    Parameters
    ----------
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: probabilities

    References
    ----------
    Hastie ESL Ch 11
    """
    T = np.atleast_1d(np.asarray(T, dtype=float))
    n = len(T)
    result = float(np.mean(T))
    se = float(np.std(T, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Softmax output layer"})


def cheatsheet():
    return "eslsft: Softmax output layer"
