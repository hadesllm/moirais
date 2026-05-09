# moirais.fn — function file (hadesllm/moirais)
"""Logistic sigmoid activation function."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_sigmoid"]


def geron_sigmoid(t):
    """
    Logistic sigmoid activation function

    Formula: sigma(t) = 1 / (1 + exp(-t))

    Parameters
    ----------
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: p

    References
    ----------
    Géron Ch 4
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    n = len(t)
    result = float(np.mean(t))
    se = float(np.std(t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Logistic sigmoid activation function"})


def cheatsheet():
    return "hmsigm: Logistic sigmoid activation function"
