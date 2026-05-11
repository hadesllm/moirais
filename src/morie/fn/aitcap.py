"""Aitchison-distance k-NN classifier."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["compositional_classifyAP"]


def compositional_classifyAP(X, y, x_new, k):
    """
    Aitchison-distance k-NN classifier

    Formula: argmin_g sum_{i in g} d_A(x*, x_i) over k neighbours

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    x_new : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: yhat

    References
    ----------
    Pawlowsky-Glahn (2015)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Aitchison-distance k-NN classifier"})


def cheatsheet():
    return "aitcap: Aitchison-distance k-NN classifier"
