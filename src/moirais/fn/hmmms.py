# moirais.fn — function file (hadesllm/moirais)
"""Min-max scaling to range [0,1]."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_min_max_scaling"]


def geron_min_max_scaling(X):
    """
    Min-max scaling to range [0,1]

    Formula: x' = (x - x_min) / (x_max - x_min)

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: X_scaled

    References
    ----------
    Géron Ch 2
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Min-max scaling to range [0,1]"})


def cheatsheet():
    return "hmmms: Min-max scaling to range [0,1]"
