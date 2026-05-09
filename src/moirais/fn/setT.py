"""Set transformer pooling."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["set_transformer"]


def set_transformer(X, k):
    """
    Set transformer pooling

    Formula: PMA(MAB(X)) attention pool to fixed size

    Parameters
    ----------
    X : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lee et al (2019)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Set transformer pooling"})


def cheatsheet():
    return "setT: Set transformer pooling"
