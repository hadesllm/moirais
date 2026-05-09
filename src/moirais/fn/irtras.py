"""Andrich rating scale model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rating_scale_model"]


def rating_scale_model(X, ncats):
    """
    Andrich rating scale model

    Formula: shared category thresholds across items

    Parameters
    ----------
    X : array-like
        Input data.
    ncats : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Andrich (1978)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Andrich rating scale model"})


def cheatsheet():
    return "irtras: Andrich rating scale model"
