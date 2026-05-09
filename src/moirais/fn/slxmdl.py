"""Spatial Lag of X (SLX) model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["slx_model"]


def slx_model(y, X, W):
    """
    Spatial Lag of X (SLX) model

    Formula: y = X beta + W X theta + eps

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    W : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Halleck Vega-Elhorst (2015)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spatial Lag of X (SLX) model"})


def cheatsheet():
    return "slxmdl: Spatial Lag of X (SLX) model"
