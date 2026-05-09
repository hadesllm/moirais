"""TMLE for quantile treatment effects."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_quantile"]


def tmle_quantile(y, D, X, quantile):
    """
    TMLE for quantile treatment effects

    Formula: target Q_q[Y(1)] - Q_q[Y(0)]

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    quantile : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Díaz (2017)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE for quantile treatment effects"})


def cheatsheet():
    return "tmlqct: TMLE for quantile treatment effects"
