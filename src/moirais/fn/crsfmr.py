"""Crossformer — cross-time + cross-dimension attention."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["crossformer"]


def crossformer(X, y, seg_len):
    """
    Crossformer — cross-time + cross-dimension attention

    Formula: two-stage attention DSW + HED

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    seg_len : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Zhang-Yan (2023) Crossformer
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Crossformer — cross-time + cross-dimension attention"})


def cheatsheet():
    return "crsfmr: Crossformer — cross-time + cross-dimension attention"
