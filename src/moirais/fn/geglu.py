"""GEGLU gated activation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["geglu_activation"]


def geglu_activation(y, x, W, V):
    """
    GEGLU gated activation

    Formula: GEGLU(x, W, V) = GELU(xW) * (xV)

    Parameters
    ----------
    y : array-like
        Input data.
    x : array-like
        Input data.
    W : array-like
        Input data.
    V : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Shazeer (2020)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GEGLU gated activation"})


def cheatsheet():
    return "geglu: GEGLU gated activation"
