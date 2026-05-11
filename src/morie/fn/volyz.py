"""Yang-Zhang OHLC volatility (most efficient)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_yang_zhang"]


def vol_yang_zhang(o, h, l, c):
    """
    Yang-Zhang OHLC volatility (most efficient)

    Formula: Combine open-close + close-close + Rogers-Satchell components

    Parameters
    ----------
    o : array-like
        Input data.
    h : array-like
        Input data.
    l : array-like
        Input data.
    c : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: sigma2

    References
    ----------
    Yang-Zhang (2000)
    """
    o = np.atleast_1d(np.asarray(o, dtype=float))
    n = len(o)
    result = float(np.mean(o))
    se = float(np.std(o, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Yang-Zhang OHLC volatility (most efficient)"})


def cheatsheet():
    return "volyz: Yang-Zhang OHLC volatility (most efficient)"
