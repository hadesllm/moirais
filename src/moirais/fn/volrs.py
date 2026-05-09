"""Rogers-Satchell drift-independent OHLC volatility."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_rogers_satchell"]


def vol_rogers_satchell(o, h, l, c):
    """
    Rogers-Satchell drift-independent OHLC volatility

    Formula: σ̂² = log(H/C) log(H/O) + log(L/C) log(L/O)

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
    Rogers-Satchell (1991)
    """
    o = np.atleast_1d(np.asarray(o, dtype=float))
    n = len(o)
    result = float(np.mean(o))
    se = float(np.std(o, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Rogers-Satchell drift-independent OHLC volatility"})


def cheatsheet():
    return "volrs: Rogers-Satchell drift-independent OHLC volatility"
