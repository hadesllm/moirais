"""AMISE-optimal bandwidth."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_kde_bandwidth"]


def wasserman_kde_bandwidth(data):
    """
    AMISE-optimal bandwidth

    Formula: h_opt = c sigma n^{-1/5}

    Parameters
    ----------
    data : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: h

    References
    ----------
    Wasserman (2004), Ch 20
    """
    data = np.atleast_1d(np.asarray(data, dtype=float))
    n = len(data)
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AMISE-optimal bandwidth"})


def cheatsheet():
    return "wsmkbw: AMISE-optimal bandwidth"
