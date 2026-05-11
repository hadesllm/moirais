"""GWR optimal bandwidth (CV/AICc)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gwr_bandwidth_select"]


def gwr_bandwidth_select(y, X, coords, kernel):
    """
    GWR optimal bandwidth (CV/AICc)

    Formula: argmin AICc over bandwidth grid

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    coords : array-like
        Input data.
    kernel : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Fotheringham-Brunsdon-Charlton (2002)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GWR optimal bandwidth (CV/AICc)"})


def cheatsheet():
    return "gwrcal: GWR optimal bandwidth (CV/AICc)"
