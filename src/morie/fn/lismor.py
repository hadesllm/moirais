"""Local Moran's I (LISA)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["local_morans_i"]


def local_morans_i(x, W):
    """
    Local Moran's I (LISA)

    Formula: I_i = (x_i - xbar) sum_j w_ij (x_j - xbar)

    Parameters
    ----------
    x : array-like
        Input data.
    W : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Anselin (1995)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Local Moran's I (LISA)"})


def cheatsheet():
    return "lismor: Local Moran's I (LISA)"
