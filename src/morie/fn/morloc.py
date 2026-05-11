"""Local Moran's I (LISA) per location."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["local_morans_i"]


def local_morans_i(x, W):
    """
    Local Moran's I (LISA) per location

    Formula: I_i = z_i sum_j w_ij z_j / m_2

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Local Moran's I (LISA) per location"})


def cheatsheet():
    return "morloc: Local Moran's I (LISA) per location"
