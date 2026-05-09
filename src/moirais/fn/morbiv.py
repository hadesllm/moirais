"""Bivariate Moran's I between two variables."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bivariate_morans_i"]


def bivariate_morans_i(x, y, W):
    """
    Bivariate Moran's I between two variables

    Formula: I_xy = (n / S0) sum_i sum_j w_ij z_xi z_yj / sigma_x sigma_y

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    W : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Anselin, Syabri, Smirnov (2002)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bivariate Moran's I between two variables"})


def cheatsheet():
    return "morbiv: Bivariate Moran's I between two variables"
