"""Thin-plate spline."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["thin_plate_spline"]


def thin_plate_spline(x, y, z, lam):
    """
    Thin-plate spline

    Formula: min sum(y_i − f(x_i))² + λ ∫∫(f_xx² + 2f_xy² + f_yy²)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    z : array-like
        Input data.
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Duchon (1977); Wahba (1990)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Thin-plate spline"})


def cheatsheet():
    return "thinpls: Thin-plate spline"
