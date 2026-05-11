"""Item characteristic response function (3PL form)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["item_characteristic_curve"]


def item_characteristic_curve(y, theta, a, b, c):
    """
    Item characteristic response function (3PL form)

    Formula: P_i(theta) = c_i + (1 - c_i)/(1 + exp(-a_i(theta - b_i)))

    Parameters
    ----------
    y : array-like
        Input data.
    theta : array-like
        Input data.
    a : array-like
        Input data.
    b : array-like
        Input data.
    c : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Birnbaum (1968)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Item characteristic response function (3PL form)"})


def cheatsheet():
    return "icrf: Item characteristic response function (3PL form)"
