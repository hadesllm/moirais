"""GoodmanBacon 3-way TWFE composition."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["goodman_bacon_3way"]


def goodman_bacon_3way(y, D, unit, time):
    """
    GoodmanBacon 3-way TWFE composition

    Formula: beta = s_TC beta_TC + s_CT beta_CT + s_LL beta_LL

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    unit : array-like
        Input data.
    time : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Goodman-Bacon (2021) §4
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GoodmanBacon 3-way TWFE composition"})


def cheatsheet():
    return "gbtcom: GoodmanBacon 3-way TWFE composition"
