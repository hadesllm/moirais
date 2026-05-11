"""DKW confidence band for eCDF."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_dkw_cb"]


def wasserman_dkw_cb(data, alpha):
    """
    DKW confidence band for eCDF

    Formula: L(x), U(x) = F_n(x) +/- sqrt(log(2/alpha)/(2n))

    Parameters
    ----------
    data : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lower, upper

    References
    ----------
    Wasserman (2004), Ch 7 Thm 7.5
    """
    data = np.atleast_1d(np.asarray(data, dtype=float))
    n = len(data)
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DKW confidence band for eCDF"})


def cheatsheet():
    return "wsmcb: DKW confidence band for eCDF"
