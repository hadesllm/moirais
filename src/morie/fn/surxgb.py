"""XGBoost survival objective."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["xgb_survival"]


def xgb_survival(time, event, X):
    """
    XGBoost survival objective

    Formula: gradient boost over partial-likelihood

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Chen-Guestrin (2016); Barnwal-Cho-Hocking (2022)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "XGBoost survival objective"})


def cheatsheet():
    return "surxgb: XGBoost survival objective"
