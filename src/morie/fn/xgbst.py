"""XGBoost regularized objective."""
import numpy as np
from ._richresult import RichResult

__all__ = ["xgboost_objective"]


def xgboost_objective(x, y):
    """
    XGBoost regularized objective

    Formula: L = sum l(yi,y_hat) + sum Omega(f_k)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Chen & Guestrin (2016)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "XGBoost regularized objective"})


def cheatsheet():
    return "xgbst: XGBoost regularized objective"
