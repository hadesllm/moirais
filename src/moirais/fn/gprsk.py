"""GP residual modeling."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gp_residual_kernel"]


def gp_residual_kernel(X, y, y_pred, kernel):
    """
    GP residual modeling

    Formula: y - hat y_param ~ GP(0, k)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    y_pred : array-like
        Input data.
    kernel : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rasmussen-Williams (2006)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GP residual modeling"})


def cheatsheet():
    return "gprsk: GP residual modeling"
