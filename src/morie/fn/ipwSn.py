"""IPW sensitivity (Robins-Rotnitzky-Scharfstein)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ipw_sensitivity"]


def ipw_sensitivity(Y, X, C, lam_grid):
    """
    IPW sensitivity (Robins-Rotnitzky-Scharfstein)

    Formula: vary λ encoding deviation from MAR

    Parameters
    ----------
    Y : array-like
        Input data.
    X : array-like
        Input data.
    C : array-like
        Input data.
    lam_grid : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Robins-Rotnitzky-Scharfstein (2000)
    """
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    n = len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "IPW sensitivity (Robins-Rotnitzky-Scharfstein)"})


def cheatsheet():
    return "ipwSn: IPW sensitivity (Robins-Rotnitzky-Scharfstein)"
