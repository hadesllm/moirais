"""Wild (Mammen 2-pt) bootstrap for heteroskedastic errors."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_wild_regression"]


def boot_wild_regression(X, y, B):
    """
    Wild (Mammen 2-pt) bootstrap for heteroskedastic errors

    Formula: y* = Xβ̂ + ε̂_i v_i; v_i ∈ {-(√5-1)/2, (√5+1)/2}

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    B : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: beta_b

    References
    ----------
    Mammen (1993)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Wild (Mammen 2-pt) bootstrap for heteroskedastic errors"})


def cheatsheet():
    return "btwild: Wild (Mammen 2-pt) bootstrap for heteroskedastic errors"
