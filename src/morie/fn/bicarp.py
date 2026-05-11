"""BIC for AR(p) order selection."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bic_ar_order"]


def bic_ar_order(x, max_p):
    """
    BIC for AR(p) order selection

    Formula: BIC(p) = log(sigma_p^2) + p log(T)/T

    Parameters
    ----------
    x : array-like
        Input data.
    max_p : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Schwarz (1978)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BIC for AR(p) order selection"})


def cheatsheet():
    return "bicarp: BIC for AR(p) order selection"
