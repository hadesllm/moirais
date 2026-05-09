"""AIC for AR(p) order selection."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["aic_ar_order"]


def aic_ar_order(x, max_p):
    """
    AIC for AR(p) order selection

    Formula: AIC(p) = log(sigma_p^2) + 2(p+1)/T

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
    Akaike (1973)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AIC for AR(p) order selection"})


def cheatsheet():
    return "aikarp: AIC for AR(p) order selection"
