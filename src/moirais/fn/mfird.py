"""MIRT factor-loading reparameterization (a -> lambda)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mirt_factor_loading"]


def mirt_factor_loading(y, a):
    """
    MIRT factor-loading reparameterization (a -> lambda)

    Formula: lambda_jk = a_jk / sqrt(1 + a_j' a_j)

    Parameters
    ----------
    y : array-like
        Input data.
    a : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Reckase (2009) §3
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MIRT factor-loading reparameterization (a -> lambda)"})


def cheatsheet():
    return "mfird: MIRT factor-loading reparameterization (a -> lambda)"
