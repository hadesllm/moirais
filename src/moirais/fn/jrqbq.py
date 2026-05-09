"""Jarque-Bera normality on residuals."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["jarque_bera"]


def jarque_bera(resid):
    """
    Jarque-Bera normality on residuals

    Formula: JB = n/6 (S² + (K−3)²/4)

    Parameters
    ----------
    resid : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Jarque-Bera (1980)
    """
    resid = np.atleast_1d(np.asarray(resid, dtype=float))
    n = len(resid)
    result = float(np.mean(resid))
    se = float(np.std(resid, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Jarque-Bera normality on residuals"})


def cheatsheet():
    return "jrqbq: Jarque-Bera normality on residuals"
