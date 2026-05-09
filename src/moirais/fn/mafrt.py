"""Freeman-Tukey arcsine transform for proportions."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_freeman_tukey"]


def ma_freeman_tukey(x, n):
    """
    Freeman-Tukey arcsine transform for proportions

    Formula: FT(p) = arcsin sqrt(x/(n+1)) + arcsin sqrt((x+1)/(n+1))

    Parameters
    ----------
    x : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ft, var

    References
    ----------
    Freeman & Tukey (1950)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Freeman-Tukey arcsine transform for proportions"})


def cheatsheet():
    return "mafrt: Freeman-Tukey arcsine transform for proportions"
