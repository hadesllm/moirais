"""Grand-mean centering for level-1 or level-2 covariate."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["grand_mean_centering"]


def grand_mean_centering(y):
    """
    Grand-mean centering for level-1 or level-2 covariate

    Formula: x_ij_CGM = x_ij - xbar..

    Parameters
    ----------
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Enders & Tofighi (2007)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Grand-mean centering for level-1 or level-2 covariate"})


def cheatsheet():
    return "gmcent: Grand-mean centering for level-1 or level-2 covariate"
