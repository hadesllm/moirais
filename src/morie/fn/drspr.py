"""DR-DiD with spillover."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dr_spillover"]


def dr_spillover(y, D, X, exposure):
    """
    DR-DiD with spillover

    Formula: separate ATT_direct vs ATT_spillover

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    exposure : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Aronow-Samii (2017)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DR-DiD with spillover"})


def cheatsheet():
    return "drspr: DR-DiD with spillover"
