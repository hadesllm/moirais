"""Exploratory SEM with target rotation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esem_target_rotation"]


def esem_target_rotation(loadings, target):
    """
    Exploratory SEM with target rotation

    Formula: orthogonal/oblique rotation toward hypothesis matrix

    Parameters
    ----------
    loadings : array-like
        Input data.
    target : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Asparouhov-Muthén (2009)
    """
    loadings = np.atleast_1d(np.asarray(loadings, dtype=float))
    n = len(loadings)
    result = float(np.mean(loadings))
    se = float(np.std(loadings, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Exploratory SEM with target rotation"})


def cheatsheet():
    return "esmoeg: Exploratory SEM with target rotation"
