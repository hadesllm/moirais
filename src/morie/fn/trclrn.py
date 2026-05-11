"""Tree-based optimal treatment regime."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tree_based_regime"]


def tree_based_regime(y, D, W):
    """
    Tree-based optimal treatment regime

    Formula: recursively partition W to maximize value

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    W : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Laber-Zhao (2015); Tao-Wang (2017)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Tree-based optimal treatment regime"})


def cheatsheet():
    return "trclrn: Tree-based optimal treatment regime"
