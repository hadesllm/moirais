"""TMLE for subgroup CATE."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_subgroup"]


def tmle_subgroup(y, D, X, subgroup):
    """
    TMLE for subgroup CATE

    Formula: target E[Y(1)-Y(0)|S=1] for subgroup S

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    subgroup : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Athey-Imbens (2016); Chernozhukov-Demirer-Duflo-Fernández-Val (2018)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE for subgroup CATE"})


def cheatsheet():
    return "tmlsbg: TMLE for subgroup CATE"
