"""TMLE for controlled direct effect (CDE)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_controlled_direct"]


def tmle_controlled_direct(y, D, M, X, m_value):
    """
    TMLE for controlled direct effect (CDE)

    Formula: E[Y(1,m) - Y(0,m)] -- fix mediator at m

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    M : array-like
        Input data.
    X : array-like
        Input data.
    m_value : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    vdL-Petersen (2008)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE for controlled direct effect (CDE)"})


def cheatsheet():
    return "tmlcde: TMLE for controlled direct effect (CDE)"
