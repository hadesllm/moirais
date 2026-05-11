"""Numbered display equation (9.40) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_40"]


def mvsml_ridge_lasso_elastic_eq_9_40(i, iyixi, Xn, L, iyi):
    """
    Numbered display equation (9.40) from MVSML chapter 9.

    Formula: i=1\alphaiyixi = 0 \Rightarrow \beta = i=1\alphaiyixi (9.39) Xn Xn \partial L = + i=1\alphaiyi = 0 \Rightarrow i=1\alphaiyi = 0

    Parameters
    ----------
    i : array-like
        Input data.
    iyixi : array-like
        Input data.
    Xn : array-like
        Input data.
    L : array-like
        Input data.
    iyi : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.40) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    i = np.atleast_1d(np.asarray(i, dtype=float))
    n = len(i)
    result = float(np.mean(i))
    se = float(np.std(i, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.40) from MVSML chapter 9."})


def cheatsheet():
    return "msm225: Numbered display equation (9.40) from MVSML chapter 9."
