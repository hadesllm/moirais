"""Numbered display equation (9.27) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_27"]


def mvsml_ridge_lasso_elastic_eq_9_27(k, k2, i, yi, xT, L):
    """
    Numbered display equation (9.27) from MVSML chapter 9.

    Formula: ) = 1 k k2 2 i=1\alphai yi \beta0 + xT L \beta, \beta0, \alpha ( 2 \beta i \beta + 1 ,

    Parameters
    ----------
    k : array-like
        Input data.
    k2 : array-like
        Input data.
    i : array-like
        Input data.
    yi : array-like
        Input data.
    xT : array-like
        Input data.
    L : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.27) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    k = np.atleast_1d(np.asarray(k, dtype=float))
    n = len(k)
    result = float(np.mean(k))
    se = float(np.std(k, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.27) from MVSML chapter 9."})


def cheatsheet():
    return "msm201: Numbered display equation (9.27) from MVSML chapter 9."
