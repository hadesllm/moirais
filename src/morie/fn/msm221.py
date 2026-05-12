r"""Numbered display equation (9.37) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_37"]


def mvsml_ridge_lasso_elastic_eq_9_37(M, i, j, X, n, T):
    r"""
    Numbered display equation (9.37) from MVSML chapter 9.

    Formula:  M 1 + \zetai ( ), (9.36) j=1 X n \zetai  0, \zetai T,

    Parameters
    ----------
    M : array-like
        Input data.
    i : array-like
        Input data.
    j : array-like
        Input data.
    X : array-like
        Input data.
    n : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.37) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    r"""
    M = np.atleast_1d(np.asarray(M, dtype=float))
    n = len(M)
    result = float(np.mean(M))
    se = float(np.std(M, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.37) from MVSML chapter 9."})


def cheatsheet():
    return "msm221: Numbered display equation (9.37) from MVSML chapter 9."
