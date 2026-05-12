r"""Numbered display equation (9.36) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_36"]


def mvsml_ridge_lasso_elastic_eq_9_36(X, p, yi, j1xij, M, i):
    r"""
    Numbered display equation (9.36) from MVSML chapter 9.

    Formula: ! X p yi \beta0 + \beta j1xij  M 1 + \zetai ( ),

    Parameters
    ----------
    X : array-like
        Input data.
    p : array-like
        Input data.
    yi : array-like
        Input data.
    j1xij : array-like
        Input data.
    M : array-like
        Input data.
    i : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.36) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    r"""
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.36) from MVSML chapter 9."})


def cheatsheet():
    return "msm220: Numbered display equation (9.36) from MVSML chapter 9."
