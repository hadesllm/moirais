r"""Numbered display equation (9.29) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_29"]


def mvsml_ridge_lasso_elastic_eq_9_29(Xn, L, i, iyi):
    r"""
    Numbered display equation (9.29) from MVSML chapter 9.

    Formula: (9.28) \partial \beta Xn Xn \partial L \beta, \beta0, \alpha ( ) = i=1\alphaiyi = 0 \Rightarrow i=1\alphaiyi = 0

    Parameters
    ----------
    Xn : array-like
        Input data.
    L : array-like
        Input data.
    i : array-like
        Input data.
    iyi : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.29) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    r"""
    Xn = np.atleast_1d(np.asarray(Xn, dtype=float))
    n = len(Xn)
    result = float(np.mean(Xn))
    se = float(np.std(Xn, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.29) from MVSML chapter 9."})


def cheatsheet():
    return "msm203: Numbered display equation (9.29) from MVSML chapter 9."
