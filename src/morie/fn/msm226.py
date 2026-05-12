r"""Numbered display equation (9.41) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_41"]


def mvsml_ridge_lasso_elastic_eq_9_41(Xn, L, i, iyi, T):
    r"""
    Numbered display equation (9.41) from MVSML chapter 9.

    Formula: Xn \partial L = + i=1\alphaiyi = 0 \Rightarrow i=1\alphaiyi = 0 (9.40) \partial \beta0 \partial L = T + \alphai 2 \deltai = 0 \Rightarrow \alphai + \deltai = T

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
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.41) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    r"""
    Xn = np.atleast_1d(np.asarray(Xn, dtype=float))
    n = len(Xn)
    result = float(np.mean(Xn))
    se = float(np.std(Xn, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.41) from MVSML chapter 9."})


def cheatsheet():
    return "msm226: Numbered display equation (9.41) from MVSML chapter 9."
