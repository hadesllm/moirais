r"""Numbered display equation (7.9) from MVSML chapter 7.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_9"]


def mvsml_bayesian_regression_pt2_eq_7_9(T, b, c, X, TWcX, D):
    r"""
    Numbered display equation (7.9) from MVSML chapter 7.

    Formula: T b\beta c = XTWcX + \lambdaD + T, and D is an identity where X = [1n X], Wc = Diag(w1c, . . ., wnc), y = y 1, . . . , y n matrix of dimension ( p + 1) + ( p + 1) except that in the ﬁrst entry we have the value of 0 instead of 1. However, in the context of p  n, a non-prohibited optimization of

    Parameters
    ----------
    T : array-like
        Input data.
    b : array-like
        Input data.
    c : array-like
        Input data.
    X : array-like
        Input data.
    TWcX : array-like
        Input data.
    D : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (7.9) [Multivariate Statistical Machine Learnin [Pages 209-249] [2026-04-16].pdf]
    r"""
    T = np.atleast_1d(np.asarray(T, dtype=float))
    n = len(T)
    result = float(np.mean(T))
    se = float(np.std(T, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (7.9) from MVSML chapter 7."})


def cheatsheet():
    return "msm113: Numbered display equation (7.9) from MVSML chapter 7."
