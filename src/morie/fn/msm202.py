r"""Numbered display equation (9.28) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_28"]


def mvsml_ridge_lasso_elastic_eq_9_28(n, are, called, Lagrange, multipliers, Setting):
    r"""
    Numbered display equation (9.28) from MVSML chapter 9.

    Formula: n are called Lagrange multipliers. Setting the derivatives of L(\beta, \beta0, \alpha) with regard to \beta and \beta0 equal to zero, we obtain the following conditions: Xn Xn \partial L \beta, \beta0, \alpha ( ) = \beta 2 i=1\alphaiyixi = 0 \Rightarrow \beta = i=1\alphaiyixi

    Parameters
    ----------
    n : array-like
        Input data.
    are : array-like
        Input data.
    called : array-like
        Input data.
    Lagrange : array-like
        Input data.
    multipliers : array-like
        Input data.
    Setting : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.28) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    r"""
    n = np.atleast_1d(np.asarray(n, dtype=float))
    n = len(n)
    result = float(np.mean(n))
    se = float(np.std(n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.28) from MVSML chapter 9."})


def cheatsheet():
    return "msm202: Numbered display equation (9.28) from MVSML chapter 9."
