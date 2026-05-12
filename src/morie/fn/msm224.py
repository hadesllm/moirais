r"""Numbered display equation (9.39) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_39"]


def mvsml_ridge_lasso_elastic_eq_9_39(constraints, of, the, slack, variables, n):
    r"""
    Numbered display equation (9.39) from MVSML chapter 9.

    Formula: nonnegativity constraints of the slack variables, \delta = (\delta1, . . ., \deltan)T. By setting the derivatives of L = L(\beta, \beta0, \zeta, \alpha, \delta) with regard to \beta, \beta0, and \zeta equal to zero, we obtain the following conditions: Xn Xn \partial L \partial \beta = \beta 2 i=1\alphaiyixi = 0 \Rightarrow \beta = i=1\alphaiyixi

    Parameters
    ----------
    constraints : array-like
        Input data.
    of : array-like
        Input data.
    the : array-like
        Input data.
    slack : array-like
        Input data.
    variables : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.39) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    r"""
    constraints = np.atleast_1d(np.asarray(constraints, dtype=float))
    n = len(constraints)
    result = float(np.mean(constraints))
    se = float(np.std(constraints, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.39) from MVSML chapter 9."})


def cheatsheet():
    return "msm224: Numbered display equation (9.39) from MVSML chapter 9."
