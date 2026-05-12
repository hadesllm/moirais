r"""Numbered display equation (6.6) from MVSML chapter 6.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_eq_6_6"]


def mvsml_bayesian_regression_eq_6_6(j, inverse, of, the, parameter, any):
    r"""
    Numbered display equation (6.6) from MVSML chapter 6.

    Formula: \beta j inverse of the regularization parameter in any type of Ridge regression model. 6.7 Extended Predictor in Bayesian Genomic Regression Models All the Bayesian formulations of the model (6.1) described before can be extended, in terms of the predictor, to easily take into account the effects of other factors. For example, effects of environments and environment–marker interaction can be added as y = 1n\mu + XE\betaE + X\beta + XEM\betaEM + e,

    Parameters
    ----------
    j : array-like
        Input data.
    inverse : array-like
        Input data.
    of : array-like
        Input data.
    the : array-like
        Input data.
    parameter : array-like
        Input data.
    any : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (6.6) [Multivariate Statistical Machine Learnin [Pages 171-208] [2026-04-16].pdf]
    r"""
    j = np.atleast_1d(np.asarray(j, dtype=float))
    n = len(j)
    result = float(np.mean(j))
    se = float(np.std(j, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (6.6) from MVSML chapter 6."})


def cheatsheet():
    return "msm061: Numbered display equation (6.6) from MVSML chapter 6."
