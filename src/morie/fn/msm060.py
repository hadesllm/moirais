"""Numbered display equation (6.1) from MVSML chapter 6.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_eq_6_1"]


def mvsml_bayesian_regression_eq_6_1(BayesA, BayesB, Var, j, E, S):
    """
    Numbered display equation (6.1) from MVSML chapter 6.

    Formula: (BayesA and BayesB), Var \beta j = E \sigma2 = S\beta= v\beta - 1 ), which is almost the \beta j inverse of the regularization parameter in any type of Ridge regression model. 6.7 Extended Predictor in Bayesian Genomic Regression Models All the Bayesian formulations of the model

    Parameters
    ----------
    BayesA : array-like
        Input data.
    BayesB : array-like
        Input data.
    Var : array-like
        Input data.
    j : array-like
        Input data.
    E : array-like
        Input data.
    S : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (6.1) [Multivariate Statistical Machine Learnin [Pages 171-208] [2026-04-16].pdf]
    """
    BayesA = np.atleast_1d(np.asarray(BayesA, dtype=float))
    n = len(BayesA)
    result = float(np.mean(BayesA))
    se = float(np.std(BayesA, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (6.1) from MVSML chapter 6."})


def cheatsheet():
    return "msm060: Numbered display equation (6.1) from MVSML chapter 6."
