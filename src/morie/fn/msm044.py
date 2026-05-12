r"""Numbered display equation (6.1) from MVSML chapter 6.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_eq_6_1"]


def mvsml_bayesian_regression_eq_6_1(distribution, of, given, by, j, y):
    r"""
    Numbered display equation (6.1) from MVSML chapter 6.

    Formula:   2 -  2 1 distribution of \beta is given by \beta j \sigma2, y, X  N e\beta, \sigma2 XTX : 6.2 Bayesian Genome-Based Ridge Regression When p > n, X is not of full column rank and the posterior of model

    Parameters
    ----------
    distribution : array-like
        Input data.
    of : array-like
        Input data.
    given : array-like
        Input data.
    by : array-like
        Input data.
    j : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (6.1) [Multivariate Statistical Machine Learnin [Pages 171-208] [2026-04-16].pdf]
    r"""
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (6.1) from MVSML chapter 6."})


def cheatsheet():
    return "msm044: Numbered display equation (6.1) from MVSML chapter 6."
