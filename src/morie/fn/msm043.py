r"""Numbered display equation (6.2) from MVSML chapter 6.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_eq_6_2"]


def mvsml_bayesian_regression_eq_6_2(X, j, E, a, random, error):
    r"""
    Numbered display equation (6.2) from MVSML chapter 6.

    Formula: X j\beta j + E (6.1) j=1 with E a random error with normal distribution with mean 0 and variance \sigma2, is fully speciﬁed by assuming the next non-informative prior distribution: \beta and log(\sigma) approximately independent and locally uniform. -  f \beta, \sigma2 / \sigma-2

    Parameters
    ----------
    X : array-like
        Input data.
    j : array-like
        Input data.
    E : array-like
        Input data.
    a : array-like
        Input data.
    random : array-like
        Input data.
    error : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (6.2) [Multivariate Statistical Machine Learnin [Pages 171-208] [2026-04-16].pdf]
    r"""
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (6.2) from MVSML chapter 6."})


def cheatsheet():
    return "msm043: Numbered display equation (6.2) from MVSML chapter 6."
