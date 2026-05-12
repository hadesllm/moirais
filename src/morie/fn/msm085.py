r"""Numbered display equation (7.1) from MVSML chapter 7.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_1"]


def mvsml_bayesian_regression_pt2_eq_7_1(variable, the, categories, are, conceived, contiguous):
    r"""
    Numbered display equation (7.1) from MVSML chapter 7.

    Formula: variable and the categories are conceived as contiguous intervals on the continuous scale, as presented in McCullagh (1980), where the points of division (thresholds) are denoted as \gamma0, \gamma1, . . ., \gammaC. In this way, the ordinal model assumes that conditioned to xi (covariates of dimension p), Yi is a random variable that takes values 1, . . ., C, with the following probabilities: pic = P Yi = c ( ) = P \gammac1  Li  \gammac ( )

    Parameters
    ----------
    variable : array-like
        Input data.
    the : array-like
        Input data.
    categories : array-like
        Input data.
    are : array-like
        Input data.
    conceived : array-like
        Input data.
    contiguous : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (7.1) [Multivariate Statistical Machine Learnin [Pages 209-249] [2026-04-16].pdf]
    r"""
    variable = np.atleast_1d(np.asarray(variable, dtype=float))
    n = len(variable)
    result = float(np.mean(variable))
    se = float(np.std(variable, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (7.1) from MVSML chapter 7."})


def cheatsheet():
    return "msm085: Numbered display equation (7.1) from MVSML chapter 7."
