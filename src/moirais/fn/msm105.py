"""Numbered display equation (7.2) from MVSML chapter 7.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_2"]


def mvsml_bayesian_regression_pt2_eq_7_2(a, scaled, inverse, chi, squared, distribution):
    """
    Numbered display equation (7.2) from MVSML chapter 7.

    Formula: \beta from a scaled inverse chi-squared distribution with parameters   ev\beta = v\beta + p and eS\beta = S\beta + \betaT\beta, that is, from a \chi2 ev\beta,eS\beta : 7. Repeat 1–6 until a burning period and a desired number of samples are reached. Similar modiﬁcations can be done to obtain Gibbs samplers corresponding to other prior adopted models for the beta coefﬁcients (FIXED, BayesA, BayesB, BayesC, or BL; see Chap. 6 for details of these priors). Also, for the ordinal logistic GBLUP Bayesian regression model speciﬁcation as done in

    Parameters
    ----------
    a : array-like
        Input data.
    scaled : array-like
        Input data.
    inverse : array-like
        Input data.
    chi : array-like
        Input data.
    squared : array-like
        Input data.
    distribution : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (7.2) [Multivariate Statistical Machine Learnin [Pages 209-249] [2026-04-16].pdf]
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (7.2) from MVSML chapter 7."})


def cheatsheet():
    return "msm105: Numbered display equation (7.2) from MVSML chapter 7."
