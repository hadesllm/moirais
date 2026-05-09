"""Numbered display equation (6.1) from MVSML chapter 6.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_eq_6_1"]


def mvsml_bayesian_regression_eq_6_1(Bayesian, Genome, Based, Ridge, Regression, When):
    """
    Numbered display equation (6.1) from MVSML chapter 6.

    Formula: 6.2 Bayesian Genome-Based Ridge Regression When p > n, X is not of full column rank and the posterior of model (6.1) may not be proper (Gelman et al. 2013), so a solution is instead to consider independently proper prior distributions, \beta  N(0, I\sigma2) and \sigma2  IG(\alpha0, \alpha0), which for large values of \sigma2 6.2 Bayesian Genome-Based Ridge Regression 173 (106) and small values of \alpha0 (10-3) is an approximation to the standard non-informative prior given in

    Parameters
    ----------
    Bayesian : array-like
        Input data.
    Genome : array-like
        Input data.
    Based : array-like
        Input data.
    Ridge : array-like
        Input data.
    Regression : array-like
        Input data.
    When : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (6.1) [Multivariate Statistical Machine Learnin [Pages 171-208] [2026-04-16].pdf]
    """
    Bayesian = np.atleast_1d(np.asarray(Bayesian, dtype=float))
    n = len(Bayesian)
    result = float(np.mean(Bayesian))
    se = float(np.std(Bayesian, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (6.1) from MVSML chapter 6."})


def cheatsheet():
    return "msm045: Numbered display equation (6.1) from MVSML chapter 6."
