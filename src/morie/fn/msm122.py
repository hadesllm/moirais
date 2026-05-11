"""Numbered display equation (7.11) from MVSML chapter 7.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_11"]


def mvsml_bayesian_regression_pt2_eq_7_11(Poisson, regression, Given, vector, covariates, xi):
    """
    Numbered display equation (7.11) from MVSML chapter 7.

    Formula: Poisson regression. Given vector covariates xi = (xi1, . . ., xip)T, the Poisson log-linear regression modeled the number of events Yi, as a Poisson random variable with mass density ) = \lambday i exp \lambdai ( ) P Yi = yjxi ( , y = 0, 1, 2, . . . ,

    Parameters
    ----------
    Poisson : array-like
        Input data.
    regression : array-like
        Input data.
    Given : array-like
        Input data.
    vector : array-like
        Input data.
    covariates : array-like
        Input data.
    xi : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (7.11) [Multivariate Statistical Machine Learnin [Pages 209-249] [2026-04-16].pdf]
    """
    Poisson = np.atleast_1d(np.asarray(Poisson, dtype=float))
    n = len(Poisson)
    result = float(np.mean(Poisson))
    se = float(np.std(Poisson, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (7.11) from MVSML chapter 7."})


def cheatsheet():
    return "msm122: Numbered display equation (7.11) from MVSML chapter 7."
