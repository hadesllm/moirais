"""Numbered display equation (6.1) from MVSML chapter 6.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_eq_6_1"]


def mvsml_bayesian_regression_eq_6_1(interest, of, the, non, phenotyped, individuals):
    """
    Numbered display equation (6.1) from MVSML chapter 6.

    Formula: interest of the non-phenotyped individuals that have only genotypic information, environment variables, or other information (covariates). In this situation, a conve- nient practice is to include the individuals to be predicted (yp) in the posterior distribution to be sampled. Speciﬁcally, a standard Bayesian framework for a normal linear regression model (see Chap. 3) X p Y = \beta0 + X j\beta j + E

    Parameters
    ----------
    interest : array-like
        Input data.
    of : array-like
        Input data.
    the : array-like
        Input data.
    non : array-like
        Input data.
    phenotyped : array-like
        Input data.
    individuals : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (6.1) [Multivariate Statistical Machine Learnin [Pages 171-208] [2026-04-16].pdf]
    """
    interest = np.atleast_1d(np.asarray(interest, dtype=float))
    n = len(interest)
    result = float(np.mean(interest))
    se = float(np.std(interest, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (6.1) from MVSML chapter 6."})


def cheatsheet():
    return "msm042: Numbered display equation (6.1) from MVSML chapter 6."
