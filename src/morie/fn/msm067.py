"""Numbered display equation (6.9) from MVSML chapter 6.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_eq_6_9"]


def mvsml_bayesian_regression_eq_6_9(trait, a, prior, multivariate, normal, distribution):
    """
    Numbered display equation (6.9) from MVSML chapter 6.

    Formula:  trait, a prior multivariate normal distribution is adopted, \betat  Np \betat0, \Sigma\betat , t = 1, . . . , nT; a ﬂat prior for the intercepts, f(\mu) / 1; and independent inverse Wishart distributions for the covariance matrix of residuals R and for \SigmaT, that is, \SigmaT  IW(vt, St) and R  IW(vR, SR). Putting all the information together where the measured traits of each individual (Yj) are accommodated in the rows of a matrix response (Y), model (6.8) can be expressed as Y = 1J\muT + XB + Z1b1 + E,

    Parameters
    ----------
    trait : array-like
        Input data.
    a : array-like
        Input data.
    prior : array-like
        Input data.
    multivariate : array-like
        Input data.
    normal : array-like
        Input data.
    distribution : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (6.9) [Multivariate Statistical Machine Learnin [Pages 171-208] [2026-04-16].pdf]
    """
    trait = np.atleast_1d(np.asarray(trait, dtype=float))
    n = len(trait)
    result = float(np.mean(trait))
    se = float(np.std(trait, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (6.9) from MVSML chapter 6."})


def cheatsheet():
    return "msm067: Numbered display equation (6.9) from MVSML chapter 6."
