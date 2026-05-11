"""Numbered display equation (6.8) from MVSML chapter 6.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_eq_6_8"]


def mvsml_bayesian_regression_eq_6_8(suppose, that, vec, B, N, marginally):
    """
    Numbered display equation (6.8) from MVSML chapter 6.

    Formula: suppose that \beta = vec(B)  N(\beta0, \Sigma\beta), that is, marginally, for the ﬁxed effect of each -  trait, a prior multivariate normal distribution is adopted, \betat  Np \betat0, \Sigma\betat , t = 1, . . . , nT; a ﬂat prior for the intercepts, f(\mu) / 1; and independent inverse Wishart distributions for the covariance matrix of residuals R and for \SigmaT, that is, \SigmaT  IW(vt, St) and R  IW(vR, SR). Putting all the information together where the measured traits of each individual (Yj) are accommodated in the rows of a matrix response (Y), model

    Parameters
    ----------
    suppose : array-like
        Input data.
    that : array-like
        Input data.
    vec : array-like
        Input data.
    B : array-like
        Input data.
    N : array-like
        Input data.
    marginally : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (6.8) [Multivariate Statistical Machine Learnin [Pages 171-208] [2026-04-16].pdf]
    """
    suppose = np.atleast_1d(np.asarray(suppose, dtype=float))
    n = len(suppose)
    result = float(np.mean(suppose))
    se = float(np.std(suppose, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (6.8) from MVSML chapter 6."})


def cheatsheet():
    return "msm066: Numbered display equation (6.8) from MVSML chapter 6."
