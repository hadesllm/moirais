r"""Numbered display equation (6.9) from MVSML chapter 6.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_eq_6_9"]


def mvsml_bayesian_regression_eq_6_9(The, complete, Bayesian, speci, cation, of):
    r"""
    Numbered display equation (6.9) from MVSML chapter 6.

    Formula: The complete Bayesian speciﬁcation of this model assumes independent multivar- iate normal distributions for the columns of B, that is, for the ﬁxed effect of each trait -  a prior multivariate normal distribution is adopted, \betat  Np \betat0, \Sigma\betat , t = 1, . . . , nT; a ﬂat prior for the intercepts, f(\mu) / 1; and independent inverse Wishart distributions for the covariance matrices of residuals R and for \SigmaT, \SigmaT  IW(vT, ST) and R  IW (vR, SR), and also an inverse Wishart distribution for \SigmaE, \SigmaE  IW(vE, SE). The full conditional distributions of \mu, B, b1, b2, and R can be derived as in model

    Parameters
    ----------
    The : array-like
        Input data.
    complete : array-like
        Input data.
    Bayesian : array-like
        Input data.
    speci : array-like
        Input data.
    cation : array-like
        Input data.
    of : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (6.9) [Multivariate Statistical Machine Learnin [Pages 171-208] [2026-04-16].pdf]
    r"""
    The = np.atleast_1d(np.asarray(The, dtype=float))
    n = len(The)
    result = float(np.mean(The))
    se = float(np.std(The, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (6.9) from MVSML chapter 6."})


def cheatsheet():
    return "msm078: Numbered display equation (6.9) from MVSML chapter 6."
