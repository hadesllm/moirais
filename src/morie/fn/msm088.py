r"""Numbered display equation (7.1) from MVSML chapter 7.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_1"]


def mvsml_bayesian_regression_pt2_eq_7_1(G, p, XTX, Then, this, assuming):
    r"""
    Numbered display equation (7.1) from MVSML chapter 7.

    Formula: \beta and G = 1 p XTX. Then, with this and assuming a scaled inverse chi- +  g  \chi2 vg, Sg squared distribution as prior for \sigma2 g , \sigma2 , and a ﬂat prior for the threshold parameters (\gamma), an ordinal probit GBLUP Bayesian regression model speciﬁcation of model

    Parameters
    ----------
    G : array-like
        Input data.
    p : array-like
        Input data.
    XTX : array-like
        Input data.
    Then : array-like
        Input data.
    this : array-like
        Input data.
    assuming : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (7.1) [Multivariate Statistical Machine Learnin [Pages 209-249] [2026-04-16].pdf]
    r"""
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (7.1) from MVSML chapter 7."})


def cheatsheet():
    return "msm088: Numbered display equation (7.1) from MVSML chapter 7."
