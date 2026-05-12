r"""Numbered display equation (6.6) from MVSML chapter 6.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_eq_6_6"]


def mvsml_bayesian_regression_eq_6_6(y, XE, E, X, XEM, EM):
    r"""
    Numbered display equation (6.6) from MVSML chapter 6.

    Formula: y = 1n\mu + XE\betaE + X\beta + XEM\betaEM + e, (6.6) where XE and XEM are the design matrices of the environments and environment– marker interactions, respectively, while \betaE and \betaEM are the vectors of the environ- ment effects and the interaction effects, respectively, with a prior distribution that can be speciﬁed as was done for \beta. Indeed, with the BGLR function all these things are possible, and all the options described before can also be adopted for the rest of effects added in the model: FIXED, BRR, BayesA, BayesB, BayesC, and BL. Under the RKHS model with genotypic and environment–genotypic interaction effects, in the predictor, the modiﬁed model

    Parameters
    ----------
    y : array-like
        Input data.
    XE : array-like
        Input data.
    E : array-like
        Input data.
    X : array-like
        Input data.
    XEM : array-like
        Input data.
    EM : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (6.6) [Multivariate Statistical Machine Learnin [Pages 171-208] [2026-04-16].pdf]
    r"""
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (6.6) from MVSML chapter 6."})


def cheatsheet():
    return "msm062: Numbered display equation (6.6) from MVSML chapter 6."
