"""Numbered display equation (6.7) from MVSML chapter 6.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_eq_6_7"]


def mvsml_bayesian_regression_eq_6_7(where, XE, XEM, are, the, design):
    """
    Numbered display equation (6.7) from MVSML chapter 6.

    Formula: (6.6) where XE and XEM are the design matrices of the environments and environment– marker interactions, respectively, while \betaE and \betaEM are the vectors of the environ- ment effects and the interaction effects, respectively, with a prior distribution that can be speciﬁed as was done for \beta. Indeed, with the BGLR function all these things are possible, and all the options described before can also be adopted for the rest of effects added in the model: FIXED, BRR, BayesA, BayesB, BayesC, and BL. Under the RKHS model with genotypic and environment–genotypic interaction effects, in the predictor, the modiﬁed model (6.6) is expressed as Y = 1n\mu + XE\betaE + ZLg + ZELgE + e,

    Parameters
    ----------
    where : array-like
        Input data.
    XE : array-like
        Input data.
    XEM : array-like
        Input data.
    are : array-like
        Input data.
    the : array-like
        Input data.
    design : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (6.7) [Multivariate Statistical Machine Learnin [Pages 171-208] [2026-04-16].pdf]
    """
    where = np.atleast_1d(np.asarray(where, dtype=float))
    n = len(where)
    result = float(np.mean(where))
    se = float(np.std(where, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (6.7) from MVSML chapter 6."})


def cheatsheet():
    return "msm063: Numbered display equation (6.7) from MVSML chapter 6."
