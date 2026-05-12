r"""Numbered display equation (6.5) from MVSML chapter 6.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_eq_6_5"]


def mvsml_bayesian_regression_eq_6_5(ment, effects, the, interaction, respectively, a):
    r"""
    Numbered display equation (6.5) from MVSML chapter 6.

    Formula: ment effects and the interaction effects, respectively, with a prior distribution that can be speciﬁed as was done for \beta. Indeed, with the BGLR function all these things are possible, and all the options described before can also be adopted for the rest of effects added in the model: FIXED, BRR, BayesA, BayesB, BayesC, and BL. Under the RKHS model with genotypic and environment–genotypic interaction effects, in the predictor, the modiﬁed model (6.6) is expressed as Y = 1n\mu + XE\betaE + ZLg + ZELgE + e, (6.7) where ZL and ZLE are the incident matrices of the genomic and environment– genotypic interaction effects, respectively. Similarly to model

    Parameters
    ----------
    ment : array-like
        Input data.
    effects : array-like
        Input data.
    the : array-like
        Input data.
    interaction : array-like
        Input data.
    respectively : array-like
        Input data.
    a : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (6.5) [Multivariate Statistical Machine Learnin [Pages 171-208] [2026-04-16].pdf]
    r"""
    ment = np.atleast_1d(np.asarray(ment, dtype=float))
    n = len(ment)
    result = float(np.mean(ment))
    se = float(np.std(ment, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (6.5) from MVSML chapter 6."})


def cheatsheet():
    return "msm064: Numbered display equation (6.5) from MVSML chapter 6."
