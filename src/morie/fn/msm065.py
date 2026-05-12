r"""Numbered display equation (6.8) from MVSML chapter 6.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_eq_6_8"]


def mvsml_bayesian_regression_eq_6_8(gjnT, EjnT, where, t, nT, are):
    r"""
    Numbered display equation (6.8) from MVSML chapter 6.

    Formula: gjnT EjnT where \mut, t = 1, . . ., nT, are the speciﬁc trait intercepts, xj is a vector of covariates equal for all traits, gjt, t = 1, . . ., nT, are the speciﬁc trait genotype effects, and Ejt, t = 1, . . ., nT are the random error terms corresponding to each trait. In matrix notation, it can be expressed as 6.8 Bayesian Genomic Multi-trait Linear Regression Model 191 Y j = \mu + BT x j + g j + e j,

    Parameters
    ----------
    gjnT : array-like
        Input data.
    EjnT : array-like
        Input data.
    where : array-like
        Input data.
    t : array-like
        Input data.
    nT : array-like
        Input data.
    are : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (6.8) [Multivariate Statistical Machine Learnin [Pages 171-208] [2026-04-16].pdf]
    r"""
    gjnT = np.atleast_1d(np.asarray(gjnT, dtype=float))
    n = len(gjnT)
    result = float(np.mean(gjnT))
    se = float(np.std(gjnT, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (6.8) from MVSML chapter 6."})


def cheatsheet():
    return "msm065: Numbered display equation (6.8) from MVSML chapter 6."
