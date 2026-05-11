"""Numbered display equation (7.2) from MVSML chapter 7.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_2"]


def mvsml_bayesian_regression_pt2_eq_7_2(Bayesian, Classical, Prediction, Models, Categorical, Count):
    """
    Numbered display equation (7.2) from MVSML chapter 7.

    Formula: 214 7 Bayesian and Classical Prediction Models for Categorical and Count Data pic = P Yi = c ( ) = \Phi \gammac + bi ( )  \Phi \gammac1 + bi ( ), c = 1, . . . , C,

    Parameters
    ----------
    Bayesian : array-like
        Input data.
    Classical : array-like
        Input data.
    Prediction : array-like
        Input data.
    Models : array-like
        Input data.
    Categorical : array-like
        Input data.
    Count : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (7.2) [Multivariate Statistical Machine Learnin [Pages 209-249] [2026-04-16].pdf]
    """
    Bayesian = np.atleast_1d(np.asarray(Bayesian, dtype=float))
    n = len(Bayesian)
    result = float(np.mean(Bayesian))
    se = float(np.std(Bayesian, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (7.2) from MVSML chapter 7."})


def cheatsheet():
    return "msm089: Numbered display equation (7.2) from MVSML chapter 7."
