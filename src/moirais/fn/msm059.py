"""Numbered display equation (6.1) from MVSML chapter 6.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_eq_6_1"]


def mvsml_bayesian_regression_eq_6_1(BGLR, y, ETA, nIter, burnIn, df0):
    """
    Numbered display equation (6.1) from MVSML chapter 6.

    Formula: = BGLR(y=y, ETA = ETA, nIter = 1e4, burnIn = 1e3, df0 = v, S0 = S, probIn = \pip0, counts = ϕ0, R2 = R2) and ETA = list( list( model = ‘BayesB’, X=X1 ) ) A = BGLR(y=y, ETA = ETA, nIter = 1e4, burnIn = 1e3, df0 = v, rate0 = r, shape0 = s, probIn = \pip0, counts = ϕ0, R2 = R2) 6.6 Genomic-Enabled Prediction Bayesian Lasso Model Another variant of the model

    Parameters
    ----------
    BGLR : array-like
        Input data.
    y : array-like
        Input data.
    ETA : array-like
        Input data.
    nIter : array-like
        Input data.
    burnIn : array-like
        Input data.
    df0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (6.1) [Multivariate Statistical Machine Learnin [Pages 171-208] [2026-04-16].pdf]
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (6.1) from MVSML chapter 6."})


def cheatsheet():
    return "msm059: Numbered display equation (6.1) from MVSML chapter 6."
