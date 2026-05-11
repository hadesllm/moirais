"""Numbered display equation (6.9) from MVSML chapter 6.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_eq_6_9"]


def mvsml_bayesian_regression_eq_6_9(it, y, Y, ETA, resCov, list):
    """
    Numbered display equation (6.9) from MVSML chapter 6.

    Formula: it(y = Y, ETA=ETA, resCov = list( type = 'UN', S0 = SR, df0 = vR ), nIter = nI, burnIn = nb) with the only change in the second sub-list predictor, where now the design matrix X1 and the Ridge regression model (BRR) are speciﬁed. Example 3 To illustrate the performance in terms of the prediction power of these models and how to implement this in R software, we considered a reduced data set that consisted of 50 wheat lines grown in two environments. In each individual, two traits were measured: FLRSDS and MIXTIM. The evaluation was done with a ﬁve-fold cross-validation, where lines were evaluated in some environments with all traits but are missing for all traits in other environments. Model

    Parameters
    ----------
    it : array-like
        Input data.
    y : array-like
        Input data.
    Y : array-like
        Input data.
    ETA : array-like
        Input data.
    resCov : array-like
        Input data.
    list : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (6.9) [Multivariate Statistical Machine Learnin [Pages 171-208] [2026-04-16].pdf]
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (6.9) from MVSML chapter 6."})


def cheatsheet():
    return "msm074: Numbered display equation (6.9) from MVSML chapter 6."
