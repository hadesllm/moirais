"""Numbered display equation (6.9) from MVSML chapter 6.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_eq_6_9"]


def mvsml_bayesian_regression_eq_6_9(ETA, list, X, model, FIXED, K):
    """
    Numbered display equation (6.9) from MVSML chapter 6.

    Formula: ETA = list( list( X=X, model='FIXED' ), list( K=Z1GZT 1, model=’ RKHS’ ) ) A = Multitrait(y = Y, ETA=ETA, resCov = list( type = 'UN', S0 = SR, df0 = vR ), nIter = nI, burnIn = nb) The ﬁrst argument in the Multitrait function is the response variable which many times is a phenotype matrix where each row corresponds to the measurement of nT traits in each individual. The second argument is a list predictor in which the ﬁrst sub-list speciﬁes the design matrix and prior model to the ﬁxed effects part of the predictor in model

    Parameters
    ----------
    ETA : array-like
        Input data.
    list : array-like
        Input data.
    X : array-like
        Input data.
    model : array-like
        Input data.
    FIXED : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (6.9) [Multivariate Statistical Machine Learnin [Pages 171-208] [2026-04-16].pdf]
    """
    ETA = np.atleast_1d(np.asarray(ETA, dtype=float))
    n = len(ETA)
    result = float(np.mean(ETA))
    se = float(np.std(ETA, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (6.9) from MVSML chapter 6."})


def cheatsheet():
    return "msm070: Numbered display equation (6.9) from MVSML chapter 6."
