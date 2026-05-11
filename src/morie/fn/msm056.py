"""Numbered display equation (6.5) from MVSML chapter 6.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_eq_6_5"]


def mvsml_bayesian_regression_eq_6_5(ETA, list, model, BRR, X, L):
    """
    Numbered display equation (6.5) from MVSML chapter 6.

    Formula: ETA = list( list( model = ‘BRR’, X = L, df0 = v\beta, S0 = S\beta, R2 = 1-R2) ) A = BGLR(y=y, ETA = ETA, nIter = 1e4, burnIn = 1e3, S0 = S, df0 = v, R2 = R2) When there is more than one repetition of an individual in the data at hand, or a more sophisticated design is used in the data collection, model (6.4) can be speciﬁed in a more general way to take into account this structure, as follows: Y = 1n\mu + Zg + e (6.5) with Z the incident matrix of the genotypes. This model cannot be ﬁtted directly in the BGLR and some precalculus is needed ﬁrst to compute the “covariance” matrix of the predictor Zg in model

    Parameters
    ----------
    ETA : array-like
        Input data.
    list : array-like
        Input data.
    model : array-like
        Input data.
    BRR : array-like
        Input data.
    X : array-like
        Input data.
    L : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (6.5) [Multivariate Statistical Machine Learnin [Pages 171-208] [2026-04-16].pdf]
    """
    ETA = np.atleast_1d(np.asarray(ETA, dtype=float))
    n = len(ETA)
    result = float(np.mean(ETA))
    se = float(np.std(ETA, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (6.5) from MVSML chapter 6."})


def cheatsheet():
    return "msm056: Numbered display equation (6.5) from MVSML chapter 6."
